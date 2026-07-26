"""Bounded provider-neutral model-cost semantics.

This module deliberately owns no rates or provider schedules.  Providers
produce estimates, routing and the loop aggregate them, and persisted run
results retain the exact estimate that was produced at execution time.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import re
from typing import TypeVar, cast

_MAX_COMPONENTS = 64
_MAX_IDENTIFIER_CHARACTERS = 256
_MAX_DECIMAL_PLACES = 18
_MAX_INTEGER_PLACES = 18
_REASON_CODE = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z")
_T = TypeVar("_T")


class CostEstimateStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class CostBasis(str, Enum):
    PUBLIC_LIST = "public_list"
    CONFIGURED_CONTRACT = "configured_contract"
    PROVIDER_REPORTED = "provider_reported"


@dataclass(frozen=True, slots=True)
class CostComponent:
    """One bounded named subtotal already calculated by an admitted schedule."""

    name: str
    amount_usd: Decimal
    basis: CostBasis | None = None
    rate_schedule_id: str | None = None

    def __post_init__(self) -> None:
        _bounded_identifier(self.name, "cost-component name")
        _bounded_amount(self.amount_usd, "cost-component amount_usd")
        if self.basis is not None and not isinstance(self.basis, CostBasis):
            raise TypeError("cost-component basis must be CostBasis or None")
        if self.rate_schedule_id is not None:
            _bounded_identifier(
                self.rate_schedule_id,
                "cost-component rate_schedule_id",
            )


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """One immutable estimate whose completeness can never be mistaken for zero."""

    amount_usd: Decimal | None
    status: CostEstimateStatus
    basis: CostBasis | None = None
    rate_schedule_id: str | None = None
    components: tuple[CostComponent, ...] = ()
    code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, CostEstimateStatus):
            raise TypeError("cost-estimate status must be CostEstimateStatus")
        if self.basis is not None and not isinstance(self.basis, CostBasis):
            raise TypeError("cost-estimate basis must be CostBasis or None")
        if self.rate_schedule_id is not None:
            _bounded_identifier(
                self.rate_schedule_id,
                "cost-estimate rate_schedule_id",
            )
        components = tuple(self.components)
        if len(components) > _MAX_COMPONENTS:
            raise ValueError("cost-estimate components exceed their bound")
        if any(not isinstance(item, CostComponent) for item in components):
            raise TypeError(
                "cost-estimate components must contain CostComponent records"
            )
        if self.code is not None and (
            not isinstance(self.code, str) or _REASON_CODE.fullmatch(self.code) is None
        ):
            raise ValueError("cost-estimate code must be a bounded reason identifier")

        if self.status is CostEstimateStatus.UNAVAILABLE:
            if self.amount_usd is not None:
                raise ValueError("unavailable cost estimate cannot have an amount")
            if (
                self.basis is not None
                or self.rate_schedule_id is not None
                or components
            ):
                raise ValueError(
                    "unavailable cost estimate cannot claim pricing details"
                )
            if self.code is None:
                raise ValueError("unavailable cost estimate requires a reason code")
        else:
            if self.amount_usd is None:
                raise ValueError(
                    "complete and partial cost estimates require an amount"
                )
            _bounded_amount(self.amount_usd, "cost-estimate amount_usd")
            if self.status is CostEstimateStatus.COMPLETE and self.code is not None:
                raise ValueError(
                    "complete cost estimate cannot have an incomplete code"
                )
            if self.status is CostEstimateStatus.PARTIAL and self.code is None:
                raise ValueError("partial cost estimate requires a reason code")

        object.__setattr__(self, "components", components)

    @classmethod
    def complete(
        cls,
        amount_usd: Decimal,
        *,
        basis: CostBasis | None = None,
        rate_schedule_id: str | None = None,
        components: tuple[CostComponent, ...] = (),
    ) -> CostEstimate:
        return cls(
            amount_usd=amount_usd,
            status=CostEstimateStatus.COMPLETE,
            basis=basis,
            rate_schedule_id=rate_schedule_id,
            components=components,
        )

    @classmethod
    def partial(
        cls,
        amount_usd: Decimal,
        *,
        code: str,
        basis: CostBasis | None = None,
        rate_schedule_id: str | None = None,
        components: tuple[CostComponent, ...] = (),
    ) -> CostEstimate:
        return cls(
            amount_usd=amount_usd,
            status=CostEstimateStatus.PARTIAL,
            basis=basis,
            rate_schedule_id=rate_schedule_id,
            components=components,
            code=code,
        )

    @classmethod
    def unavailable(
        cls,
        code: str = "pricing_schedule_unavailable",
    ) -> CostEstimate:
        return cls(
            amount_usd=None,
            status=CostEstimateStatus.UNAVAILABLE,
            code=code,
        )

    def render(self) -> str:
        """Return the bounded canonical user-facing pricing state."""

        return format_cost_estimate(self)


def aggregate_cost_estimates(estimates: Iterable[CostEstimate]) -> CostEstimate:
    """Aggregate attempt estimates without inventing prices for unknown work."""

    items = tuple(estimates)
    if any(not isinstance(item, CostEstimate) for item in items):
        raise TypeError("cost aggregation requires CostEstimate records")
    if not items:
        return CostEstimate.unavailable("no_model_attempts")

    known = tuple(item for item in items if item.amount_usd is not None)
    if not known:
        return CostEstimate.unavailable(_incomplete_code(items))

    amount = sum(
        (cast(Decimal, item.amount_usd) for item in known),
        Decimal("0"),
    )
    components = _aggregate_components(known)
    basis = _common_optional(item.basis for item in known)
    schedule_id = _common_optional(item.rate_schedule_id for item in known)
    if all(item.status is CostEstimateStatus.COMPLETE for item in items):
        return CostEstimate.complete(
            amount,
            basis=basis,
            rate_schedule_id=schedule_id,
            components=components,
        )
    return CostEstimate.partial(
        amount,
        basis=basis,
        rate_schedule_id=schedule_id,
        components=components,
        code=_incomplete_code(items),
    )


def format_cost_estimate(estimate: CostEstimate) -> str:
    """Render the four user-visible states without turning unknown into ``$0``."""

    if not isinstance(estimate, CostEstimate):
        raise TypeError("estimate must be a CostEstimate")
    if estimate.status is CostEstimateStatus.UNAVAILABLE:
        return "cost unavailable"

    assert estimate.amount_usd is not None
    amount = f"${canonical_decimal(estimate.amount_usd)}"
    if estimate.status is CostEstimateStatus.PARTIAL:
        return f"≥{amount} estimated; some attempts were unpriced"
    basis = {
        CostBasis.PUBLIC_LIST: " at public list rates",
        CostBasis.CONFIGURED_CONTRACT: " using configured contract rates",
        CostBasis.PROVIDER_REPORTED: " as reported by the provider",
        None: "",
    }[estimate.basis]
    if estimate.amount_usd == 0:
        return f"{amount} explicit estimate{basis}"
    return f"{amount} estimated{basis}"


def canonical_decimal(value: Decimal) -> str:
    _bounded_amount(value, "decimal value")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _bounded_identifier(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_IDENTIFIER_CHARACTERS
        or any(character in "\r\n\x00" for character in value)
    ):
        raise ValueError(f"{field_name} must be a bounded single-line identifier")


def _bounded_amount(value: object, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite() or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    exponent = value.as_tuple().exponent
    if value != 0 and (
        value.adjusted() >= _MAX_INTEGER_PLACES
        or not isinstance(exponent, int)
        or exponent < -_MAX_DECIMAL_PLACES
    ):
        raise ValueError(f"{field_name} exceeds its decimal bound")


def _common_optional(values: Iterable[_T]) -> _T | None:
    items = tuple(values)
    first = items[0]
    return first if all(item == first for item in items) else None


def _aggregate_components(
    estimates: tuple[CostEstimate, ...],
) -> tuple[CostComponent, ...]:
    components = tuple(
        CostComponent(
            name=component.name,
            amount_usd=component.amount_usd,
            basis=component.basis or estimate.basis,
            rate_schedule_id=(component.rate_schedule_id or estimate.rate_schedule_id),
        )
        for estimate in estimates
        for component in (
            estimate.components
            or (
                CostComponent(
                    "attempt",
                    cast(Decimal, estimate.amount_usd),
                ),
            )
        )
    )
    if len(components) <= _MAX_COMPONENTS:
        return components

    grouped: dict[tuple[CostBasis | None, str | None], Decimal] = {}
    for component in components:
        key = (component.basis, component.rate_schedule_id)
        grouped[key] = grouped.get(key, Decimal("0")) + component.amount_usd
    if len(grouped) > _MAX_COMPONENTS:
        raise ValueError("aggregate cost-estimate schedules exceed their bound")
    return tuple(
        CostComponent(
            name="aggregated_attempts",
            amount_usd=amount,
            basis=basis,
            rate_schedule_id=schedule_id,
        )
        for (basis, schedule_id), amount in grouped.items()
    )


def _incomplete_code(estimates: tuple[CostEstimate, ...]) -> str:
    codes = tuple(
        dict.fromkeys(
            item.code
            for item in estimates
            if item.status is not CostEstimateStatus.COMPLETE and item.code is not None
        )
    )
    if len(codes) == 1:
        return codes[0]
    return "multiple_incomplete_reasons"


__all__ = [
    "CostBasis",
    "CostComponent",
    "CostEstimate",
    "CostEstimateStatus",
    "aggregate_cost_estimates",
    "canonical_decimal",
    "format_cost_estimate",
]
