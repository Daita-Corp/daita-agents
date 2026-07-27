"""Bounded provider pricing schedules and provider-neutral cost semantics.

Provider adapters own native usage interpretation. This module owns the small
release-reviewed schedule format, strict admission, effective-date selection,
pure ``Decimal`` arithmetic, and immutable estimates persisted with run
history. No pricing data is downloaded at runtime.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from functools import lru_cache
from importlib import resources
import json
import re
from typing import TypeVar, cast
from urllib.parse import urlsplit

_MAX_SCHEDULE_FILE_BYTES = 256 * 1_024
_MAX_SCHEDULES = 128
_MAX_RATES = 32
_MAX_QUALIFIERS = 16
_MAX_MODIFIERS = 16
_MAX_SOURCES = 8
_MAX_COMPONENTS = 64
_MAX_IDENTIFIER_CHARACTERS = 256
_MAX_URL_CHARACTERS = 2_048
_MAX_DECIMAL_PLACES = 18
_MAX_INTEGER_PLACES = 18
_MAX_QUANTITY_INTEGER_PLACES = 24

_REASON_CODE = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z")
_QUALIFIER_KEY = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_DECIMAL_STRING = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_SENSITIVE_QUALIFIER_FRAGMENTS = (
    "account",
    "api_key",
    "credential",
    "customer",
    "organization",
    "project_id",
    "secret",
)

_BILLABLE_METRICS = frozenset(
    {
        "audio_seconds",
        "cache_storage_token_hours",
        "document_pages",
        "grounding_requests",
        "images",
        "input_cache_read_tokens",
        "input_cache_write_1h_tokens",
        "input_cache_write_5m_tokens",
        "input_cache_write_tokens",
        "input_uncached_tokens",
        "output_tokens",
        "server_side_tool_calls",
        "web_search_calls",
    }
)
_COMPONENT_ONLY_METRICS = frozenset({"provider_reported_total"})
_USAGE_RANGE_METRICS = frozenset(
    {
        "request_audio_seconds",
        "request_document_pages",
        "request_image_count",
        "request_input_tokens",
        "request_tool_calls",
    }
)
_UNITS = frozenset(
    {
        "call",
        "image",
        "page",
        "request",
        "second",
        "token",
        "token_hour",
    }
)

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
class PricingQualifier:
    """One exact bounded billing dimension used during schedule selection."""

    name: str
    value: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or _QUALIFIER_KEY.fullmatch(self.name) is None
        ):
            raise ValueError("pricing qualifier name must be a bounded identifier")
        if any(fragment in self.name for fragment in _SENSITIVE_QUALIFIER_FRAGMENTS):
            raise ValueError("pricing qualifier name cannot identify billing accounts")
        _bounded_identifier(self.value, "pricing qualifier value")


@dataclass(frozen=True, slots=True)
class PricingUsageRange:
    """One inclusive whole-request range used to select a flat schedule."""

    metric: str
    minimum_inclusive: int | None = None
    maximum_inclusive: int | None = None

    def __post_init__(self) -> None:
        if self.metric not in _USAGE_RANGE_METRICS:
            raise ValueError("pricing usage range uses an unknown metric")
        for value, field_name in (
            (self.minimum_inclusive, "minimum_inclusive"),
            (self.maximum_inclusive, "maximum_inclusive"),
        ):
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value >= 10**_MAX_QUANTITY_INTEGER_PLACES
            ):
                raise ValueError(f"pricing usage range {field_name} is invalid")
        if (
            self.minimum_inclusive is not None
            and self.maximum_inclusive is not None
            and self.minimum_inclusive > self.maximum_inclusive
        ):
            raise ValueError("pricing usage range is empty")

    def contains(self, value: Decimal) -> bool:
        _bounded_quantity(value, "pricing usage range value")
        return (self.minimum_inclusive is None or value >= self.minimum_inclusive) and (
            self.maximum_inclusive is None or value <= self.maximum_inclusive
        )


@dataclass(frozen=True, slots=True)
class PricingModifier:
    """One named flat multiplier retained in calculated components."""

    name: str
    multiplier: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _REASON_CODE.fullmatch(self.name) is None:
            raise ValueError("pricing modifier name must be a bounded identifier")
        _bounded_amount(self.multiplier, "pricing modifier multiplier")


@dataclass(frozen=True, slots=True)
class PricingRate:
    """One generic metric/quantity/unit price contract."""

    metric: str
    unit: str
    unit_size: int
    price_usd: Decimal

    def __post_init__(self) -> None:
        if self.metric not in _BILLABLE_METRICS:
            raise ValueError("pricing rate uses an unknown billable metric")
        if self.unit not in _UNITS:
            raise ValueError("pricing rate uses an unsupported unit")
        if (
            not isinstance(self.unit_size, int)
            or isinstance(self.unit_size, bool)
            or self.unit_size <= 0
            or self.unit_size >= 10**_MAX_QUANTITY_INTEGER_PLACES
        ):
            raise ValueError(
                "pricing rate unit_size must be a bounded positive integer"
            )
        _bounded_amount(self.price_usd, "pricing rate price_usd")


@dataclass(frozen=True, slots=True)
class PricingSource:
    """One bounded review source without credentials or account identifiers."""

    purpose: str
    url: str
    reviewed_on: date

    def __post_init__(self) -> None:
        if (
            not isinstance(self.purpose, str)
            or _REASON_CODE.fullmatch(self.purpose) is None
        ):
            raise ValueError("pricing source purpose must be a bounded identifier")
        if not isinstance(self.reviewed_on, date) or isinstance(
            self.reviewed_on, datetime
        ):
            raise TypeError("pricing source reviewed_on must be a date")
        _validate_source_url(self.url)


@dataclass(frozen=True, slots=True)
class PricingSchedule:
    """One exact immutable provider/model/endpoint pricing period."""

    schedule_id: str
    basis: CostBasis
    provider: str
    model: str
    endpoint: str
    effective_from: datetime
    effective_until: datetime | None
    qualifiers: tuple[PricingQualifier, ...]
    usage_range: PricingUsageRange | None
    rates: tuple[PricingRate, ...]
    modifiers: tuple[PricingModifier, ...]
    sources: tuple[PricingSource, ...]

    def __post_init__(self) -> None:
        _bounded_identifier(self.schedule_id, "pricing schedule_id")
        _bounded_identifier(self.provider, "pricing provider")
        _bounded_identifier(self.model, "pricing model")
        _bounded_identifier(self.endpoint, "pricing endpoint")
        if self.basis not in {
            CostBasis.PUBLIC_LIST,
            CostBasis.CONFIGURED_CONTRACT,
        }:
            raise ValueError("pricing schedules use an unsupported basis")
        _require_utc_datetime(self.effective_from, "pricing effective_from")
        if self.effective_until is not None:
            _require_utc_datetime(self.effective_until, "pricing effective_until")
            if self.effective_until <= self.effective_from:
                raise ValueError("pricing effective period must be non-empty")

        qualifiers = tuple(sorted(self.qualifiers, key=lambda item: item.name))
        rates = tuple(self.rates)
        modifiers = tuple(self.modifiers)
        sources = tuple(self.sources)
        _bounded_record_array(
            qualifiers,
            PricingQualifier,
            _MAX_QUALIFIERS,
            "pricing qualifiers",
        )
        _bounded_record_array(rates, PricingRate, _MAX_RATES, "pricing rates")
        _bounded_record_array(
            modifiers,
            PricingModifier,
            _MAX_MODIFIERS,
            "pricing modifiers",
        )
        _bounded_record_array(
            sources,
            PricingSource,
            _MAX_SOURCES,
            "pricing sources",
        )
        if not rates:
            raise ValueError("pricing schedule requires rates")
        if len({item.name for item in qualifiers}) != len(qualifiers):
            raise ValueError("pricing schedule qualifier names cannot repeat")
        if len({item.metric for item in rates}) != len(rates):
            raise ValueError("pricing schedule rate metrics cannot repeat")
        if len({item.name for item in modifiers}) != len(modifiers):
            raise ValueError("pricing schedule modifier names cannot repeat")
        if self.usage_range is not None and not isinstance(
            self.usage_range, PricingUsageRange
        ):
            raise TypeError("pricing schedule usage_range is invalid")
        object.__setattr__(self, "qualifiers", qualifiers)
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "modifiers", modifiers)
        object.__setattr__(self, "sources", sources)


@dataclass(frozen=True, slots=True)
class BillableQuantity:
    """One provider-normalized mutually exclusive metered quantity."""

    metric: str
    quantity: Decimal
    unit: str

    def __post_init__(self) -> None:
        if self.metric not in _BILLABLE_METRICS:
            raise ValueError("billable quantity uses an unknown metric")
        _bounded_quantity(self.quantity, "billable quantity")
        if self.unit not in _UNITS:
            raise ValueError("billable quantity uses an unsupported unit")


@dataclass(frozen=True, slots=True)
class CostComponent:
    """One bounded auditable subtotal produced by a schedule or provider charge."""

    name: str
    amount_usd: Decimal
    basis: CostBasis | None = None
    rate_schedule_id: str | None = None
    metric: str | None = None
    quantity: Decimal | None = None
    unit: str | None = None
    unit_size: int | None = None
    rate_usd: Decimal | None = None
    usage_range: PricingUsageRange | None = None
    modifiers: tuple[PricingModifier, ...] = ()

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
        modifiers = tuple(self.modifiers)
        _bounded_record_array(
            modifiers,
            PricingModifier,
            _MAX_MODIFIERS,
            "cost-component modifiers",
        )
        details = (
            self.metric,
            self.quantity,
            self.unit,
            self.unit_size,
            self.rate_usd,
        )
        if all(item is None for item in details):
            if self.usage_range is not None or modifiers:
                raise ValueError(
                    "cost-component range and modifiers require arithmetic details"
                )
        elif any(item is None for item in details):
            raise ValueError("cost-component arithmetic details must be complete")
        else:
            assert self.metric is not None
            assert self.quantity is not None
            assert self.unit is not None
            assert self.unit_size is not None
            assert self.rate_usd is not None
            if self.metric not in _BILLABLE_METRICS | _COMPONENT_ONLY_METRICS:
                raise ValueError("cost-component metric is unknown")
            _bounded_quantity(self.quantity, "cost-component quantity")
            if self.unit not in _UNITS:
                raise ValueError("cost-component unit is unsupported")
            if (
                not isinstance(self.unit_size, int)
                or isinstance(self.unit_size, bool)
                or self.unit_size <= 0
            ):
                raise ValueError("cost-component unit_size must be positive")
            _bounded_amount(self.rate_usd, "cost-component rate_usd")
            if self.usage_range is not None and not isinstance(
                self.usage_range, PricingUsageRange
            ):
                raise TypeError("cost-component usage_range is invalid")
            expected = self.quantity / Decimal(self.unit_size) * self.rate_usd
            for modifier in modifiers:
                expected *= modifier.multiplier
            if expected != self.amount_usd:
                raise ValueError(
                    "cost-component subtotal does not match its arithmetic"
                )
            if self.basis is CostBasis.PROVIDER_REPORTED:
                if self.rate_schedule_id is not None:
                    raise ValueError(
                        "provider-reported component cannot claim a local schedule"
                    )
            elif (
                self.basis
                in {
                    CostBasis.PUBLIC_LIST,
                    CostBasis.CONFIGURED_CONTRACT,
                }
                and self.rate_schedule_id is None
            ):
                raise ValueError(
                    "schedule-calculated component requires a rate_schedule_id"
                )
        object.__setattr__(self, "modifiers", modifiers)


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
        if (
            self.basis is CostBasis.PROVIDER_REPORTED
            and self.rate_schedule_id is not None
        ):
            raise ValueError("provider-reported estimate cannot claim a local schedule")
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
            if (
                components
                and sum(
                    (component.amount_usd for component in components),
                    Decimal("0"),
                )
                != self.amount_usd
            ):
                raise ValueError(
                    "cost-estimate amount must equal its component subtotals"
                )
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


def parse_pricing_schedules(
    data: bytes | str,
) -> tuple[PricingSchedule, ...]:
    """Parse and strictly validate one bounded schedule document."""

    if isinstance(data, str):
        encoded = data.encode("utf-8")
    elif isinstance(data, bytes):
        encoded = data
    else:
        raise TypeError("pricing schedule document must be bytes or text")
    if len(encoded) > _MAX_SCHEDULE_FILE_BYTES:
        raise ValueError("pricing schedule document exceeds its size bound")
    try:
        decoded = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("pricing schedule document must be UTF-8") from error
    try:
        raw = json.loads(
            decoded,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("pricing schedule document is malformed JSON") from error
    document = _strict_object(raw, {"schedules"}, "pricing schedule document")
    raw_schedules = _strict_array(
        document["schedules"],
        _MAX_SCHEDULES,
        "pricing schedules",
    )
    schedules = tuple(_parse_schedule(item) for item in raw_schedules)
    return validate_pricing_schedules(schedules)


@lru_cache(maxsize=1)
def load_bundled_pricing_schedules() -> tuple[PricingSchedule, ...]:
    """Load the one release-reviewed package data file lazily."""

    data = resources.files("daita.llm").joinpath("pricing_schedules.json").read_bytes()
    return parse_pricing_schedules(data)


def validate_pricing_schedules(
    schedules: Iterable[PricingSchedule],
) -> tuple[PricingSchedule, ...]:
    """Reject duplicate identities and any pair that can match one request."""

    items = tuple(schedules)
    _bounded_record_array(
        items,
        PricingSchedule,
        _MAX_SCHEDULES,
        "pricing schedules",
    )
    ids = tuple(item.schedule_id for item in items)
    if len(ids) != len(set(ids)):
        raise ValueError("pricing schedule IDs cannot repeat")
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if _schedules_overlap(left, right):
                raise ValueError("pricing schedules overlap ambiguously")
    return items


def select_pricing_schedule(
    schedules: Iterable[PricingSchedule],
    *,
    provider: str,
    model: str,
    endpoint: str,
    requested_at: datetime,
    qualifiers: Mapping[str, str] | Iterable[PricingQualifier] = (),
    usage_values: Mapping[str, Decimal] = {},
) -> PricingSchedule | None:
    """Select the only exact schedule effective for one provider request."""

    items = validate_pricing_schedules(schedules)
    _bounded_identifier(provider, "pricing selection provider")
    _bounded_identifier(model, "pricing selection model")
    _bounded_identifier(endpoint, "pricing selection endpoint")
    _require_utc_datetime(requested_at, "pricing selection requested_at")
    normalized_qualifiers = _normalize_qualifiers(qualifiers)
    normalized_usage = _normalize_usage_values(usage_values)
    matches = tuple(
        schedule
        for schedule in items
        if schedule.provider == provider
        and schedule.model == model
        and schedule.endpoint == endpoint
        and schedule.qualifiers == normalized_qualifiers
        and schedule.effective_from <= requested_at
        and (
            schedule.effective_until is None or requested_at < schedule.effective_until
        )
        and _usage_range_matches(schedule.usage_range, normalized_usage)
    )
    if len(matches) > 1:
        raise ValueError("pricing schedule selection is ambiguous")
    return matches[0] if matches else None


def has_complete_pricing_coverage(
    schedules: Iterable[PricingSchedule],
    *,
    provider: str,
    model: str,
    endpoint: str,
    requested_at: datetime,
    qualifiers: Mapping[str, str] | Iterable[PricingQualifier] = (),
    required_metrics: Iterable[str],
    usage_range_metric: str,
) -> bool:
    """Fail closed unless effective schedules cover every non-negative range."""

    try:
        items = validate_pricing_schedules(schedules)
        _require_utc_datetime(requested_at, "pricing coverage requested_at")
        normalized_qualifiers = _normalize_qualifiers(qualifiers)
        required = frozenset(required_metrics)
        if (
            not required
            or not required <= _BILLABLE_METRICS
            or usage_range_metric not in _USAGE_RANGE_METRICS
        ):
            return False
        candidates = tuple(
            item
            for item in items
            if item.provider == provider
            and item.model == model
            and item.endpoint == endpoint
            and item.qualifiers == normalized_qualifiers
            and item.effective_from <= requested_at
            and (item.effective_until is None or requested_at < item.effective_until)
            and required <= {rate.metric for rate in item.rates}
        )
        if not candidates:
            return False
        ranges = tuple(item.usage_range for item in candidates)
        if any(item is None for item in ranges):
            return len(candidates) == 1
        typed_ranges = tuple(cast(PricingUsageRange, item) for item in ranges)
        if any(item.metric != usage_range_metric for item in typed_ranges):
            return False
        ordered = sorted(
            typed_ranges,
            key=lambda item: (
                0 if item.minimum_inclusive is None else item.minimum_inclusive
            ),
        )
        expected_minimum = 0
        for usage_range in ordered:
            minimum = (
                0
                if usage_range.minimum_inclusive is None
                else usage_range.minimum_inclusive
            )
            if minimum != expected_minimum:
                return False
            if usage_range.maximum_inclusive is None:
                return usage_range is ordered[-1]
            expected_minimum = usage_range.maximum_inclusive + 1
        return False
    except (TypeError, ValueError):
        return False


def calculate_cost_estimate(
    schedules: Iterable[PricingSchedule],
    *,
    provider: str,
    model: str,
    endpoint: str,
    requested_at: datetime,
    quantities: Iterable[BillableQuantity],
    qualifiers: Mapping[str, str] | Iterable[PricingQualifier] = (),
    usage_values: Mapping[str, Decimal] = {},
) -> CostEstimate:
    """Calculate every admitted quantity exactly, retaining component arithmetic."""

    items = tuple(quantities)
    if not items:
        return CostEstimate.unavailable("usage_unavailable")
    if len(items) > _MAX_COMPONENTS or any(
        not isinstance(item, BillableQuantity) for item in items
    ):
        return CostEstimate.unavailable("billable_quantities_invalid")
    if len({item.metric for item in items}) != len(items):
        return CostEstimate.unavailable("billable_quantities_inconsistent")
    try:
        schedule = select_pricing_schedule(
            schedules,
            provider=provider,
            model=model,
            endpoint=endpoint,
            requested_at=requested_at,
            qualifiers=qualifiers,
            usage_values=usage_values,
        )
    except (TypeError, ValueError):
        return CostEstimate.unavailable("pricing_dimensions_invalid")
    if schedule is None:
        return CostEstimate.unavailable("pricing_schedule_unavailable")

    rates = {rate.metric: rate for rate in schedule.rates}
    components: list[CostComponent] = []
    missing = False
    for quantity in items:
        rate = rates.get(quantity.metric)
        if rate is None:
            missing = True
            continue
        if rate.unit != quantity.unit:
            missing = True
            continue
        subtotal = quantity.quantity / Decimal(rate.unit_size) * rate.price_usd
        for modifier in schedule.modifiers:
            subtotal *= modifier.multiplier
        components.append(
            CostComponent(
                name=quantity.metric,
                amount_usd=subtotal,
                basis=schedule.basis,
                rate_schedule_id=schedule.schedule_id,
                metric=quantity.metric,
                quantity=quantity.quantity,
                unit=quantity.unit,
                unit_size=rate.unit_size,
                rate_usd=rate.price_usd,
                usage_range=schedule.usage_range,
                modifiers=schedule.modifiers,
            )
        )
    if not components:
        return CostEstimate.unavailable("pricing_rate_unavailable")
    amount = sum(
        (component.amount_usd for component in components),
        Decimal("0"),
    )
    if missing:
        return CostEstimate.partial(
            amount,
            code="pricing_rate_unavailable",
            basis=schedule.basis,
            rate_schedule_id=schedule.schedule_id,
            components=tuple(components),
        )
    return CostEstimate.complete(
        amount,
        basis=schedule.basis,
        rate_schedule_id=schedule.schedule_id,
        components=tuple(components),
    )


def provider_reported_cost_estimate(
    amount_usd: Decimal,
    *,
    currency: str,
    unit: str,
    components: Sequence[CostComponent] = (),
) -> CostEstimate:
    """Admit one authoritative bounded per-request USD provider charge."""

    _bounded_amount(amount_usd, "provider-reported amount_usd")
    if currency != "USD":
        raise ValueError("provider-reported charge must use USD")
    if unit != "request":
        raise ValueError("provider-reported charge must be per request")
    admitted = tuple(components)
    if admitted:
        if len(admitted) > _MAX_COMPONENTS or any(
            not isinstance(item, CostComponent) for item in admitted
        ):
            raise ValueError("provider-reported components are invalid")
        if any(
            item.basis not in {None, CostBasis.PROVIDER_REPORTED}
            or item.rate_schedule_id is not None
            for item in admitted
        ):
            raise ValueError("provider-reported components cannot claim schedules")
        admitted = tuple(
            replace(item, basis=CostBasis.PROVIDER_REPORTED) for item in admitted
        )
        if sum((item.amount_usd for item in admitted), Decimal("0")) != amount_usd:
            raise ValueError("provider-reported components do not equal total")
    else:
        admitted = (
            CostComponent(
                name="provider_reported_total",
                amount_usd=amount_usd,
                basis=CostBasis.PROVIDER_REPORTED,
                metric="provider_reported_total",
                quantity=Decimal("1"),
                unit="request",
                unit_size=1,
                rate_usd=amount_usd,
            ),
        )
    return CostEstimate.complete(
        amount_usd,
        basis=CostBasis.PROVIDER_REPORTED,
        components=admitted,
    )


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
    """Render pricing states without turning unknown work into ``$0``."""

    if not isinstance(estimate, CostEstimate):
        raise TypeError("estimate must be a CostEstimate")
    if estimate.status is CostEstimateStatus.UNAVAILABLE:
        return "cost unavailable"

    assert estimate.amount_usd is not None
    amount = f"${canonical_decimal(estimate.amount_usd)}"
    if estimate.status is CostEstimateStatus.PARTIAL:
        return f"≥{amount} estimated; some attempts were unpriced"
    if estimate.basis is CostBasis.PROVIDER_REPORTED:
        suffix = "; local compute not estimated" if estimate.amount_usd == 0 else ""
        return f"provider API charge {amount}{suffix}"
    basis = {
        CostBasis.PUBLIC_LIST: " at public list rates",
        CostBasis.CONFIGURED_CONTRACT: " using configured contract rates",
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


def _parse_schedule(value: object) -> PricingSchedule:
    raw = _strict_object(
        value,
        {
            "schedule_id",
            "basis",
            "provider",
            "model",
            "endpoint",
            "effective_from",
            "effective_until",
            "qualifiers",
            "usage_range",
            "rates",
            "modifiers",
            "sources",
        },
        "pricing schedule",
    )
    try:
        basis = CostBasis(_required_text(raw["basis"], "pricing basis"))
    except ValueError as error:
        raise ValueError("pricing schedule basis is unsupported") from error
    qualifiers_object = _strict_mapping(
        raw["qualifiers"],
        _MAX_QUALIFIERS,
        "pricing qualifiers",
    )
    qualifiers = tuple(
        PricingQualifier(
            _required_text(name, "pricing qualifier name"),
            _required_text(item, "pricing qualifier value"),
        )
        for name, item in qualifiers_object.items()
    )
    usage_range = (
        None if raw["usage_range"] is None else _parse_usage_range(raw["usage_range"])
    )
    rates = tuple(
        _parse_rate(item)
        for item in _strict_array(raw["rates"], _MAX_RATES, "pricing rates")
    )
    modifiers = tuple(
        _parse_modifier(item)
        for item in _strict_array(
            raw["modifiers"],
            _MAX_MODIFIERS,
            "pricing modifiers",
        )
    )
    sources = tuple(
        _parse_source(item)
        for item in _strict_array(raw["sources"], _MAX_SOURCES, "pricing sources")
    )
    return PricingSchedule(
        schedule_id=_required_text(raw["schedule_id"], "pricing schedule_id"),
        basis=basis,
        provider=_required_text(raw["provider"], "pricing provider"),
        model=_required_text(raw["model"], "pricing model"),
        endpoint=_required_text(raw["endpoint"], "pricing endpoint"),
        effective_from=_parse_timestamp(raw["effective_from"], "effective_from"),
        effective_until=(
            None
            if raw["effective_until"] is None
            else _parse_timestamp(raw["effective_until"], "effective_until")
        ),
        qualifiers=qualifiers,
        usage_range=usage_range,
        rates=rates,
        modifiers=modifiers,
        sources=sources,
    )


def _parse_usage_range(value: object) -> PricingUsageRange:
    raw = _strict_object(
        value,
        {"metric", "minimum_inclusive", "maximum_inclusive"},
        "pricing usage_range",
    )
    return PricingUsageRange(
        metric=_required_text(raw["metric"], "pricing usage range metric"),
        minimum_inclusive=_optional_bounded_int(
            raw["minimum_inclusive"],
            "pricing usage range minimum",
        ),
        maximum_inclusive=_optional_bounded_int(
            raw["maximum_inclusive"],
            "pricing usage range maximum",
        ),
    )


def _parse_rate(value: object) -> PricingRate:
    raw = _strict_object(
        value,
        {"metric", "unit", "unit_size", "price_usd"},
        "pricing rate",
    )
    return PricingRate(
        metric=_required_text(raw["metric"], "pricing rate metric"),
        unit=_required_text(raw["unit"], "pricing rate unit"),
        unit_size=_positive_bounded_int(raw["unit_size"], "pricing rate unit_size"),
        price_usd=_decimal_from_string(raw["price_usd"], "pricing rate price_usd"),
    )


def _parse_modifier(value: object) -> PricingModifier:
    raw = _strict_object(
        value,
        {"name", "multiplier"},
        "pricing modifier",
    )
    return PricingModifier(
        name=_required_text(raw["name"], "pricing modifier name"),
        multiplier=_decimal_from_string(
            raw["multiplier"],
            "pricing modifier multiplier",
        ),
    )


def _parse_source(value: object) -> PricingSource:
    raw = _strict_object(
        value,
        {"purpose", "url", "reviewed_on"},
        "pricing source",
    )
    reviewed_value = _required_text(raw["reviewed_on"], "pricing source reviewed_on")
    try:
        reviewed_on = date.fromisoformat(reviewed_value)
    except ValueError as error:
        raise ValueError("pricing source reviewed_on is malformed") from error
    if reviewed_on.isoformat() != reviewed_value:
        raise ValueError("pricing source reviewed_on must use YYYY-MM-DD")
    return PricingSource(
        purpose=_required_text(raw["purpose"], "pricing source purpose"),
        url=_required_text(raw["url"], "pricing source URL"),
        reviewed_on=reviewed_on,
    )


def _parse_timestamp(value: object, field_name: str) -> datetime:
    text = _required_text(value, f"pricing {field_name}")
    if not text.endswith("Z"):
        raise ValueError(f"pricing {field_name} must be UTC with a Z suffix")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ValueError(f"pricing {field_name} is malformed") from error
    return parsed


def _decimal_from_string(value: object, field_name: str) -> Decimal:
    if not isinstance(value, str) or _DECIMAL_STRING.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a non-negative decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{field_name} is malformed") from error
    _bounded_amount(parsed, field_name)
    return parsed


def _strict_object(
    value: object,
    fields: set[str],
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{field_name} must contain exactly its documented fields")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be text")
    return cast(dict[str, object], value)


def _strict_mapping(
    value: object,
    maximum: int,
    field_name: str,
) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or len(value) > maximum
        or any(not isinstance(key, str) for key in value)
    ):
        raise ValueError(f"{field_name} must be a bounded object")
    return cast(dict[str, object], value)


def _strict_array(value: object, maximum: int, field_name: str) -> list[object]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{field_name} must be a bounded array")
    return value


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("pricing schedule JSON object keys cannot repeat")
        result[key] = value
    return result


def _reject_json_float(value: str) -> object:
    raise ValueError("pricing schedule JSON numbers cannot use binary-float syntax")


def _reject_json_constant(value: str) -> object:
    raise ValueError("pricing schedule JSON constants are unsupported")


def _required_text(value: object, field_name: str) -> str:
    _bounded_identifier(value, field_name)
    return cast(str, value)


def _optional_bounded_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value >= 10**_MAX_QUANTITY_INTEGER_PLACES
    ):
        raise ValueError(f"{field_name} must be a bounded non-negative integer")
    return value


def _positive_bounded_int(value: object, field_name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        or value >= 10**_MAX_QUANTITY_INTEGER_PLACES
    ):
        raise ValueError(f"{field_name} must be a bounded positive integer")
    return value


def _bounded_record_array(
    items: tuple[object, ...],
    record_type: type[object],
    maximum: int,
    field_name: str,
) -> None:
    if len(items) > maximum or any(not isinstance(item, record_type) for item in items):
        raise ValueError(f"{field_name} must contain bounded immutable records")


def _normalize_qualifiers(
    qualifiers: Mapping[str, str] | Iterable[PricingQualifier],
) -> tuple[PricingQualifier, ...]:
    items: tuple[PricingQualifier, ...]
    if isinstance(qualifiers, Mapping):
        mapping = cast(Mapping[str, str], qualifiers)
        items = tuple(PricingQualifier(name, value) for name, value in mapping.items())
    else:
        items = tuple(cast(Iterable[PricingQualifier], qualifiers))
    _bounded_record_array(
        items,
        PricingQualifier,
        _MAX_QUALIFIERS,
        "pricing selection qualifiers",
    )
    if len({item.name for item in items}) != len(items):
        raise ValueError("pricing selection qualifiers cannot repeat")
    return tuple(sorted(items, key=lambda item: item.name))


def _normalize_usage_values(
    values: Mapping[str, Decimal],
) -> dict[str, Decimal]:
    if not isinstance(values, Mapping) or len(values) > _MAX_QUALIFIERS:
        raise ValueError("pricing usage values must be a bounded mapping")
    result: dict[str, Decimal] = {}
    for metric, value in values.items():
        if metric not in _USAGE_RANGE_METRICS:
            raise ValueError("pricing usage values use an unknown metric")
        _bounded_quantity(value, "pricing usage value")
        result[metric] = value
    return result


def _usage_range_matches(
    usage_range: PricingUsageRange | None,
    values: Mapping[str, Decimal],
) -> bool:
    if usage_range is None:
        return True
    value = values.get(usage_range.metric)
    return value is not None and usage_range.contains(value)


def _schedules_overlap(left: PricingSchedule, right: PricingSchedule) -> bool:
    if (
        left.provider,
        left.model,
        left.endpoint,
        left.qualifiers,
    ) != (
        right.provider,
        right.model,
        right.endpoint,
        right.qualifiers,
    ):
        return False
    if not _effective_periods_overlap(left, right):
        return False
    return _usage_ranges_overlap(left.usage_range, right.usage_range)


def _effective_periods_overlap(
    left: PricingSchedule,
    right: PricingSchedule,
) -> bool:
    left_until = left.effective_until or datetime.max.replace(tzinfo=timezone.utc)
    right_until = right.effective_until or datetime.max.replace(tzinfo=timezone.utc)
    return left.effective_from < right_until and right.effective_from < left_until


def _usage_ranges_overlap(
    left: PricingUsageRange | None,
    right: PricingUsageRange | None,
) -> bool:
    if left is None or right is None:
        return True
    if left.metric != right.metric:
        return True
    left_minimum = 0 if left.minimum_inclusive is None else left.minimum_inclusive
    right_minimum = 0 if right.minimum_inclusive is None else right.minimum_inclusive
    left_maximum = (
        10**_MAX_QUANTITY_INTEGER_PLACES
        if left.maximum_inclusive is None
        else left.maximum_inclusive
    )
    right_maximum = (
        10**_MAX_QUANTITY_INTEGER_PLACES
        if right.maximum_inclusive is None
        else right.maximum_inclusive
    )
    return left_minimum <= right_maximum and right_minimum <= left_maximum


def _validate_source_url(value: object) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_URL_CHARACTERS:
        raise ValueError("pricing source URL must be bounded")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "pricing source URL must be credential-free HTTPS without query data"
        )


def _require_utc_datetime(value: object, field_name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError(f"{field_name} must be timezone-aware UTC")


def _bounded_identifier(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_IDENTIFIER_CHARACTERS
        or any(character in "\r\n\x00" for character in value)
    ):
        raise ValueError(f"{field_name} must be a bounded single-line identifier")


def _bounded_amount(value: object, field_name: str) -> None:
    _bounded_decimal(
        value,
        field_name,
        maximum_integer_places=_MAX_INTEGER_PLACES,
    )


def _bounded_quantity(value: object, field_name: str) -> None:
    _bounded_decimal(
        value,
        field_name,
        maximum_integer_places=_MAX_QUANTITY_INTEGER_PLACES,
    )


def _bounded_decimal(
    value: object,
    field_name: str,
    *,
    maximum_integer_places: int,
) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite() or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    exponent = value.as_tuple().exponent
    if value != 0 and (
        value.adjusted() >= maximum_integer_places
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
        replace(
            component,
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

    grouped: dict[
        tuple[
            str,
            CostBasis | None,
            str | None,
            str | None,
            str | None,
            int | None,
            Decimal | None,
            PricingUsageRange | None,
            tuple[PricingModifier, ...],
        ],
        tuple[Decimal, Decimal | None],
    ] = {}
    for component in components:
        key = (
            component.name,
            component.basis,
            component.rate_schedule_id,
            component.metric,
            component.unit,
            component.unit_size,
            component.rate_usd,
            component.usage_range,
            component.modifiers,
        )
        amount, quantity = grouped.get(key, (Decimal("0"), None))
        grouped[key] = (
            amount + component.amount_usd,
            (
                None
                if component.quantity is None
                else (quantity or Decimal("0")) + component.quantity
            ),
        )
    if len(grouped) > _MAX_COMPONENTS:
        raise ValueError("aggregate cost-estimate components exceed their bound")
    return tuple(
        CostComponent(
            name=key[0],
            amount_usd=amount,
            basis=key[1],
            rate_schedule_id=key[2],
            metric=key[3],
            quantity=quantity,
            unit=key[4],
            unit_size=key[5],
            rate_usd=key[6],
            usage_range=key[7],
            modifiers=key[8],
        )
        for key, (amount, quantity) in grouped.items()
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
    "BillableQuantity",
    "CostBasis",
    "CostComponent",
    "CostEstimate",
    "CostEstimateStatus",
    "PricingModifier",
    "PricingQualifier",
    "PricingRate",
    "PricingSchedule",
    "PricingSource",
    "PricingUsageRange",
    "aggregate_cost_estimates",
    "calculate_cost_estimate",
    "canonical_decimal",
    "format_cost_estimate",
    "has_complete_pricing_coverage",
    "load_bundled_pricing_schedules",
    "parse_pricing_schedules",
    "provider_reported_cost_estimate",
    "select_pricing_schedule",
    "validate_pricing_schedules",
]
