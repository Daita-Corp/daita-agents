"""Atomic admission of explicitly configured extension declarations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import re

from ..capabilities import CapabilityRegistry, Executor
from ..errors import ErrorRetryability, PluginError
from .manifest import ExtensionManifest, _validate_extension_id

_DECLARATION_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)*$")
_MAX_DECLARATION_ID_BYTES = 128


def _validate_declaration_id(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > _MAX_DECLARATION_ID_BYTES
        or _DECLARATION_ID_RE.fullmatch(value) is None
    ):
        raise ValueError(f"{field_name} must be a bounded stable identifier")
    return value


@dataclass(frozen=True, slots=True)
class RegistryDiagnostic:
    """Safe machine-readable extension admission fact."""

    code: str
    extension_id: str
    declaration_kind: str
    declaration_id: str

    def __post_init__(self) -> None:
        _validate_declaration_id(self.code, "registry diagnostic code")
        _validate_extension_id(self.extension_id, "registry diagnostic extension_id")
        _validate_declaration_id(
            self.declaration_kind,
            "registry diagnostic declaration_kind",
        )
        _validate_declaration_id(
            self.declaration_id,
            "registry diagnostic declaration_id",
        )


class ExtensionLoadError(PluginError):
    """Public extension-load failure without extension-controlled error text."""

    def __init__(self, diagnostic: RegistryDiagnostic) -> None:
        if not isinstance(diagnostic, RegistryDiagnostic):
            raise TypeError("extension load error requires a RegistryDiagnostic")
        self.diagnostic = diagnostic
        super().__init__(
            f"{diagnostic.code} for configured extension " f"{diagnostic.extension_id}",
            plugin_id=diagnostic.extension_id,
            error_code="plugin_error",
            retryability=ErrorRetryability.PERMANENT,
        )


@dataclass(frozen=True, slots=True)
class ExtensionRegistration:
    """One validated manifest plus its declared executor implementations."""

    manifest: ExtensionManifest
    executors: tuple[Executor, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ExtensionManifest):
            raise TypeError("extension registration requires an ExtensionManifest")
        executors = tuple(self.executors)
        actual_ids: list[str] = []
        for executor in executors:
            actual_ids.append(
                _validate_declaration_id(
                    getattr(executor, "executor_id", None),
                    "extension executor id",
                )
            )
            if not callable(getattr(executor, "execute", None)):
                raise TypeError("extension executor must provide execute(request)")
        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError("extension executor implementations must be unique")
        if set(actual_ids) != set(self.manifest.declarations.executor_ids):
            raise ValueError(
                "extension executor implementations must exactly match the manifest"
            )
        object.__setattr__(self, "executors", executors)


@dataclass(frozen=True, slots=True)
class ConfiguredExtension:
    """One explicitly enabled, lazily invoked extension factory."""

    id: str
    factory: Callable[[], ExtensionRegistration]

    def __post_init__(self) -> None:
        _validate_extension_id(self.id, "configured extension id")
        if not callable(self.factory):
            raise TypeError("configured extension factory must be callable")


class ExtensionRegistry:
    """Immutable extension registry backed by the capability registry owner."""

    def __init__(
        self,
        registrations: Iterable[ExtensionRegistration] = (),
    ) -> None:
        values = tuple(registrations)
        for registration in values:
            if not isinstance(registration, ExtensionRegistration):
                raise TypeError(
                    "extension registry requires ExtensionRegistration records"
                )

        extension_ids: set[str] = set()
        seen: dict[str, dict[str, str]] = {
            "capability": {},
            "executor": {},
            "tool_view": {},
        }
        capabilities = []
        executors: list[Executor] = []
        tool_views = []
        diagnostics: list[RegistryDiagnostic] = []

        for registration in values:
            manifest = registration.manifest
            if manifest.id in extension_ids:
                raise _load_error(
                    "extension.id_collision",
                    manifest.id,
                    "manifest",
                    manifest.id,
                )
            extension_ids.add(manifest.id)
            diagnostics.append(
                RegistryDiagnostic(
                    code="extension.loaded",
                    extension_id=manifest.id,
                    declaration_kind="manifest",
                    declaration_id=manifest.id,
                )
            )

            for capability in manifest.declarations.capabilities:
                declaration_id = _validate_declaration_id(
                    capability.id,
                    "extension capability id",
                )
                if capability.owner != manifest.id:
                    raise _load_error(
                        "extension.owner_mismatch",
                        manifest.id,
                        "capability",
                        declaration_id,
                    )
                _admit_identity(
                    seen,
                    extension_id=manifest.id,
                    declaration_kind="capability",
                    declaration_id=declaration_id,
                )
                capabilities.append(capability)
                diagnostics.append(
                    _declared_diagnostic(
                        manifest.id,
                        "capability",
                        declaration_id,
                    )
                )

            for executor in registration.executors:
                executor_id = _validate_declaration_id(
                    getattr(executor, "executor_id", None),
                    "extension executor id",
                )
                _admit_identity(
                    seen,
                    extension_id=manifest.id,
                    declaration_kind="executor",
                    declaration_id=executor_id,
                )
                executors.append(executor)
                diagnostics.append(
                    _declared_diagnostic(manifest.id, "executor", executor_id)
                )

            for view in manifest.declarations.tool_views:
                view_name = _validate_declaration_id(
                    view.name,
                    "extension tool view name",
                )
                _admit_identity(
                    seen,
                    extension_id=manifest.id,
                    declaration_kind="tool_view",
                    declaration_id=view_name,
                )
                tool_views.append(view)
                diagnostics.append(
                    _declared_diagnostic(manifest.id, "tool_view", view_name)
                )

        capability_registry: CapabilityRegistry | None = None
        invalid_declarations = False
        try:
            capability_registry = CapabilityRegistry(
                capabilities=capabilities,
                executors=executors,
                tool_views=tool_views,
            )
            for registration in values:
                capability_registry.validate_declarations(
                    registration.manifest.declarations
                )
        except (TypeError, ValueError):
            invalid_declarations = True
        if invalid_declarations:
            extension_id = values[-1].manifest.id if values else "registry"
            raise _load_error(
                "extension.invalid_declarations",
                extension_id,
                "manifest",
                extension_id,
            ) from None
        assert capability_registry is not None

        self._registrations = values
        self._manifest_by_id = {
            registration.manifest.id: registration.manifest for registration in values
        }
        self._capability_registry = capability_registry
        self._diagnostics = tuple(diagnostics)

    @classmethod
    def load(
        cls,
        configured: Iterable[ConfiguredExtension],
    ) -> ExtensionRegistry:
        """Load only the explicit list, returning no partial registry on failure."""

        values = tuple(configured)
        seen_ids: set[str] = set()
        for item in values:
            if not isinstance(item, ConfiguredExtension):
                raise TypeError(
                    "configured extensions must be ConfiguredExtension records"
                )
            if item.id in seen_ids:
                raise _load_error(
                    "extension.id_collision",
                    item.id,
                    "manifest",
                    item.id,
                )
            seen_ids.add(item.id)

        registrations: list[ExtensionRegistration] = []
        for item in values:
            registration: object = None
            load_failed = False
            try:
                registration = item.factory()
            except Exception:
                load_failed = True
            if load_failed:
                raise _load_error(
                    "extension.load_failed",
                    item.id,
                    "manifest",
                    item.id,
                ) from None
            if not isinstance(registration, ExtensionRegistration):
                raise _load_error(
                    "extension.invalid_registration",
                    item.id,
                    "manifest",
                    item.id,
                )
            if registration.manifest.id != item.id:
                raise _load_error(
                    "extension.identity_mismatch",
                    item.id,
                    "manifest",
                    item.id,
                )
            registrations.append(registration)
        return cls(registrations)

    @property
    def extension_ids(self) -> tuple[str, ...]:
        return tuple(self._manifest_by_id)

    @property
    def manifests(self) -> tuple[ExtensionManifest, ...]:
        return tuple(self._manifest_by_id.values())

    @property
    def capability_registry(self) -> CapabilityRegistry:
        return self._capability_registry

    @property
    def diagnostics(self) -> tuple[RegistryDiagnostic, ...]:
        return self._diagnostics

    def manifest(self, extension_id: str) -> ExtensionManifest:
        try:
            return self._manifest_by_id[extension_id]
        except KeyError as error:
            raise KeyError(f"unknown extension: {extension_id}") from error


def _admit_identity(
    seen: dict[str, dict[str, str]],
    *,
    extension_id: str,
    declaration_kind: str,
    declaration_id: str,
) -> None:
    owners = seen[declaration_kind]
    if declaration_id in owners:
        raise _load_error(
            "extension.declaration_collision",
            extension_id,
            declaration_kind,
            declaration_id,
        )
    owners[declaration_id] = extension_id


def _declared_diagnostic(
    extension_id: str,
    declaration_kind: str,
    declaration_id: str,
) -> RegistryDiagnostic:
    return RegistryDiagnostic(
        code="extension.declared",
        extension_id=extension_id,
        declaration_kind=declaration_kind,
        declaration_id=declaration_id,
    )


def _load_error(
    code: str,
    extension_id: str,
    declaration_kind: str,
    declaration_id: str,
) -> ExtensionLoadError:
    return ExtensionLoadError(
        RegistryDiagnostic(
            code=code,
            extension_id=extension_id,
            declaration_kind=declaration_kind,
            declaration_id=declaration_id,
        )
    )


__all__ = [
    "ConfiguredExtension",
    "ExtensionLoadError",
    "ExtensionRegistration",
    "ExtensionRegistry",
    "RegistryDiagnostic",
]
