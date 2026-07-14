"""Hosted monitor delivery extension declarations."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Mapping

from daita.plugins import PluginContext, PluginKind, PluginManifest
from daita.plugins.base import RuntimeExtensionPlugin
from daita.runtime import AccessMode, Capability, Evidence, Operation, RiskLevel, Task


class HostedInAppMonitorDeliveryPlugin(RuntimeExtensionPlugin):
    """Hosted runtime extension for in-app monitor notifications."""

    manifest = PluginManifest(
        id="hosted_monitor_delivery",
        display_name="Hosted Monitor Delivery",
        version="1.0.0",
        kind=PluginKind.RUNTIME_EXTENSION,
        domains=frozenset({"monitor", "notification"}),
    )

    def __init__(
        self,
        *,
        service: Any | None = None,
        service_name: str = "hosted_in_app_notification_service",
    ) -> None:
        self.service = service
        self.service_name = service_name
        self.executor = HostedInAppMonitorDeliveryExecutor(self)

    async def setup(self, context: PluginContext) -> None:
        if self.service is None:
            self.service = context.services.get(self.service_name)

    def declare_capabilities(self) -> tuple[Capability, ...]:
        return (
            Capability(
                id="monitor.delivery.in_app",
                owner=self.manifest.id,
                description="Deliver hosted monitor notifications to the requesting user.",
                domains=frozenset({"monitor", "notification"}),
                operation_types=frozenset({"monitor.delivery"}),
                access=AccessMode.NONE,
                risk=RiskLevel.LOW,
                input_schema={
                    "type": "object",
                    "required": ["delivery_kind", "target", "payload_source"],
                    "properties": {
                        "delivery_kind": {"type": "string"},
                        "target": {"type": "object"},
                        "payload_source": {"type": "object"},
                        "format": {"type": "string"},
                        "subject": {"type": "string"},
                        "idempotency_key": {"type": "string"},
                    },
                },
                output_evidence=frozenset({"hosted.in_app.notification"}),
                executor="hosted_monitor_delivery.deliver_in_app",
                runtime_only=True,
                side_effecting=True,
                replay_safe=True,
                idempotent=True,
                metadata={
                    "monitor_roles": ["delivery"],
                    "delivery_kind": "in_app",
                    "accepted_payload_kinds": ["monitor.report"],
                    "accepted_formats": ["markdown", "plain", "text"],
                    "accepted_target_types": ["requesting_user"],
                    "default_target": {"type": "requesting_user"},
                    "supports_idempotency_key": True,
                },
            ),
        )

    def get_executors(self) -> tuple["HostedInAppMonitorDeliveryExecutor", ...]:
        return (self.executor,)


@dataclass(frozen=True)
class HostedInAppMonitorDeliveryExecutor:
    """Executor adapter for hosted in-app monitor notification services."""

    plugin: HostedInAppMonitorDeliveryPlugin
    id: str = "hosted_monitor_delivery.deliver_in_app"
    capability_ids: frozenset[str] = frozenset({"monitor.delivery.in_app"})

    async def execute(
        self,
        task: Task,
        operation: Operation,
        context: Mapping[str, Any],
    ) -> list[Evidence]:
        service = self.plugin.service
        if service is None:
            raise RuntimeError(
                "hosted in-app monitor delivery requires a registered "
                f"{self.plugin.service_name!r} service"
            )
        result = await _deliver_with_service(service, task.input)
        return [
            Evidence(
                kind="hosted.in_app.notification",
                owner=self.plugin.manifest.id,
                operation_id=operation.id,
                task_id=task.id,
                accepted=True,
                payload={
                    "delivered": True,
                    "delivery_kind": task.input.get("delivery_kind"),
                    "target": dict(task.input.get("target") or {}),
                    "idempotency_key": task.input.get("idempotency_key"),
                    "result": result,
                },
                metadata={
                    "monitor_delivery_kind": task.input.get("delivery_kind"),
                    "monitor_delivery_target": dict(task.input.get("target") or {}),
                    "idempotency_key": task.input.get("idempotency_key"),
                },
            )
        ]


async def _deliver_with_service(
    service: Any,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    for method_name in (
        "deliver_monitor_notification",
        "deliver_in_app_notification",
        "send_in_app_notification",
    ):
        method = getattr(service, method_name, None)
        if method is None:
            continue
        result = method(dict(payload))
        if inspect.isawaitable(result):
            result = await result
        return _result_mapping(result)
    if callable(service):
        result = service(dict(payload))
        if inspect.isawaitable(result):
            result = await result
        return _result_mapping(result)
    raise TypeError(
        "hosted in-app notification service must be callable or implement one of "
        "deliver_monitor_notification, deliver_in_app_notification, or "
        "send_in_app_notification"
    )


def _result_mapping(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, Mapping):
        return dict(result)
    return {"value": result}
