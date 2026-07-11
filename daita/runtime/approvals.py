"""Approval channel helpers for runtime governance."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace

from .primitives import (
    ApprovalRequest,
    ApprovalStatus,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStore,
)


class InMemoryApprovalChannel:
    """Small approval channel backed by the active runtime store."""

    def __init__(
        self,
        store: RuntimeStore,
        *,
        event_committer: (
            Callable[[ApprovalRequest, RuntimeEvent], Awaitable[None]] | None
        ) = None,
    ) -> None:
        self.store = store
        self._event_committer = event_committer

    def bind_event_committer(
        self,
        committer: Callable[[ApprovalRequest, RuntimeEvent], Awaitable[None]],
    ) -> None:
        """Bind approval event persistence to the active runtime kernel."""
        self._event_committer = committer

    async def request(self, approval: ApprovalRequest) -> ApprovalRequest:
        """Persist a pending approval request.

        Governance enforcement creates approval requests through runtime-store
        transitions so approval state, audit, and blocked lifecycle state commit
        together. This helper remains for tests and non-governance adapters that
        already have a durable request record to register.
        """
        for existing in await self.store.list_approval_requests():
            if existing.approval_id == approval.approval_id:
                return existing
        await self.store.save_approval_request(approval)
        return approval

    async def approve(self, approval_id: str) -> ApprovalRequest:
        """Mark an approval request approved."""
        return await self._set_status(approval_id, ApprovalStatus.APPROVED)

    async def reject(self, approval_id: str) -> ApprovalRequest:
        """Mark an approval request rejected."""
        return await self._set_status(approval_id, ApprovalStatus.REJECTED)

    async def expire(self, approval_id: str) -> ApprovalRequest:
        """Mark an approval request expired."""
        return await self._set_status(approval_id, ApprovalStatus.EXPIRED)

    async def cancel(self, approval_id: str) -> ApprovalRequest:
        """Mark an approval request cancelled."""
        return await self._set_status(approval_id, ApprovalStatus.CANCELLED)

    async def pending(
        self, operation_id: str | None = None
    ) -> tuple[ApprovalRequest, ...]:
        """Return pending approval requests, optionally for one operation."""
        requests = await self.store.list_approval_requests(operation_id)
        return tuple(
            request for request in requests if request.status is ApprovalStatus.PENDING
        )

    async def _set_status(
        self,
        approval_id: str,
        status: ApprovalStatus,
    ) -> ApprovalRequest:
        for request in await self.store.list_approval_requests():
            if request.approval_id == approval_id:
                if request.status is status:
                    return request
                if request.status is not ApprovalStatus.PENDING:
                    raise ValueError(
                        f"approval {approval_id} is already {request.status.value}; "
                        "create a new approval request to renew it"
                    )
                updated = replace(request, status=status)
                operation = await self.store.load_operation(request.operation_id)
                metadata = operation.metadata if operation is not None else {}
                trace_id, span_id = _current_trace_ids()
                event = RuntimeEvent(
                    type=RuntimeEventType.APPROVAL_UPDATED,
                    runtime_id=metadata.get("runtime_id"),
                    runtime_kind=metadata.get("runtime_kind"),
                    operation_id=request.operation_id,
                    policy_id=request.requested_by_policy_id,
                    approval_id=request.approval_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    message=(f"Approval {approval_id} marked {status.value}."),
                    payload={"approval": updated.to_dict()},
                )
                if self._event_committer is not None:
                    await self._event_committer(updated, event)
                else:
                    await self.store.commit_approval_update(updated, event)
                return updated
        raise KeyError(approval_id)


def _current_trace_ids() -> tuple[str | None, str | None]:
    try:
        from daita.core.tracing import get_trace_manager

        context = get_trace_manager().trace_context
        return context.current_trace_id, context.current_span_id
    except Exception:
        return None, None
