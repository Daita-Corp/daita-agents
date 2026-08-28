"""Phase 3 vertical locks for bound UTF-8 edit artifacts and exact publication."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

import daita.artifacts.delivery as delivery_module
from daita import Agent, ApprovalDecision, ApprovalRequest, LocalWorkspace
from daita.adapters.local_workspace import LocalWorkspaceBackend
from daita.artifacts.models import (
    ArtifactDeliveryMode,
    ArtifactDeliveryOutcome,
)
from daita.artifacts.renderers import apply_bounded_text_edits
from daita.artifacts.store import AgentHomeArtifactStore
from daita.capabilities import AccessMode, ExecutionScope, OperationalEffect
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ToolCall,
    ToolResultBlock,
)
from daita.loop.models import (
    InstructionAuthority,
    RunInput,
    RunOrigin,
    RunStartEnvelope,
)


class _EditWorkflowProvider:
    provider_id = "mock:local-text-edit"

    def __init__(
        self,
        replacements: tuple[dict[str, object], ...],
        *,
        after_read: Callable[[], None] | None = None,
        after_edit: Callable[[], None] | None = None,
        target_path: str = "config.yaml",
        save: bool = True,
    ) -> None:
        self.replacements = replacements
        self.after_read = after_read
        self.after_edit = after_edit
        self.target_path = target_path
        self.save = save
        self.requests: list[ModelRequest] = []
        self._load_count = 0
        self._after_edit_called = False

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return True

    def has_complete_pricing(self, request: ModelRequest) -> bool:
        return False

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        results = _results(request)
        visible = {tool.name for tool in request.tools}
        read = results.get("read")
        if read is None:
            return _call("read", "file_read", {"path": self.target_path})
        if read.is_error:
            return _stop("read failed")
        if self.after_read is not None:
            after_read = self.after_read
            self.after_read = None
            after_read()
        edit = results.get("edit")
        if edit is None:
            if "artifact_edit_text" not in visible:
                return self._load("artifact_edit_text")
            data = read.output.get("data")
            assert isinstance(data, Mapping)
            binding = data.get("binding")
            assert isinstance(binding, str)
            return _call(
                "edit",
                "artifact_edit_text",
                {"binding": binding, "replacements": self.replacements},
            )
        if edit.is_error:
            return _stop("edit failed")
        if not self.save:
            return _stop("edit prepared")
        if self.after_edit is not None and not self._after_edit_called:
            self._after_edit_called = True
            self.after_edit()
        save = results.get("save")
        if save is None:
            if "artifact_save_local" not in visible:
                return self._load("artifact_save_local")
            artifact = edit.output.get("artifact")
            assert isinstance(artifact, Mapping)
            artifact_id = artifact.get("artifact_id")
            assert isinstance(artifact_id, str)
            return _call(
                "save",
                "artifact_save_local",
                {"artifact_id": artifact_id, "mode": "replace_bound_file"},
            )
        return _stop("done")

    def _load(self, tool_name: str) -> ModelResponse:
        self._load_count += 1
        return _call(
            f"load-{self._load_count}",
            "toolbox_load",
            {"tool_names": [tool_name]},
        )


def _profile(provider: _EditWorkflowProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=128_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=False,
    )


def _call(call_id: str, name: str, arguments: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
    )


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text)


def _results(request: ModelRequest) -> dict[str, ToolResultBlock]:
    return {
        block.call_id: block
        for message in request.messages
        if message.role is MessageRole.TOOL
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }


def _ids() -> Callable[[str], str]:
    counts: dict[str, int] = {}

    def create(prefix: str) -> str:
        counts[prefix] = counts.get(prefix, 0) + 1
        if prefix in {"run", "conversation", "artifact", "destination"}:
            return f"{prefix}-{counts[prefix]:032x}"
        return f"{prefix}-{counts[prefix]}"

    return create


async def _agent(
    tmp_path: Path,
    provider: _EditWorkflowProvider,
    *,
    approval: Callable[[ApprovalRequest], object] | None = None,
) -> tuple[Agent, Path]:
    workspace = tmp_path / "workspace"
    downloads = tmp_path / "downloads"
    state = tmp_path / "state"
    workspace.mkdir()
    downloads.mkdir()
    handler = approval
    agent = await Agent.create(
        "local-edit",
        workspace=LocalWorkspace(workspace),
        root=state,
        model=provider,
        model_profile=_profile(provider),
        approval_handler=handler,  # type: ignore[arg-type]
        downloads_directory=downloads,
        id_factory=_ids(),
    )
    return agent, workspace


async def test_public_read_edit_approval_save_is_one_bound_atomic_workflow(
    tmp_path: Path,
) -> None:
    provider = _EditWorkflowProvider(
        (
            {
                "old_text": "timeout: 30",
                "new_text": "timeout: 60",
                "expected_occurrences": 1,
            },
        )
    )
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent, workspace = await _agent(tmp_path, provider, approval=approve)
    target = workspace / "config.yaml"
    target.write_text("name: demo\ntimeout: 30\n", encoding="utf-8")
    target.chmod(0o640)
    prior_facts = target.stat()
    try:
        result = await agent.run("Change the timeout in config.yaml from 30 to 60")
        assert target.read_text(encoding="utf-8") == "name: demo\ntimeout: 60\n"
        assert stat_mode(target) == 0o640
        published_facts = target.stat()
        assert published_facts.st_ino != prior_facts.st_ino
        assert published_facts.st_uid == prior_facts.st_uid
        assert published_facts.st_gid == prior_facts.st_gid
        assert len(result.artifacts) == 1
        binding = result.artifacts[0].provenance.local_file_binding
        assert binding is not None
        assert binding.relative_path == "config.yaml"
        assert binding.workspace_id.startswith("workspace:sha256:")
        assert binding.observed_content_sha256.startswith("sha256:")
        assert str(workspace) not in repr(binding)
        assert binding.change_summary.replacement_count == 1
        assert len(approvals) == 1
        assert approvals[0].arguments["mode"] == "replace_bound_file"
        assert approvals[0].arguments["relative_path"] == "config.yaml"
        assert approvals[0].arguments["sensitivity"] == "internal"
        review = approvals[0].render_arguments_for_review()
        assert isinstance(review, str)
        assert str(workspace) not in review
        assert "config.yaml" in approvals[0].reason
        receipt = result.artifact_deliveries[0]
        assert receipt.mode is ArtifactDeliveryMode.REPLACE_BOUND_FILE
        assert receipt.outcome is ArtifactDeliveryOutcome.SUCCEEDED
        assert receipt.saved_path == "config.yaml"
        assert receipt.workspace_id == binding.workspace_id
        assert receipt.relative_path == binding.relative_path
        assert receipt.sha256 == result.artifacts[0].sha256
        assert receipt.byte_size == result.artifacts[0].byte_size
        assert receipt.prior_physical_revision == binding.original_physical_revision
        assert receipt.result_physical_revision is not None
        assert receipt.result_physical_revision != receipt.prior_physical_revision
    finally:
        await agent.close()


def test_ordered_replacement_insertion_and_deletion_are_deterministic() -> None:
    result = apply_bounded_text_edits(
        source=b"alpha=1\nbeta=2\ngamma=3\n",
        relative_path="config.txt",
        replacements=(
            {"old_text": "alpha=1", "new_text": "alpha=10", "expected_occurrences": 1},
            {
                "old_text": "beta=2\n",
                "new_text": "beta=2\ninserted=yes\n",
                "expected_occurrences": 1,
            },
            {"old_text": "gamma=3\n", "new_text": "", "expected_occurrences": 1},
        ),
    )

    assert result.content == b"alpha=10\nbeta=2\ninserted=yes\n"
    assert result.summary.replacement_count == 1
    assert result.summary.insertion_count == 1
    assert result.summary.deletion_count == 1


def test_text_edit_preserves_untouched_utf8_bytes_and_crlf_convention() -> None:
    source = "café\r\ntimeout: 30\r\nunchanged: ✓\r\n".encode("utf-8")
    result = apply_bounded_text_edits(
        source=source,
        relative_path="config.yaml",
        replacements=(
            {
                "old_text": "timeout: 30\n",
                "new_text": "timeout: 60\n",
                "expected_occurrences": 1,
            },
        ),
    )

    assert result.content == "café\r\ntimeout: 60\r\nunchanged: ✓\r\n".encode("utf-8")


@pytest.mark.parametrize(
    ("source", "replacements", "code"),
    [
        (
            b"alpha\n",
            ({"old_text": "missing", "new_text": "value", "expected_occurrences": 1},),
            "artifact_edit_anchor_missing",
        ),
        (
            b"same\nsame\n",
            ({"old_text": "same", "new_text": "other", "expected_occurrences": 1},),
            "artifact_edit_anchor_ambiguous",
        ),
        (
            b"abc\n",
            (
                {"old_text": "abc", "new_text": "abcd", "expected_occurrences": 1},
                {"old_text": "bc", "new_text": "BC", "expected_occurrences": 1},
            ),
            "artifact_edit_invalid",
        ),
    ],
    ids=("missing-anchor", "ambiguous-anchor", "conflicting-operations"),
)
def test_invalid_or_conflicting_edits_fail_atomically(
    source: bytes,
    replacements: tuple[dict[str, object], ...],
    code: str,
) -> None:
    from daita.artifacts.models import ArtifactError

    with pytest.raises(ArtifactError) as failure:
        apply_bounded_text_edits(
            source=source,
            relative_path="config.txt",
            replacements=replacements,
        )
    assert failure.value.code == code


@pytest.mark.parametrize(
    ("source", "replacements", "clock"),
    [
        (
            b"anchor\n",
            tuple(
                {
                    "old_text": f"anchor-{index}",
                    "new_text": f"changed-{index}",
                    "expected_occurrences": 1,
                }
                for index in range(33)
            ),
            None,
        ),
        (
            b"a" * (16 * 1024 + 1),
            (
                {
                    "old_text": "a" * (16 * 1024 + 1),
                    "new_text": "b",
                    "expected_occurrences": 1,
                },
            ),
            None,
        ),
        (
            b"a" * 100,
            (
                {
                    "old_text": "a",
                    "new_text": "b" * (64 * 1024),
                    "expected_occurrences": 100,
                },
            ),
            None,
        ),
        (
            b"anchor\n",
            (
                {
                    "old_text": "anchor",
                    "new_text": "changed",
                    "expected_occurrences": 1,
                },
            ),
            iter((0.0, 11.0)).__next__,
        ),
    ],
    ids=("operation-count", "anchor-bytes", "output-bytes", "wall-time"),
)
def test_code_owned_text_edit_limits_fail_atomically(
    source: bytes,
    replacements: tuple[dict[str, object], ...],
    clock: Callable[[], float] | None,
) -> None:
    from daita.artifacts.models import ArtifactError

    arguments: dict[str, object] = {
        "source": source,
        "relative_path": "config.txt",
        "replacements": replacements,
    }
    if clock is not None:
        arguments["clock"] = clock
    with pytest.raises(ArtifactError) as failure:
        apply_bounded_text_edits(**arguments)  # type: ignore[arg-type]
    assert failure.value.code == "artifact_edit_limited"


@pytest.mark.parametrize(
    ("target_path", "content", "expected_code"),
    [
        ("config.png", b"timeout: 30\n", "artifact_edit_invalid"),
        (
            "config.yaml",
            b"x" * (50 * 1024) + b"\xff",
            "encoding_unsupported",
        ),
        (
            "config.yaml",
            b"timeout: 30\n" + b"x" * (4 * 1024 * 1024),
            "artifact_edit_limited",
        ),
    ],
    ids=("unsupported-format", "invalid-utf8", "source-size-limit"),
)
async def test_format_encoding_and_size_limits_create_no_artifact_or_mutation(
    tmp_path: Path,
    target_path: str,
    content: bytes,
    expected_code: str,
) -> None:
    provider = _EditWorkflowProvider(
        (
            {
                "old_text": "timeout: 30",
                "new_text": "timeout: 60",
                "expected_occurrences": 1,
            },
        ),
        target_path=target_path,
    )
    agent, workspace = await _agent(tmp_path, provider)
    target = workspace / target_path
    target.write_bytes(content)
    try:
        result = await agent.run("Edit this file")
        assert result.artifacts == ()
        assert result.artifact_deliveries == ()
        assert target.read_bytes() == content
        transcript = await agent.transcript(result.run_id)
        tool_results = [
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        edit = next(
            (block for block in tool_results if block.call_id == "edit"),
            None,
        )
        assert edit is not None, [
            (result.kind, result.reason, result.steps),
            len(provider.requests),
            *[(block.call_id, block.output) for block in tool_results],
        ]
        assert edit.is_error
        assert edit.output["error"]["code"] == expected_code  # type: ignore[index]
    finally:
        await agent.close()


async def test_missing_anchor_creates_no_artifact_and_never_mutates_workspace(
    tmp_path: Path,
) -> None:
    provider = _EditWorkflowProvider(
        ({"old_text": "missing", "new_text": "value", "expected_occurrences": 1},)
    )
    agent, workspace = await _agent(tmp_path, provider)
    target = workspace / "config.yaml"
    target.write_text("timeout: 30\n", encoding="utf-8")
    try:
        result = await agent.run("Edit this file")
        assert result.artifacts == ()
        assert result.artifact_deliveries == ()
        assert target.read_text(encoding="utf-8") == "timeout: 30\n"
    finally:
        await agent.close()


async def test_hard_link_target_is_never_replaced(
    tmp_path: Path,
) -> None:
    provider = _EditWorkflowProvider(
        (
            {
                "old_text": "timeout: 30",
                "new_text": "timeout: 60",
                "expected_occurrences": 1,
            },
        )
    )
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent, workspace = await _agent(tmp_path, provider, approval=approve)
    target = workspace / "config.yaml"
    sibling = workspace / "config-copy.yaml"
    target.write_text("timeout: 30\n", encoding="utf-8")
    os.link(target, sibling)
    try:
        result = await agent.run("Edit this file")
        assert len(result.artifacts) == 1
        assert result.artifact_deliveries == ()
        assert approvals == []
        assert target.read_text(encoding="utf-8") == "timeout: 30\n"
        assert sibling.read_text(encoding="utf-8") == "timeout: 30\n"
    finally:
        await agent.close()


async def test_symlink_target_never_reaches_edit_or_artifact_io(tmp_path: Path) -> None:
    provider = _EditWorkflowProvider(
        (
            {
                "old_text": "timeout: 30",
                "new_text": "timeout: 60",
                "expected_occurrences": 1,
            },
        )
    )
    agent, workspace = await _agent(tmp_path, provider)
    outside = tmp_path / "outside.yaml"
    outside.write_text("timeout: 30\n", encoding="utf-8")
    (workspace / "config.yaml").symlink_to(outside)
    try:
        result = await agent.run("Edit this file")
        assert result.artifacts == ()
        assert result.artifact_deliveries == ()
        assert outside.read_text(encoding="utf-8") == "timeout: 30\n"
    finally:
        await agent.close()


async def test_source_drift_after_edit_commit_never_requests_approval_or_overwrites(
    tmp_path: Path,
) -> None:
    target_holder: list[Path] = []

    def drift() -> None:
        target_holder[0].write_text("timeout: 99\n", encoding="utf-8")

    provider = _EditWorkflowProvider(
        (
            {
                "old_text": "timeout: 30",
                "new_text": "timeout: 60",
                "expected_occurrences": 1,
            },
        ),
        after_edit=drift,
    )
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent, workspace = await _agent(tmp_path, provider, approval=approve)
    target = workspace / "config.yaml"
    target_holder.append(target)
    target.write_text("timeout: 30\n", encoding="utf-8")
    try:
        result = await agent.run("Edit the timeout")
        assert target.read_text(encoding="utf-8") == "timeout: 99\n"
        assert len(result.artifacts) == 1
        assert result.artifact_deliveries == ()
        assert approvals == []
        transcript = await agent.transcript(result.run_id)
        save = next(
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "save"
        )
        assert save.is_error
        assert save.output["error"]["code"] == "artifact_replacement_drift"  # type: ignore[index]
    finally:
        await agent.close()


async def test_approval_drift_fails_closed_without_second_approval_or_retry(
    tmp_path: Path,
) -> None:
    provider = _EditWorkflowProvider(
        (
            {
                "old_text": "timeout: 30",
                "new_text": "timeout: 60",
                "expected_occurrences": 1,
            },
        )
    )
    target_holder: list[Path] = []
    approvals: list[ApprovalRequest] = []

    async def drift_then_approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        target_holder[0].write_text("timeout: 45\n", encoding="utf-8")
        return ApprovalDecision.APPROVE

    agent, workspace = await _agent(tmp_path, provider, approval=drift_then_approve)
    target = workspace / "config.yaml"
    target_holder.append(target)
    target.write_text("timeout: 30\n", encoding="utf-8")
    try:
        result = await agent.run("Edit the timeout")
        assert target.read_text(encoding="utf-8") == "timeout: 45\n"
        assert len(approvals) == 1
        assert result.artifact_deliveries == ()
        transcript = await agent.transcript(result.run_id)
        save = next(
            block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == "save"
        )
        assert save.is_error
        assert save.output["error"]["code"] == "state_changed"  # type: ignore[index]
    finally:
        await agent.close()


async def test_user_cancellation_keeps_committed_edit_artifact_without_mutation(
    tmp_path: Path,
) -> None:
    provider = _EditWorkflowProvider(
        (
            {
                "old_text": "timeout: 30",
                "new_text": "timeout: 60",
                "expected_occurrences": 1,
            },
        )
    )
    approvals: list[ApprovalRequest] = []

    async def cancel(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.DENY

    agent, workspace = await _agent(tmp_path, provider, approval=cancel)
    target = workspace / "config.yaml"
    target.write_text("timeout: 30\n", encoding="utf-8")
    try:
        result = await agent.run("Edit the timeout")
        assert len(approvals) == 1
        assert len(result.artifacts) == 1
        assert result.artifact_deliveries == ()
        assert target.read_text(encoding="utf-8") == "timeout: 30\n"
    finally:
        await agent.close()


async def test_drift_before_full_observation_creates_no_edit_artifact(
    tmp_path: Path,
) -> None:
    target_holder: list[Path] = []

    def drift() -> None:
        target_holder[0].write_text("timeout: 45\n", encoding="utf-8")

    provider = _EditWorkflowProvider(
        (
            {
                "old_text": "timeout: 30",
                "new_text": "timeout: 60",
                "expected_occurrences": 1,
            },
        ),
        after_read=drift,
    )
    agent, workspace = await _agent(tmp_path, provider)
    target = workspace / "config.yaml"
    target_holder.append(target)
    target.write_text("timeout: 30\n", encoding="utf-8")
    try:
        result = await agent.run("Edit the timeout")
        assert result.artifacts == ()
        assert result.artifact_deliveries == ()
        assert target.read_text(encoding="utf-8") == "timeout: 45\n"
    finally:
        await agent.close()


async def test_whole_file_deletion_commits_and_atomically_publishes_empty_text(
    tmp_path: Path,
) -> None:
    provider = _EditWorkflowProvider(
        ({"old_text": "remove me\n", "new_text": "", "expected_occurrences": 1},),
        target_path="service.config.txt",
    )

    async def approve(_request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.APPROVE

    agent, workspace = await _agent(tmp_path, provider, approval=approve)
    target = workspace / "service.config.txt"
    target.write_text("remove me\n", encoding="utf-8")
    try:
        result = await agent.run("Delete the complete file contents")
        assert target.read_bytes() == b""
        assert result.artifacts[0].filename == "service.config.txt"
        assert result.artifacts[0].byte_size == 0
        assert result.artifact_deliveries[0].byte_size == 0
        assert (
            result.artifact_deliveries[0].outcome is ArtifactDeliveryOutcome.SUCCEEDED
        )
    finally:
        await agent.close()


async def _prepare_only_edit(
    tmp_path: Path,
    *,
    relative_path: str = "config.yaml",
) -> tuple[Agent, Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    provider = _EditWorkflowProvider(
        (
            {
                "old_text": "timeout: 30",
                "new_text": "timeout: 60",
                "expected_occurrences": 1,
            },
        ),
        save=False,
        target_path=relative_path,
    )
    agent, workspace = await _agent(tmp_path, provider)
    target = workspace / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("timeout: 30\n", encoding="utf-8")
    result = await agent.run("Prepare the timeout edit")
    assert len(result.artifacts) == 1
    assert result.artifact_deliveries == ()
    return agent, target, result.artifacts[0].artifact_id


@pytest.mark.parametrize("rename_scope", ("workspace", "parent"))
async def test_namespace_drift_immediately_before_publication_preserves_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rename_scope: str,
) -> None:
    relative_path = (
        "config.yaml" if rename_scope == "workspace" else "nested/config.yaml"
    )
    agent, target, artifact_id = await _prepare_only_edit(
        tmp_path,
        relative_path=relative_path,
    )
    delivery = agent._embedded._artifact_delivery
    assert delivery is not None
    source_directory = target.parent
    destination_directory = (
        tmp_path / "workspace-renamed"
        if rename_scope == "workspace"
        else source_directory.parent / "nested-renamed"
    )
    drifted_target = destination_directory / target.name
    original_verify = delivery_module._verify_staged_bound_file

    def verify_then_rename(*args: Any, **kwargs: Any) -> None:
        original_verify(*args, **kwargs)
        source_directory.rename(destination_directory)

    monkeypatch.setattr(
        delivery_module,
        "_verify_staged_bound_file",
        verify_then_rename,
    )
    try:
        receipt = await delivery.save_committed(
            run_id=f"run-{rename_scope}-namespace-drift",
            artifact_id=artifact_id,
            mode="replace_bound_file",
        )
        assert receipt.outcome is ArtifactDeliveryOutcome.FAILED
        assert receipt.failure_code == "artifact_delivery_failed"
        assert receipt.result_physical_revision is None
        assert drifted_target.read_text(encoding="utf-8") == "timeout: 30\n"
        assert not tuple(destination_directory.glob(".daita-edit-*.tmp"))
    finally:
        await agent.close()


async def test_unsupported_metadata_and_ineligible_owner_fail_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daita.artifacts.models import ArtifactError

    metadata_agent, metadata_target, metadata_artifact_id = await _prepare_only_edit(
        tmp_path / "metadata"
    )
    metadata_delivery = metadata_agent._embedded._artifact_delivery
    assert metadata_delivery is not None
    try:
        with monkeypatch.context() as metadata_patch:
            metadata_patch.setattr(
                delivery_module.os,
                "listxattr",
                lambda _descriptor: ["com.daita.phase3"],
                raising=False,
            )
            with pytest.raises(ArtifactError) as metadata_failure:
                await metadata_delivery.preflight_save(
                    run_id="run-preflight-metadata",
                    artifact_id=metadata_artifact_id,
                    mode="replace_bound_file",
                )
        assert metadata_failure.value.code == "artifact_edit_binding_invalid"
        assert metadata_target.read_text(encoding="utf-8") == "timeout: 30\n"
    finally:
        await metadata_agent.close()

    owner_agent, owner_target, owner_artifact_id = await _prepare_only_edit(
        tmp_path / "owner"
    )
    owner_delivery = owner_agent._embedded._artifact_delivery
    assert owner_delivery is not None
    try:
        actual_uid = os.geteuid()
        monkeypatch.setattr(delivery_module.os, "geteuid", lambda: actual_uid + 1)
        with pytest.raises(ArtifactError) as owner_failure:
            await owner_delivery.preflight_save(
                run_id="run-preflight-owner",
                artifact_id=owner_artifact_id,
                mode="replace_bound_file",
            )
        assert owner_failure.value.code == "artifact_edit_binding_invalid"
        assert owner_target.read_text(encoding="utf-8") == "timeout: 30\n"
    finally:
        await owner_agent.close()


async def test_prepublication_failure_returns_failed_receipt_and_preserves_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, target, artifact_id = await _prepare_only_edit(tmp_path)
    delivery = agent._embedded._artifact_delivery
    assert delivery is not None

    def fail_staged_verification(*_args: object, **_kwargs: object) -> None:
        raise OSError("staged verification unavailable")

    monkeypatch.setattr(
        delivery_module,
        "_verify_staged_bound_file",
        fail_staged_verification,
    )
    try:
        receipt = await delivery.save_committed(
            run_id="run-failed-replacement",
            artifact_id=artifact_id,
            mode="replace_bound_file",
        )
        assert receipt.outcome is ArtifactDeliveryOutcome.FAILED
        assert receipt.failure_code == "artifact_delivery_failed"
        assert receipt.result_physical_revision is None
        assert target.read_text(encoding="utf-8") == "timeout: 30\n"
        assert not tuple(target.parent.glob(".daita-edit-*.tmp"))
    finally:
        await agent.close()


async def test_post_replace_syscall_interruption_returns_uncertain_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, target, artifact_id = await _prepare_only_edit(tmp_path)
    delivery = agent._embedded._artifact_delivery
    assert delivery is not None
    original_replace = delivery_module.os.replace

    def replace_then_interrupt(*args: Any, **kwargs: Any) -> None:
        original_replace(*args, **kwargs)
        raise OSError("interrupted after atomic replacement")

    monkeypatch.setattr(delivery_module.os, "replace", replace_then_interrupt)
    try:
        receipt = await delivery.save_committed(
            run_id="run-post-syscall-uncertain",
            artifact_id=artifact_id,
            mode="replace_bound_file",
        )
        assert receipt.outcome is ArtifactDeliveryOutcome.UNCERTAIN
        assert receipt.failure_code == "artifact_replacement_uncertain"
        assert receipt.result_physical_revision is not None
        assert target.read_text(encoding="utf-8") == "timeout: 60\n"
        assert not tuple(target.parent.glob(".daita-edit-*.tmp"))
    finally:
        await agent.close()


async def test_nonwritable_parent_permission_never_expands_for_replacement(
    tmp_path: Path,
) -> None:
    from daita.artifacts.models import ArtifactError

    agent, target, artifact_id = await _prepare_only_edit(tmp_path)
    delivery = agent._embedded._artifact_delivery
    assert delivery is not None
    original_mode = stat_mode(target.parent)
    target.parent.chmod(0o500)
    try:
        with pytest.raises(ArtifactError) as failure:
            await delivery.preflight_save(
                run_id="run-permission-replacement",
                artifact_id=artifact_id,
                mode="replace_bound_file",
            )
        assert failure.value.code == "artifact_edit_binding_invalid"
        assert target.read_text(encoding="utf-8") == "timeout: 30\n"
    finally:
        target.parent.chmod(original_mode)
        await agent.close()


async def test_raw_edit_and_replacement_targets_fail_before_workspace_or_artifact_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daita.llm.providers.mock import MockModelProvider

    async def unexpected_workspace_io(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("forged edit reached workspace I/O")

    async def unexpected_artifact_io(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("forged replacement reached artifact I/O")

    monkeypatch.setattr(
        LocalWorkspaceBackend,
        "observe_bound_text",
        unexpected_workspace_io,
    )
    monkeypatch.setattr(AgentHomeArtifactStore, "find_ref", unexpected_artifact_io)
    provider = MockModelProvider(
        (
            _call(
                "load-edit",
                "toolbox_load",
                {"tool_names": ["artifact_edit_text"]},
            ),
            _call(
                "forged-edit",
                "artifact_edit_text",
                {
                    "path": "config.yaml",
                    "revision": "sha256:" + "0" * 64,
                    "content": "timeout: 60\n",
                    "replacements": [
                        {
                            "old_text": "timeout: 30",
                            "new_text": "timeout: 60",
                            "expected_occurrences": 1,
                        }
                    ],
                },
            ),
            _call(
                "load-save",
                "toolbox_load",
                {"tool_names": ["artifact_save_local"]},
            ),
            _call(
                "forged-save",
                "artifact_save_local",
                {
                    "artifact_id": "artifact-" + "0" * 32,
                    "mode": "replace_bound_file",
                    "path": "redirected.yaml",
                },
            ),
            _stop("forged calls rejected"),
        ),
        provider_id="mock:forged-local-text-edit",
    )
    (tmp_path / "workspace").mkdir()
    (tmp_path / "downloads").mkdir()
    agent = await Agent.create(
        "forged-local-edit",
        workspace=LocalWorkspace(tmp_path / "workspace"),
        root=tmp_path / "state",
        model=provider,
        model_profile=ModelProfile(
            id=provider.provider_id,
            context_window_tokens=128_000,
            max_output_tokens=2_000,
            supports_tools=True,
            supports_parallel_tools=False,
        ),
        downloads_directory=tmp_path / "downloads",
    )
    target = tmp_path / "workspace" / "config.yaml"
    target.write_text("timeout: 30\n", encoding="utf-8")
    try:
        result = await agent.run("Attempt forged file edits")
        assert result.artifacts == ()
        assert result.artifact_deliveries == ()
        assert target.read_text(encoding="utf-8") == "timeout: 30\n"
        transcript = await agent.transcript(result.run_id)
        errors = {
            block.call_id: block
            for message in transcript.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
            and block.call_id in {"forged-edit", "forged-save"}
        }
        assert set(errors) == {"forged-edit", "forged-save"}
        assert all(block.is_error for block in errors.values())
        assert errors["forged-edit"].output["error"]["code"] in {  # type: ignore[index]
            "missing_arguments",
            "unexpected_arguments",
        }
        assert errors["forged-save"].output["error"]["code"] == (  # type: ignore[index]
            "unexpected_arguments"
        )
    finally:
        await agent.close()


async def test_machine_origin_cannot_project_or_forge_ambient_workspace_edit_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _EditWorkflowProvider(
        ({"old_text": "a", "new_text": "b", "expected_occurrences": 1},),
        save=False,
    )
    agent, _workspace = await _agent(tmp_path, provider)
    embedded = agent._embedded

    async def unexpected_workspace_io(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("machine forged call reached workspace I/O")

    async def unexpected_artifact_io(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("machine forged call reached artifact I/O")

    monkeypatch.setattr(
        LocalWorkspaceBackend,
        "observe_bound_text",
        unexpected_workspace_io,
    )
    monkeypatch.setattr(AgentHomeArtifactStore, "find_ref", unexpected_artifact_io)
    instruction = "Process one bounded job event."
    scope = ExecutionScope(
        scope_id="scope-local-edit-negative",
        revision=1,
        agent_id=agent.id,
        principal_id=f"agent:{agent.id}",
        grant_id="grant-local-edit-negative",
        job_id="job-local-edit-negative",
        job_revision=1,
        allowed_source_ids=("source-none",),
        allowed_resource_ids=("resource-none",),
        allowed_capability_ids=("artifact.edit_text", "artifact.save_local"),
        allowed_access_modes=frozenset({AccessMode.NONE, AccessMode.READ}),
        allowed_operational_effects=frozenset(
            {OperationalEffect.NONE, OperationalEffect.CHANGE_INFRASTRUCTURE}
        ),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        eligible_model_routes=(provider.provider_id,),
        per_run_max_cost_usd=Decimal("0.01"),
        per_run_max_tokens=1_000,
        delivery_destination="conversation_inbox:conversation-machine-edit",
    )
    run = RunInput(
        id="run-machine-local-edit-negative",
        agent_id=agent.id,
        message=instruction,
        conversation_id="conversation-machine-edit",
        created_at=datetime.now(UTC),
        start=RunStartEnvelope(
            origin=RunOrigin.JOB_EVENT,
            instruction_authority=InstructionAuthority.CODE_OWNED,
            trusted_instruction_id="followup-local-edit-v1",
            trusted_instruction=instruction,
            instruction_digest=(
                "sha256:" + sha256(instruction.encode("utf-8")).hexdigest()
            ),
            untrusted_payload={},
            payload_digest="sha256:" + sha256(b"{}").hexdigest(),
            execution_scope=scope,
        ),
    )
    messages = (run.start_message(),)
    try:
        catalog = await embedded._capability_runtime.prepare_run(run)
        assert "artifact_edit_text" not in {
            entry.view.name for entry in catalog.entries
        }
        assert "artifact_save_local" not in {
            entry.view.name for entry in catalog.entries
        }
        assert "file_read" not in {entry.view.name for entry in catalog.entries}
        projection = embedded._capability_runtime.project(catalog, messages)
        outcome = await embedded._capability_runtime.execute_all(
            run,
            (
                ToolCall(
                    id="machine-forged-edit",
                    name="artifact_edit_text",
                    arguments={
                        "binding": "forged",
                        "replacements": (
                            {
                                "old_text": "a",
                                "new_text": "b",
                                "expected_occurrences": 1,
                            },
                        ),
                    },
                ),
                ToolCall(
                    id="machine-forged-save",
                    name="artifact_save_local",
                    arguments={
                        "artifact_id": "artifact-" + "0" * 32,
                        "mode": "replace_bound_file",
                    },
                ),
            ),
            projection=projection,
            messages=messages,
            sensitivity=ModelSensitivity.INTERNAL,
        )
        assert len(outcome) == 2
        assert all(item.is_error for item in outcome)
        for item in outcome:
            error = item.output.get("error")
            assert isinstance(error, Mapping)
            assert error.get("code") == "tool_not_available"
    finally:
        await agent.close()


async def test_cancellation_before_publication_cleans_staging_and_preserves_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, target, artifact_id = await _prepare_only_edit(tmp_path)
    delivery = agent._embedded._artifact_delivery
    assert delivery is not None
    entered = threading.Event()
    release = threading.Event()
    original_verify = delivery_module._verify_staged_bound_file

    def blocked_verify(*args: Any, **kwargs: Any) -> None:
        original_verify(*args, **kwargs)
        entered.set()
        release.wait(2)

    monkeypatch.setattr(delivery_module, "_verify_staged_bound_file", blocked_verify)
    try:
        saving = asyncio.create_task(
            delivery.save_committed(
                run_id="run-cancelled-replacement",
                artifact_id=artifact_id,
                mode="replace_bound_file",
            )
        )
        assert await asyncio.to_thread(entered.wait, 2)
        saving.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await saving
        assert target.read_text(encoding="utf-8") == "timeout: 30\n"
        assert not tuple(target.parent.glob(".daita-edit-*.tmp"))
    finally:
        release.set()
        await agent.close()


async def test_post_publication_cancellation_returns_uncertain_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, target, artifact_id = await _prepare_only_edit(tmp_path)
    delivery = agent._embedded._artifact_delivery
    assert delivery is not None
    published = threading.Event()
    fsync_blocked = threading.Event()
    release = threading.Event()
    original_replace = delivery_module.os.replace
    original_fsync = delivery_module._fsync_descriptor

    def observed_replace(*args: Any, **kwargs: Any) -> None:
        original_replace(*args, **kwargs)
        published.set()

    def blocked_post_publish_fsync(descriptor: int) -> None:
        if published.is_set():
            fsync_blocked.set()
            release.wait(2)
        original_fsync(descriptor)

    monkeypatch.setattr(delivery_module.os, "replace", observed_replace)
    monkeypatch.setattr(
        delivery_module, "_fsync_descriptor", blocked_post_publish_fsync
    )
    try:
        saving = asyncio.create_task(
            delivery.save_committed(
                run_id="run-uncertain-replacement",
                artifact_id=artifact_id,
                mode="replace_bound_file",
            )
        )
        assert await asyncio.to_thread(fsync_blocked.wait, 2)
        saving.cancel()
        release.set()
        receipt = await saving
        assert receipt.outcome is ArtifactDeliveryOutcome.UNCERTAIN
        assert receipt.failure_code == "artifact_replacement_uncertain"
        assert target.read_text(encoding="utf-8") == "timeout: 60\n"
        assert not tuple(target.parent.glob(".daita-edit-*.tmp"))
    finally:
        release.set()
        await agent.close()


def stat_mode(path: Path) -> int:
    import stat

    return stat.S_IMODE(path.stat().st_mode)
