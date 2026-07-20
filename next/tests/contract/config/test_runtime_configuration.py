from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import sqlite3

import pytest

from daita import AgentConfig, RetryPolicy, RetryStrategy
from daita.config import AgentRuntimeDefaults, AgentRuntimeDefaultsConflictError
from daita.identity import AgentIdentity
from daita.llm import ProviderErrorCode
from daita.loop.models import LoopBudgets
from daita.operations.governance import DefaultPolicyProfile
from daita.storage import sqlite as sqlite_owner
from daita.storage.sqlite import SQLiteCorruptionError, SQLiteOperationStore

NOW = datetime(2026, 7, 19, 18, 0, tzinfo=timezone.utc)


def test_runtime_configuration_is_composed_from_bounded_owner_records() -> None:
    budgets = LoopBudgets(
        max_turns=3,
        max_actions=5,
        max_total_tokens=8_000,
        max_estimated_cost_usd=Decimal("0.50"),
    )
    retry = RetryPolicy(
        max_attempts_per_provider=2,
        strategy=RetryStrategy.FIXED,
        base_delay_seconds=0.1,
        max_delay_seconds=0.1,
        retryable_codes=frozenset({ProviderErrorCode.TIMEOUT}),
    )
    policy = DefaultPolicyProfile(allow_destructive=False)

    config = AgentConfig(
        budgets=budgets,
        retry_policy=retry,
        policy_profile=policy,
    )

    assert config.budgets.max_estimated_cost_usd == Decimal("0.50")
    assert config.retry_policy.retryable_codes == frozenset({ProviderErrorCode.TIMEOUT})
    assert config.policy_profile.fingerprint == policy.fingerprint


def test_retry_configuration_is_bounded_and_cannot_retry_terminal_failures() -> None:
    with pytest.raises(ValueError, match="one through four"):
        RetryPolicy(max_attempts_per_provider=5)
    with pytest.raises(ValueError, match="terminal failures"):
        RetryPolicy(retryable_codes=frozenset({ProviderErrorCode.AUTHENTICATION_ERROR}))
    with pytest.raises(ValueError, match="less than"):
        RetryPolicy(base_delay_seconds=2.0, max_delay_seconds=1.0)


async def test_sqlite_runtime_defaults_are_exact_immutable_and_restart_stable(
    tmp_path,
) -> None:
    path = tmp_path / "runtime-defaults.db"
    budgets = LoopBudgets(
        max_turns=3,
        max_actions=5,
        max_repairs=2,
        max_identical_failures=4,
        max_observation_characters=12_345,
        max_total_tokens=8_000,
        max_wall_time_seconds=12.5,
        task_timeout_seconds=3.25,
        max_estimated_cost_usd=Decimal("0.50"),
    )
    defaults = AgentRuntimeDefaults(
        budgets=budgets,
        policy_profile=DefaultPolicyProfile(
            version="2",
            allow_destructive=True,
        ),
    )
    identity = AgentIdentity(
        id="agent-runtime-defaults",
        display_name="Runtime defaults",
        created_at=NOW,
    )

    store = await SQLiteOperationStore.open(path, clock=lambda: NOW)
    await store.initialize_identity(identity)
    assert await store.load_runtime_defaults(identity.id) is None
    assert await store.bind_runtime_defaults(identity.id, defaults) == defaults
    assert await store.bind_runtime_defaults(identity.id, defaults) == defaults
    with pytest.raises(
        AgentRuntimeDefaultsConflictError,
        match="different runtime defaults",
    ):
        await store.bind_runtime_defaults(
            identity.id,
            replace(defaults, budgets=replace(budgets, max_turns=4)),
        )
    await store.close()

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT schema_version, revision, fingerprint, "
            "budget_max_turns, budget_max_wall_time_seconds, "
            "budget_max_estimated_cost_usd, policy_version, "
            "policy_allow_destructive FROM agent_runtime_defaults"
        ).fetchone() == (
            1,
            1,
            defaults.fingerprint,
            3,
            12.5,
            "0.50",
            "2",
            1,
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE agent_runtime_defaults SET budget_max_turns = 4")
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM agent_runtime_defaults")

    reopened = await SQLiteOperationStore.open(path, clock=lambda: NOW)
    assert await reopened.load_runtime_defaults(identity.id) == defaults
    await reopened.close()

    corrupted = await SQLiteOperationStore.open(path, clock=lambda: NOW)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER agent_runtime_defaults_reject_update")
        connection.execute(
            "UPDATE agent_runtime_defaults SET fingerprint = ?",
            ("0" * 64,),
        )
        connection.commit()
    try:
        with pytest.raises(SQLiteCorruptionError, match="runtime defaults"):
            await corrupted.load_runtime_defaults(identity.id)
    finally:
        await corrupted.close()


async def test_migration_fourteen_upgrades_v13_without_inventing_defaults(
    tmp_path,
) -> None:
    path = tmp_path / "legacy-v13.db"
    identity = AgentIdentity(
        id="agent-v13-runtime-defaults",
        display_name="Legacy v13",
        created_at=NOW,
    )
    legacy = await sqlite_owner._open_with_migrations(
        path,
        migrations=sqlite_owner._MIGRATIONS[:13],
        clock=lambda: NOW,
    )
    await legacy.initialize_identity(identity)
    await legacy.close()

    upgraded = await SQLiteOperationStore.open(path, clock=lambda: NOW)
    try:
        assert await upgraded.load_runtime_defaults(identity.id) is None
        defaults = AgentRuntimeDefaults(budgets=LoopBudgets(max_turns=4))
        assert await upgraded.bind_runtime_defaults(identity.id, defaults) == defaults
    finally:
        await upgraded.close()

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (14,)
        assert connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version DESC LIMIT 1"
        ).fetchone() == (14, "bind_agent_runtime_defaults")
