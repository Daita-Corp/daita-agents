"""Runtime-native factories used by eval unit tests."""

from __future__ import annotations


class FakeEvalAgent:
    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0
        self.started = False
        self.stopped = False
        self.name = "Fake Eval Agent"
        self.agent_id = "fake_eval_agent"

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def run_detailed(self, prompt, **kwargs):
        del prompt, kwargs
        if self.index < len(self.responses):
            result = self.responses[self.index]
            self.index += 1
        else:
            result = self.responses[-1]
        return result


def create_passing_agent():
    return FakeEvalAgent(
        [
            runtime_result(
                "Widget A revenue: 12840.50",
                tasks=[
                    task("task-1", "db.sql.validate", owner="sqlite"),
                    task("task-2", "db.sql.execute_read", owner="sqlite"),
                ],
                evidence=[
                    evidence_record(
                        "evidence-1",
                        "sql.validation",
                        "sqlite",
                        "task-1",
                        {
                            "sql": (
                                "SELECT product, SUM(revenue) FROM sales "
                                "GROUP BY product LIMIT 5"
                            ),
                            "statement_facts": {"statement_type": "select"},
                        },
                    ),
                    evidence_record(
                        "evidence-2",
                        "query.result",
                        "sqlite",
                        "task-2",
                        {
                            "sql": (
                                "SELECT product, SUM(revenue) FROM sales "
                                "GROUP BY product LIMIT 5"
                            ),
                            "rows": [{"product": "Widget A", "revenue": 12840.50}],
                        },
                    ),
                ],
                tokens_total=100,
                cost=0.001,
            )
        ]
    )


def create_sql_failure_agent():
    return FakeEvalAgent(
        [
            runtime_result(
                "Done",
                tasks=[
                    task("task-1", "db.sql.validate", owner="sqlite"),
                    task("task-2", "db.sql.execute_read", owner="sqlite"),
                ],
                evidence=[
                    evidence_record(
                        "evidence-1",
                        "sql.validation",
                        "sqlite",
                        "task-1",
                        {
                            "sql": "SELECT * FROM users_pii",
                            "statement_facts": {"statement_type": "select"},
                        },
                    ),
                    evidence_record(
                        "evidence-2",
                        "query.result",
                        "sqlite",
                        "task-2",
                        {"sql": "SELECT * FROM users_pii", "rows": []},
                    ),
                ],
            )
        ]
    )


def create_runtime_capability_agent():
    return FakeEvalAgent(
        [
            runtime_result(
                "Loaded runtime evidence.",
                tasks=[
                    task("task-1", "catalog.schema.search", owner="catalog"),
                    task("task-2", "db.sql.validate", owner="sqlite"),
                    task("task-3", "db.sql.execute_read", owner="sqlite"),
                ],
                evidence=[
                    evidence_record(
                        "evidence-1",
                        "schema.search_result",
                        "catalog",
                        "task-1",
                        {"tables": ["sales"]},
                    ),
                    evidence_record(
                        "evidence-2",
                        "sql.validation",
                        "sqlite",
                        "task-2",
                        {
                            "sql": "SELECT product FROM sales LIMIT 5",
                            "statement_facts": {"statement_type": "select"},
                        },
                    ),
                    evidence_record(
                        "evidence-3",
                        "query.result",
                        "sqlite",
                        "task-3",
                        {
                            "sql": "SELECT product FROM sales LIMIT 5",
                            "rows": [{"product": "Widget A"}],
                        },
                    ),
                ],
            )
        ]
    )


def create_governance_agent():
    return FakeEvalAgent(
        [
            runtime_result(
                "Governance approved runtime work.",
                tasks=[task("task-1", "db.sql.validate", owner="sqlite")],
                evidence=[
                    evidence_record(
                        "evidence-1",
                        "sql.validation",
                        "sqlite",
                        "task-1",
                        {
                            "sql": "SELECT 1",
                            "statement_facts": {"statement_type": "select"},
                        },
                    )
                ],
                governance={
                    "allowed": True,
                    "blocked": False,
                    "pending_approval": False,
                    "decisions": [
                        {
                            "policy_id": "read_only_sql",
                            "effect": "allow",
                            "reason": "Read-only SQL passed validation.",
                        }
                    ],
                    "approval_requests": [],
                },
            )
        ]
    )


def create_unstable_agent():
    return FakeEvalAgent(
        [
            runtime_result(
                "Answer A",
                tasks=[task("task-1", "catalog.schema.search", owner="catalog")],
                evidence=[
                    evidence_record(
                        "evidence-1",
                        "schema.search_result",
                        "catalog",
                        "task-1",
                        {"tables": ["a"]},
                    )
                ],
            ),
            runtime_result(
                "Answer B",
                tasks=[task("task-2", "db.sql.execute_read", owner="sqlite")],
                evidence=[
                    evidence_record(
                        "evidence-2",
                        "query.result",
                        "sqlite",
                        "task-2",
                        {"sql": "SELECT * FROM b LIMIT 5", "rows": []},
                    )
                ],
            ),
        ]
    )


def create_legacy_agent():
    class LegacyAgent(FakeEvalAgent):
        async def run_detailed(self, prompt, **kwargs):
            del prompt, kwargs
            return {
                "result": "legacy",
                "tool_calls": [
                    {"tool": "sqlite_query", "arguments": {"sql": "SELECT 1"}}
                ],
            }

    return LegacyAgent([])


def runtime_result(
    answer,
    *,
    tasks,
    evidence,
    governance=None,
    tokens_total=None,
    cost=None,
):
    return {
        "operation_id": "operation-1",
        "status": "succeeded",
        "answer": answer,
        "contract": {"operation_type": "db.query"},
        "evidence": evidence,
        "warnings": [],
        "diagnostics": {
            "execution": {
                "task_count": len(tasks),
                "tasks": tasks,
            },
            "governance": governance
            or {
                "allowed": True,
                "blocked": False,
                "pending_approval": False,
                "decisions": [],
                "approval_requests": [],
            },
            "llm": {"tokens": {"total_tokens": tokens_total}, "cost": cost},
        },
    }


def task(task_id, capability_id, *, owner, status="succeeded"):
    return {
        "id": task_id,
        "operation_id": "operation-1",
        "capability_id": capability_id,
        "executor_id": capability_id,
        "input": {},
        "status": status,
        "required_evidence": [],
        "dependencies": [],
        "metadata": {"owner": owner},
    }


def evidence_record(evidence_id, kind, owner, task_id, payload, *, accepted=True):
    return {
        "id": evidence_id,
        "kind": kind,
        "owner": owner,
        "operation_id": "operation-1",
        "task_id": task_id,
        "accepted": accepted,
        "payload": payload,
        "metadata": {},
    }
