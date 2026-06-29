"""Deterministic safety lanes for DB runtime requests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Iterable, Mapping


class DbCapabilityLane(str, Enum):
    """Permission lanes granted before model planning."""

    NONE = "none"
    SCHEMA = "schema"
    MEMORY_ANSWER = "memory_answer"
    MEMORY_WRITE = "memory_write"
    READ = "read"
    WRITE_PROPOSE = "write_propose"
    WRITE_EXECUTE = "write_execute"
    ADMIN = "admin"
    MONITOR_READ = "monitor_read"
    MONITOR_WRITE = "monitor_write"
    MONITOR_EXECUTE = "monitor_execute"


@dataclass(frozen=True)
class DbSafetyRewrite:
    """A deterministic safety downgrade applied to a candidate lane."""

    rule: str
    from_lanes: tuple[DbCapabilityLane, ...]
    to_lanes: tuple[DbCapabilityLane, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "from_lanes": [lane.value for lane in self.from_lanes],
            "to_lanes": [lane.value for lane in self.to_lanes],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DbLaneGrant:
    """Capabilities and constraints implied by one granted lane."""

    lane: DbCapabilityLane
    required_capabilities: tuple[str, ...]
    forbidden_capabilities: tuple[str, ...]
    approval_required: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane.value,
            "required_capabilities": list(self.required_capabilities),
            "forbidden_capabilities": list(self.forbidden_capabilities),
            "approval_required": self.approval_required,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DbSafetyFrame:
    """Hard permission facts for DB planning.

    The safety frame answers which runtime lanes are allowed. It does not
    decide business intent, report type, metric meaning, or query strategy.
    """

    prompt: str
    normalized_prompt: str
    explicit_schema_only: bool
    direct_memory_operation: str | None
    sql_statement_type: str | None
    has_db_target: bool
    has_mutation_payload: bool
    explicit_execution: bool
    destructive: bool
    admin: bool
    monitor_operation: str | None
    requested_capabilities: tuple[str, ...]
    granted_lanes: tuple[DbCapabilityLane, ...]
    forbidden_capabilities: tuple[str, ...]
    rewrites: tuple[DbSafetyRewrite, ...]
    assumptions: tuple[str, ...]
    lane_grants: tuple[DbLaneGrant, ...]
    required_capabilities: tuple[str, ...]
    approval_required: bool
    blocked_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "normalized_prompt": self.normalized_prompt,
            "explicit_schema_only": self.explicit_schema_only,
            "direct_memory_operation": self.direct_memory_operation,
            "sql_statement_type": self.sql_statement_type,
            "has_db_target": self.has_db_target,
            "has_mutation_payload": self.has_mutation_payload,
            "explicit_execution": self.explicit_execution,
            "destructive": self.destructive,
            "admin": self.admin,
            "monitor_operation": self.monitor_operation,
            "requested_capabilities": list(self.requested_capabilities),
            "granted_lanes": [lane.value for lane in self.granted_lanes],
            "forbidden_capabilities": list(self.forbidden_capabilities),
            "rewrites": [rewrite.to_dict() for rewrite in self.rewrites],
            "assumptions": list(self.assumptions),
            "lane_grants": [grant.to_dict() for grant in self.lane_grants],
            "required_capabilities": list(self.required_capabilities),
            "approval_required": self.approval_required,
            "blocked_actions": list(self.blocked_actions),
        }


class DbSafetyVerifier:
    """Build verified DB safety frames from deterministic prompt facts."""

    def verify(
        self,
        request: str | Any,
        *,
        requested_capabilities: Iterable[str] = (),
    ) -> DbSafetyFrame:
        prompt = _request_prompt(request)
        requested = _ordered_unique(
            (
                *_request_requested_capabilities(request),
                *(str(item) for item in requested_capabilities),
            )
        )
        request_mode = _request_mode(request)
        normalized = _normalize_prompt(prompt)
        explicit_schema_only = _explicit_schema_only(normalized)
        sql = _extract_sql_shape(normalized)
        statement_type = sql["statement_type"]
        explicit_execution = _explicit_execution(normalized)
        direct_memory_operation = _direct_memory_operation(normalized)
        monitor_operation = _monitor_operation(normalized)
        natural_schema = _natural_schema_request(normalized)
        natural_read = _natural_row_read_request(normalized)
        underspecified_write = _underspecified_write(normalized)
        destructive = _destructive_sql(statement_type)
        admin = _admin_sql(statement_type, normalized)
        has_db_target = bool(sql["tables"]) or natural_schema or natural_read
        has_mutation_payload = _has_mutation_payload(statement_type, normalized)
        rewrites: list[DbSafetyRewrite] = []
        assumptions: list[str] = []
        lanes: list[DbCapabilityLane] = []

        if request_mode in {"schema.query", "schema.relationships"}:
            lanes.append(DbCapabilityLane.SCHEMA)
            assumptions.append("explicit_request_mode_grants_schema_lane")
        elif request_mode in {"data.query", "query"}:
            lanes.append(DbCapabilityLane.READ)
            assumptions.append("explicit_request_mode_grants_read_lane")
        elif request_mode in {"memory.update", "memory.write"}:
            lanes.append(DbCapabilityLane.MEMORY_WRITE)
            assumptions.append("explicit_request_mode_grants_memory_write_lane")
        elif request_mode in {
            "memory.answer",
            "memory.recall",
            "memory.list",
            "memory.inspect",
        }:
            lanes.append(DbCapabilityLane.MEMORY_ANSWER)
            assumptions.append("explicit_request_mode_grants_memory_answer_lane")
        elif request_mode in {
            "monitor.read",
            "monitor.list",
            "monitor.inspect",
            "monitor.explain_run",
        }:
            lanes.append(DbCapabilityLane.MONITOR_READ)
            assumptions.append("explicit_request_mode_grants_monitor_read_lane")
        elif request_mode in {
            "monitor.write",
            "monitor.create",
            "monitor.update",
            "monitor.pause",
            "monitor.resume",
            "monitor.delete",
            "monitor.disable",
        }:
            lanes.append(DbCapabilityLane.MONITOR_WRITE)
            assumptions.append("explicit_request_mode_grants_monitor_write_lane")
        elif request_mode in {"monitor.execute", "monitor.delivery"}:
            lanes.append(DbCapabilityLane.MONITOR_EXECUTE)
            assumptions.append("explicit_request_mode_grants_monitor_execute_lane")
        elif explicit_schema_only:
            lanes.append(DbCapabilityLane.SCHEMA)
            assumptions.append("schema_only_forbids_row_access")
            if statement_type in _ROW_OR_WRITE_SQL:
                rewrites.append(
                    DbSafetyRewrite(
                        rule="schema_only_forbids_sql_execution",
                        from_lanes=(
                            _sql_lane(
                                statement_type,
                                explicit_execution=explicit_execution,
                            ),
                        ),
                        to_lanes=(DbCapabilityLane.SCHEMA,),
                        reason="hard_schema_only_negation",
                    )
                )
        elif direct_memory_operation is not None:
            if direct_memory_operation in {"recall", "list", "inspect"}:
                lanes.append(DbCapabilityLane.MEMORY_ANSWER)
                assumptions.append("direct_memory_prompt_forbids_sql")
            else:
                lanes.append(DbCapabilityLane.MEMORY_WRITE)
                assumptions.append("direct_memory_write_forbids_sql")
        elif monitor_operation is not None:
            lanes.append(_monitor_lane(monitor_operation))
        elif admin:
            lanes.append(DbCapabilityLane.ADMIN)
            assumptions.append("admin_requires_explicit_sql_structure")
        elif statement_type in _WRITE_SQL:
            candidate = _sql_lane(
                statement_type,
                explicit_execution=explicit_execution,
            )
            if not sql["tables"] or not has_mutation_payload:
                rewrites.append(
                    DbSafetyRewrite(
                        rule="write_requires_explicit_target_and_payload",
                        from_lanes=(candidate,),
                        to_lanes=(DbCapabilityLane.NONE,),
                        reason="underspecified_sql_write",
                    )
                )
                assumptions.append("write_downgraded_without_target_or_payload")
                lanes.append(DbCapabilityLane.NONE)
            else:
                lanes.append(candidate)
                if candidate is DbCapabilityLane.WRITE_PROPOSE:
                    assumptions.append("write_execution_requires_explicit_run_word")
        elif statement_type in _READ_SQL or natural_read:
            lanes.append(DbCapabilityLane.READ)
        elif natural_schema:
            lanes.append(DbCapabilityLane.SCHEMA)
        elif underspecified_write:
            rewrites.append(
                DbSafetyRewrite(
                    rule="write_requires_explicit_sql_shape",
                    from_lanes=(DbCapabilityLane.WRITE_PROPOSE,),
                    to_lanes=(DbCapabilityLane.NONE,),
                    reason="natural_language_write_not_enough",
                )
            )
            assumptions.append("write_downgraded_without_explicit_sql_shape")
            lanes.append(DbCapabilityLane.NONE)
        else:
            lanes.append(DbCapabilityLane.NONE)

        lanes = list(_ordered_lanes(lanes))
        lanes, override_rewrites = _apply_requested_capabilities(lanes, requested)
        rewrites.extend(override_rewrites)

        grants = tuple(_grant_for_lane(lane) for lane in lanes)
        forbidden = _ordered_unique(
            capability
            for grant in grants
            for capability in grant.forbidden_capabilities
        )
        required = _ordered_unique(
            capability
            for grant in grants
            for capability in grant.required_capabilities
            if capability not in forbidden
        )
        blocked_requested = tuple(
            capability for capability in requested if capability in forbidden
        )
        if blocked_requested:
            assumptions.append("requested_capabilities_cannot_override_forbids")
        approval_required = any(grant.approval_required for grant in grants)
        blocked_actions = _blocked_actions(lanes, forbidden)

        return DbSafetyFrame(
            prompt=prompt,
            normalized_prompt=normalized,
            explicit_schema_only=explicit_schema_only,
            direct_memory_operation=direct_memory_operation,
            sql_statement_type=statement_type,
            has_db_target=has_db_target,
            has_mutation_payload=has_mutation_payload,
            explicit_execution=explicit_execution,
            destructive=destructive,
            admin=admin,
            monitor_operation=monitor_operation,
            requested_capabilities=requested,
            granted_lanes=tuple(lanes),
            forbidden_capabilities=forbidden,
            rewrites=tuple(rewrites),
            assumptions=_ordered_unique(assumptions),
            lane_grants=grants,
            required_capabilities=required,
            approval_required=approval_required,
            blocked_actions=blocked_actions,
        )


_SQL_PREFIX = r"(?:run|execute|apply|commit)\s+"
_READ_SQL = {"select", "with"}
_WRITE_SQL = {"insert", "update", "delete", "merge"}
_ADMIN_SQL = {"alter", "create", "drop", "grant", "revoke", "truncate"}
_ROW_OR_WRITE_SQL = _READ_SQL | _WRITE_SQL | _ADMIN_SQL

_LANE_GRANTS: Mapping[DbCapabilityLane, DbLaneGrant] = {
    DbCapabilityLane.NONE: DbLaneGrant(
        lane=DbCapabilityLane.NONE,
        required_capabilities=(),
        forbidden_capabilities=("db.sql.execute_read", "db.sql.execute_write"),
        approval_required=False,
        reason="no_deterministic_db_lane",
    ),
    DbCapabilityLane.SCHEMA: DbLaneGrant(
        lane=DbCapabilityLane.SCHEMA,
        required_capabilities=(
            "catalog.schema.search",
            "catalog.asset.inspect",
            "db.schema.inspect",
        ),
        forbidden_capabilities=("db.sql.execute_read", "db.sql.execute_write"),
        approval_required=False,
        reason="schema_or_metadata_only",
    ),
    DbCapabilityLane.MEMORY_ANSWER: DbLaneGrant(
        lane=DbCapabilityLane.MEMORY_ANSWER,
        required_capabilities=(
            "memory.semantic.recall",
            "db.memory.answer_context.build",
        ),
        forbidden_capabilities=(
            "db.sql.validate",
            "db.sql.execute_read",
            "db.sql.execute_write",
        ),
        approval_required=False,
        reason="direct_memory_answer",
    ),
    DbCapabilityLane.MEMORY_WRITE: DbLaneGrant(
        lane=DbCapabilityLane.MEMORY_WRITE,
        required_capabilities=("db.memory.plan_update", "db.memory.commit_update"),
        forbidden_capabilities=(
            "db.sql.validate",
            "db.sql.execute_read",
            "db.sql.execute_write",
        ),
        approval_required=False,
        reason="direct_memory_write",
    ),
    DbCapabilityLane.READ: DbLaneGrant(
        lane=DbCapabilityLane.READ,
        required_capabilities=("db.sql.validate", "db.sql.execute_read"),
        forbidden_capabilities=("db.sql.execute_write",),
        approval_required=False,
        reason="row_read_allowed",
    ),
    DbCapabilityLane.WRITE_PROPOSE: DbLaneGrant(
        lane=DbCapabilityLane.WRITE_PROPOSE,
        required_capabilities=("db.sql.validate",),
        forbidden_capabilities=("db.sql.execute_write",),
        approval_required=True,
        reason="write_proposal_only",
    ),
    DbCapabilityLane.WRITE_EXECUTE: DbLaneGrant(
        lane=DbCapabilityLane.WRITE_EXECUTE,
        required_capabilities=("db.sql.validate", "db.sql.execute_write"),
        forbidden_capabilities=(),
        approval_required=True,
        reason="explicit_write_execution",
    ),
    DbCapabilityLane.ADMIN: DbLaneGrant(
        lane=DbCapabilityLane.ADMIN,
        required_capabilities=("db.admin.propose",),
        forbidden_capabilities=("db.sql.execute_read", "db.sql.execute_write"),
        approval_required=True,
        reason="explicit_admin_sql_structure",
    ),
    DbCapabilityLane.MONITOR_READ: DbLaneGrant(
        lane=DbCapabilityLane.MONITOR_READ,
        required_capabilities=("db.monitor.inspect",),
        forbidden_capabilities=(
            "db.monitor.commit_create",
            "db.monitor.commit_lifecycle",
            "monitor.delivery.local",
            "monitor.delivery.in_app",
            "db.sql.execute_write",
        ),
        approval_required=False,
        reason="explicit_monitor_read",
    ),
    DbCapabilityLane.MONITOR_WRITE: DbLaneGrant(
        lane=DbCapabilityLane.MONITOR_WRITE,
        required_capabilities=(
            "db.monitor.plan_create",
            "db.monitor.plan_lifecycle",
        ),
        forbidden_capabilities=("db.sql.execute_write",),
        approval_required=True,
        reason="explicit_monitor_control_plane_mutation",
    ),
    DbCapabilityLane.MONITOR_EXECUTE: DbLaneGrant(
        lane=DbCapabilityLane.MONITOR_EXECUTE,
        required_capabilities=("db.monitor.execute",),
        forbidden_capabilities=(
            "db.monitor.commit_create",
            "db.monitor.commit_lifecycle",
            "db.sql.execute_write",
        ),
        approval_required=True,
        reason="explicit_monitor_execution",
    ),
}


def _request_prompt(request: str | Any) -> str:
    if isinstance(request, str):
        return request
    return str(getattr(request, "prompt", "") or "")


def _request_requested_capabilities(request: str | Any) -> tuple[str, ...]:
    if isinstance(request, str):
        return ()
    capabilities = getattr(request, "requested_capabilities", ()) or ()
    return tuple(str(capability) for capability in capabilities)


def _request_mode(request: str | Any) -> str | None:
    if isinstance(request, str):
        return None
    value = getattr(request, "mode", None)
    return str(value) if value else None


def _normalize_prompt(prompt: str) -> str:
    lowered = prompt.lower().strip()
    lowered = re.sub(r"[`]", "", lowered)
    return " ".join(lowered.split())


def _extract_sql_shape(text: str) -> dict[str, Any]:
    statement_type, sql_text = _sql_statement(text)
    tables: list[str] = []
    if statement_type == "update":
        table_match = re.search(r"\bupdate\s+([a-z_][\w.]*)\s+set\b", sql_text)
        if table_match:
            tables.append(table_match.group(1))
    elif statement_type == "insert":
        table_match = re.search(r"\binsert\s+into\s+([a-z_][\w.]*)", sql_text)
        if table_match:
            tables.append(table_match.group(1))
    elif statement_type == "delete":
        table_match = re.search(r"\bdelete\s+from\s+([a-z_][\w.]*)", sql_text)
        if table_match:
            tables.append(table_match.group(1))
    elif statement_type in {"select", "with"}:
        tables.extend(re.findall(r"\b(?:from|join)\s+([a-z_][\w.]*)", sql_text))
    elif statement_type in _ADMIN_SQL:
        table_match = _admin_target_match(statement_type, sql_text)
        if table_match:
            tables.append(table_match.group(1))
    return {"statement_type": statement_type, "tables": _ordered_unique(tables)}


def _sql_statement(text: str) -> tuple[str | None, str]:
    prefix = rf"(?:^|[:;]\s*|\b(?:please|and|then)\s+)(?:{_SQL_PREFIX})?"
    sql_patterns = (
        ("select", rf"{prefix}select\b.+\bfrom\b"),
        ("with", rf"{prefix}with\b.+\bselect\b"),
        ("insert", rf"{prefix}insert\s+into\s+[a-z_][\w.]*\b"),
        ("update", rf"{prefix}update\s+[a-z_][\w.]*\s+set\b"),
        ("delete", rf"{prefix}delete\s+from\s+[a-z_][\w.]*\b"),
        ("merge", rf"{prefix}merge\s+into\s+[a-z_][\w.]*\b"),
        ("alter", rf"{prefix}alter\s+(table|view|schema|database|index)\b"),
        ("create", rf"{prefix}create\s+(table|view|schema|database|index)\b"),
        ("drop", rf"{prefix}drop\s+(table|view|schema|database|index)\b"),
        ("grant", rf"{prefix}grant\b.+\bon\b"),
        ("revoke", rf"{prefix}revoke\b.+\bon\b"),
        ("truncate", rf"{prefix}truncate\s+(table\s+)?[a-z_][\w.]*\b"),
    )
    for statement_type, pattern in sql_patterns:
        match = re.search(pattern, text)
        if match:
            return statement_type, text[match.start() :]
    return None, text


def _admin_target_match(statement_type: str, text: str) -> re.Match[str] | None:
    if statement_type == "truncate":
        return re.search(r"\btruncate\s+(?:table\s+)?([a-z_][\w.]*)", text)
    return re.search(
        r"\b(?:table|view|schema|database|index)\s+(?:if\s+(?:not\s+)?exists\s+)?([a-z_][\w.]*)",
        text,
    )


def _explicit_execution(text: str) -> bool:
    return bool(re.search(r"\b(run|execute|apply|commit)\b", text))


def _explicit_schema_only(text: str) -> bool:
    return bool(
        re.search(
            r"\b(schema only|metadata only|schema evidence only|do not query rows|don't query rows|without querying rows|no row data)\b",
            text,
        )
    )


def _direct_memory_operation(text: str) -> str | None:
    if re.search(
        r"\b(what do you remember|do you remember|recall (?:what )?memor(?:y|ies))\b",
        text,
    ):
        return "recall"
    if re.search(r"\b(list|show)\b.*\b(memories|memory|definitions|rules)\b", text):
        return "list"
    if re.search(r"\b(inspect|explain)\b.*\b(memories|memory)\b", text):
        return "inspect"
    if re.search(r"\b(remember|note)\s+(that|this:|this\s+|[a-z_][\w-]+)\b", text):
        return "update"
    if re.search(
        r"\b(forget|delete|remove|update|replace|change)\b.*\b(memories|memory)\b",
        text,
    ):
        return "update"
    return None


def _monitor_operation(text: str) -> str | None:
    if re.search(
        r"\b(run|trigger|execute)\b.*\bmonitors?\b.*\b(now|today|immediately)?\b",
        text,
    ):
        return "execute"
    if re.search(r"\bsend\b.*\bmonitor report\b.*\bnow\b", text):
        return "execute"
    if re.search(r"\b(list|show|inspect|describe)\b.*\bmonitors?\b", text):
        return "read"
    if re.search(r"\bmonitor status\b|\bmonitors?\b.*\bstatus\b", text):
        return "read"
    if re.search(r"\b(create|add|set up|setup)\b.*\bmonitors?\b", text):
        return "write"
    if re.search(
        r"\b(pause|resume|restart|unpause|delete|remove|update)\b.*\bmonitors?\b", text
    ):
        return "write"
    if re.search(r"\b(schedule|alert me|notify me)\b.*\bmonitors?\b", text):
        return "write"
    if re.search(r"\b(alert|notify)\s+me\b.*\b(when|if)\b", text):
        return "write"
    if re.search(r"\bwatch\b.+\b(alert|notify)\s+me\b", text):
        return "write"
    return None


def _natural_schema_request(text: str) -> bool:
    if _explicit_schema_only(text):
        return True
    return bool(
        re.search(
            r"\b(what|which|show|list|describe|inspect)\b.*\b(columns?|fields?|tables?|schema|metadata)\b",
            text,
        )
        or re.search(r"\btell me about\b.*\b(tables?|views?)\b", text)
        or re.search(r"\brelationships?\b.*\b(connects?|between|from|to)\b", text)
        or re.search(r"\brelationships?\b.*\bjoin\b", text)
        or re.search(r"\b(connects?|relationship)\b.*\b(tables?|assets?)\b", text)
        or re.search(r"\b(columns?|fields?)\b.*\b(in|for|on)\b", text)
        or re.search(r"\bupdate me on the schema\b", text)
    )


def _natural_row_read_request(text: str) -> bool:
    if _explicit_schema_only(text) or _direct_memory_operation(text):
        return False
    if _monitor_operation(text) is not None:
        return False
    if re.search(r"\brelationships?\b.*\bjoin\b", text):
        return False
    return bool(
        re.search(r"\bhow many\b.*\b\w+\b", text)
        or re.search(r"\b(count|total|sum|average|avg)\b.*\b\w+\b", text)
        or re.search(
            r"\b(calculate|calculated|compute|computed|aggregate)\b.*\b\w+\b", text
        )
        or re.search(r"\b\w+\b.*\b(be|is|are)\s+(calculated|computed)\b", text)
        or re.search(r"\b(show|list|find)\b.*\b(recent|latest|rows?|records?)\b", text)
        or re.search(
            r"\b(show|list|find)\b.*\b(orders?|transactions?|customers?|tickets?)\b",
            text,
        )
        or re.search(r"\b(show|list|find|analy[sz]e|investigate)\b.*\bqueries?\b", text)
        or re.search(r"\bmulti-step\b.*\b(analysis|queries?)\b", text)
        or re.search(r"\bjoin\b.+\b(to|with)\b", text)
    )


def _underspecified_write(text: str) -> bool:
    if _monitor_operation(text) is not None or _direct_memory_operation(text):
        return False
    if re.search(r"\b(write up|write me|drop me|update me)\b", text):
        return False
    return bool(
        re.search(r"\b(update|delete|insert|merge|drop|truncate)\b", text)
        and re.search(r"\b(orders?|customers?|transactions?|rows?|records?)\b", text)
    )


def _has_mutation_payload(statement_type: str | None, text: str) -> bool:
    if statement_type == "update":
        return bool(re.search(r"\bset\b\s+\w+", text))
    if statement_type == "delete":
        return bool(re.search(r"\bdelete\s+from\b", text))
    if statement_type == "insert":
        return bool(re.search(r"\b(values|select)\b", text))
    if statement_type == "merge":
        return bool(re.search(r"\busing\b", text))
    return False


def _destructive_sql(statement_type: str | None) -> bool:
    return statement_type in {"delete", "drop", "truncate", "alter"}


def _admin_sql(statement_type: str | None, text: str) -> bool:
    if statement_type not in _ADMIN_SQL:
        return False
    if statement_type in {"drop", "truncate", "alter"}:
        return _admin_target_match(statement_type, text) is not None
    return True


def _sql_lane(
    statement_type: str | None,
    *,
    explicit_execution: bool,
) -> DbCapabilityLane:
    if statement_type in _ADMIN_SQL:
        return DbCapabilityLane.ADMIN
    if statement_type in _WRITE_SQL:
        if explicit_execution:
            return DbCapabilityLane.WRITE_EXECUTE
        return DbCapabilityLane.WRITE_PROPOSE
    if statement_type in _READ_SQL:
        return DbCapabilityLane.READ
    return DbCapabilityLane.NONE


def _monitor_lane(operation: str) -> DbCapabilityLane:
    if operation == "read":
        return DbCapabilityLane.MONITOR_READ
    if operation == "write":
        return DbCapabilityLane.MONITOR_WRITE
    return DbCapabilityLane.MONITOR_EXECUTE


def _apply_requested_capabilities(
    lanes: list[DbCapabilityLane],
    requested: tuple[str, ...],
) -> tuple[list[DbCapabilityLane], list[DbSafetyRewrite]]:
    rewrites: list[DbSafetyRewrite] = []
    current = set(lanes)
    for capability in requested:
        lane = _lane_from_requested_capability(capability)
        if lane is None or lane in current:
            continue
        forbidden = {
            item
            for existing in current
            for item in _grant_for_lane(existing).forbidden_capabilities
        }
        if capability in forbidden:
            rewrites.append(
                DbSafetyRewrite(
                    rule="requested_capability_blocked_by_forbid",
                    from_lanes=(lane,),
                    to_lanes=tuple(_ordered_lanes(lanes)),
                    reason=capability,
                )
            )
            continue
        lanes.append(lane)
        current.add(lane)
    return list(_ordered_lanes(lanes)), rewrites


def _lane_from_requested_capability(capability: str) -> DbCapabilityLane | None:
    if capability in {
        "catalog.schema.search",
        "catalog.asset.inspect",
        "db.schema.inspect",
    }:
        return DbCapabilityLane.SCHEMA
    if capability == "db.sql.execute_read":
        return DbCapabilityLane.READ
    if capability == "db.sql.execute_write":
        return DbCapabilityLane.WRITE_EXECUTE
    if capability in {"memory.semantic.recall", "db.memory.answer_context.build"}:
        return DbCapabilityLane.MEMORY_ANSWER
    if capability in {"db.memory.plan_update", "db.memory.commit_update"}:
        return DbCapabilityLane.MEMORY_WRITE
    if capability.startswith("db.monitor.") or capability.startswith("monitor."):
        if "commit" in capability or "plan_" in capability:
            return DbCapabilityLane.MONITOR_WRITE
        if "delivery" in capability or "execute" in capability:
            return DbCapabilityLane.MONITOR_EXECUTE
        return DbCapabilityLane.MONITOR_READ
    if capability.startswith("db.admin."):
        return DbCapabilityLane.ADMIN
    return None


def _grant_for_lane(lane: DbCapabilityLane) -> DbLaneGrant:
    return _LANE_GRANTS[lane]


def _blocked_actions(
    lanes: list[DbCapabilityLane],
    forbidden: tuple[str, ...],
) -> tuple[str, ...]:
    lane_set = set(lanes)
    blocked: list[str] = []
    if DbCapabilityLane.READ not in lane_set:
        blocked.append("row_read")
    if "db.sql.execute_write" in forbidden:
        blocked.append("write_execute")
    if DbCapabilityLane.ADMIN not in lane_set:
        blocked.append("admin")
    if not (
        {DbCapabilityLane.MONITOR_WRITE, DbCapabilityLane.MONITOR_EXECUTE} & lane_set
    ):
        blocked.append("monitor_action")
    return tuple(blocked)


def _ordered_lanes(lanes: Iterable[DbCapabilityLane]) -> tuple[DbCapabilityLane, ...]:
    order = tuple(DbCapabilityLane)
    unique = set(lanes)
    if len(unique) > 1 and DbCapabilityLane.NONE in unique:
        unique.remove(DbCapabilityLane.NONE)
    return tuple(lane for lane in order if lane in unique)


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return tuple(out)
