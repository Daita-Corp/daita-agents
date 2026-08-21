"""Exact durable codec for independently keyed MCP binding aggregates."""

from __future__ import annotations

from ..._json import FrozenJsonObject
from ...adapters.mcp import (
    MCPAuthentication,
    MCPAuthenticationMode,
    MCPBindingState,
    MCPServerBinding,
    MCPToolBinding,
)
from ...capabilities import ToolDiscoveryMetadata, ToolExposureClass
from ...llm.models import ModelSensitivity
from ...security import SecretReference
from .common import (
    datetime_decode,
    datetime_encode,
    dump_payload,
    integer,
    load_payload,
    mapping,
    optional_datetime_decode,
    optional_datetime_encode,
    optional_text,
    plain_decode,
    plain_encode,
    record,
    record_fields,
    sequence,
    text,
)

_MCP_BINDING_VERSION = 1


def encode_mcp_binding(value: MCPServerBinding) -> str:
    if not isinstance(value, MCPServerBinding):
        raise TypeError("MCP binding codec requires MCPServerBinding")
    return dump_payload(
        record(
            "MCPServerBinding",
            {
                "version": _MCP_BINDING_VERSION,
                "endpoint": value.endpoint,
                "authentication_mode": value.authentication.mode.value,
                "secret_reference": (
                    None
                    if value.authentication.secret_reference is None
                    else value.authentication.secret_reference.to_uri()
                ),
                "protocol_version": value.protocol_version,
                "server_name": value.server_name,
                "server_version": value.server_version,
                "local_label": value.local_label,
                "maximum_outbound_sensitivity": (
                    value.maximum_outbound_sensitivity.value
                ),
                "tools": [_encode_tool(tool) for tool in value.tools],
                "state": value.state.value,
                "revision": value.revision,
                "admitted_at": datetime_encode(value.admitted_at),
                "last_checked_at": datetime_encode(value.last_checked_at),
                "revoked_at": optional_datetime_encode(value.revoked_at),
                "stale_reason": value.stale_reason,
            },
        )
    )


def decode_mcp_binding(
    value: str,
    *,
    agent_id: str,
    binding_id: str,
) -> MCPServerBinding:
    fields = record_fields(
        load_payload(value),
        "MCPServerBinding",
        (
            "version",
            "endpoint",
            "authentication_mode",
            "secret_reference",
            "protocol_version",
            "server_name",
            "server_version",
            "local_label",
            "maximum_outbound_sensitivity",
            "tools",
            "state",
            "revision",
            "admitted_at",
            "last_checked_at",
            "revoked_at",
            "stale_reason",
        ),
    )
    if integer(fields["version"], "MCP binding version") != _MCP_BINDING_VERSION:
        raise ValueError("stored MCP binding version is unsupported")
    try:
        authentication_mode = MCPAuthenticationMode(
            text(fields["authentication_mode"], "MCP authentication mode")
        )
        maximum_outbound = ModelSensitivity(
            text(
                fields["maximum_outbound_sensitivity"],
                "MCP maximum outbound sensitivity",
            )
        )
        state = MCPBindingState(text(fields["state"], "MCP binding state"))
    except ValueError:
        raise ValueError("stored MCP binding enum is invalid") from None
    secret_uri = optional_text(fields["secret_reference"], "MCP secret reference")
    authentication = MCPAuthentication(
        authentication_mode,
        None if secret_uri is None else SecretReference.parse(secret_uri),
    )
    tools = tuple(
        _decode_tool(item) for item in sequence(fields["tools"], "MCP binding tools")
    )
    return MCPServerBinding(
        binding_id=binding_id,
        agent_id=agent_id,
        endpoint=text(fields["endpoint"], "MCP endpoint"),
        authentication=authentication,
        protocol_version=text(fields["protocol_version"], "MCP protocol version"),
        server_name=text(fields["server_name"], "MCP server name"),
        server_version=text(fields["server_version"], "MCP server version"),
        local_label=text(fields["local_label"], "MCP local server label"),
        maximum_outbound_sensitivity=maximum_outbound,
        tools=tools,
        state=state,
        revision=integer(fields["revision"], "MCP binding revision"),
        admitted_at=datetime_decode(fields["admitted_at"]),
        last_checked_at=datetime_decode(fields["last_checked_at"]),
        revoked_at=optional_datetime_decode(fields["revoked_at"]),
        stale_reason=optional_text(fields["stale_reason"], "MCP stale reason"),
    )


def _encode_tool(value: MCPToolBinding):
    return record(
        "MCPToolBinding",
        {
            "capability_id": value.capability_id,
            "executor_id": value.executor_id,
            "local_name": value.local_name,
            "remote_name": value.remote_name,
            "description": value.description,
            "discovery_summary": value.discovery.summary,
            "discovery_when_to_use": value.discovery.when_to_use,
            "discovery_keywords": list(value.discovery.keywords),
            "discovery_exposure_class": value.discovery.exposure_class.value,
            "discovery_eager_priority": value.discovery.eager_priority,
            "input_schema": plain_encode(value.input_schema),
            "input_schema_digest": value.input_schema_digest,
            "output_schema": (
                None
                if value.output_schema is None
                else plain_encode(value.output_schema)
            ),
            "output_schema_digest": value.output_schema_digest,
            "result_sensitivity": value.result_sensitivity.value,
        },
    )


def _decode_tool(value) -> MCPToolBinding:
    fields = record_fields(
        value,
        "MCPToolBinding",
        (
            "capability_id",
            "executor_id",
            "local_name",
            "remote_name",
            "description",
            "discovery_summary",
            "discovery_when_to_use",
            "discovery_keywords",
            "discovery_exposure_class",
            "discovery_eager_priority",
            "input_schema",
            "input_schema_digest",
            "output_schema",
            "output_schema_digest",
            "result_sensitivity",
        ),
    )
    input_schema = plain_decode(fields["input_schema"])
    if not isinstance(input_schema, dict):
        raise ValueError("stored MCP input schema is invalid")
    output_raw = fields["output_schema"]
    output_schema = None if output_raw is None else plain_decode(output_raw)
    if output_schema is not None and not isinstance(output_schema, dict):
        raise ValueError("stored MCP output schema is invalid")
    try:
        sensitivity = ModelSensitivity(
            text(fields["result_sensitivity"], "MCP result sensitivity")
        )
        exposure_class = ToolExposureClass(
            text(fields["discovery_exposure_class"], "MCP exposure class")
        )
    except ValueError:
        raise ValueError("stored MCP tool enum is invalid") from None
    return MCPToolBinding(
        capability_id=text(fields["capability_id"], "MCP capability id"),
        executor_id=text(fields["executor_id"], "MCP executor id"),
        local_name=text(fields["local_name"], "MCP local name"),
        remote_name=text(fields["remote_name"], "MCP remote name"),
        description=text(fields["description"], "MCP tool description"),
        discovery=ToolDiscoveryMetadata(
            summary=text(fields["discovery_summary"], "MCP discovery summary"),
            when_to_use=text(
                fields["discovery_when_to_use"],
                "MCP discovery when_to_use",
            ),
            keywords=tuple(
                text(item, "MCP discovery keyword")
                for item in sequence(
                    fields["discovery_keywords"],
                    "MCP discovery keywords",
                )
            ),
            exposure_class=exposure_class,
            eager_priority=integer(
                fields["discovery_eager_priority"],
                "MCP discovery eager priority",
            ),
        ),
        input_schema=FrozenJsonObject.from_mapping(input_schema),
        input_schema_digest=text(
            fields["input_schema_digest"], "MCP input schema digest"
        ),
        output_schema=(
            None
            if output_schema is None
            else FrozenJsonObject.from_mapping(output_schema)
        ),
        output_schema_digest=optional_text(
            fields["output_schema_digest"], "MCP output schema digest"
        ),
        result_sensitivity=sensitivity,
    )


__all__ = ["decode_mcp_binding", "encode_mcp_binding"]
