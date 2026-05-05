"""Generic data operation assertions."""

from __future__ import annotations

from ..analysis import RunEvidence
from ..config import Expectations
from ..models import AssertionResult
from .common import fail, matches, matches_any


def operation_assertions(
    exp: Expectations, evidence: RunEvidence
) -> list[AssertionResult]:
    operations = evidence.operations
    categories = [op.category for op in operations]
    resources = [op.resource for op in operations if op.resource]
    results = []
    for index, category in enumerate(exp.operations.required_categories):
        if category not in categories:
            results.append(
                fail(
                    f"operations.required_categories[{index}]",
                    "required_operation_missing",
                    f"Required operation category missing: {category}.",
                    f"expectations.operations.required_categories[{index}]",
                    observed=categories,
                    expected=category,
                )
            )
    for index, category in enumerate(exp.operations.forbidden_categories):
        if category in categories:
            call_indexes = [
                op.call_index for op in operations if op.category == category
            ]
            results.append(
                fail(
                    f"operations.forbidden_categories[{index}]",
                    "forbidden_operation_category",
                    f"Forbidden operation category used: {category}.",
                    f"expectations.operations.forbidden_categories[{index}]",
                    observed=category,
                    expected=f"not {category}",
                    related_tool_calls=call_indexes,
                )
            )
    for index, resource in enumerate(exp.operations.required_resources):
        if not matches_any(resource, resources):
            results.append(
                fail(
                    f"operations.required_resources[{index}]",
                    "required_resource_missing",
                    f"Required resource was not touched: {resource}.",
                    f"expectations.operations.required_resources[{index}]",
                    observed=resources,
                    expected=resource,
                )
            )
    for index, resource in enumerate(exp.operations.forbidden_resources):
        matched_calls = [
            op.call_index
            for op in operations
            if op.resource and matches(resource, op.resource)
        ]
        if matched_calls:
            results.append(
                fail(
                    f"operations.forbidden_resources[{index}]",
                    "forbidden_resource_used",
                    f"Forbidden resource was touched: {resource}.",
                    f"expectations.operations.forbidden_resources[{index}]",
                    observed=resource,
                    expected=f"not {resource}",
                    related_tool_calls=matched_calls,
                )
            )
    writes = [op for op in operations if op.action == "write"]
    deletes = [op for op in operations if op.action == "delete"]
    if (
        exp.operations.max_write_operations is not None
        and len(writes) > exp.operations.max_write_operations
    ):
        results.append(
            fail(
                "operations.max_write_operations",
                "too_many_write_operations",
                "Too many write operations were used.",
                "expectations.operations.max_write_operations",
                observed=len(writes),
                expected=exp.operations.max_write_operations,
                related_tool_calls=[op.call_index for op in writes],
            )
        )
    if (
        exp.operations.max_delete_operations is not None
        and len(deletes) > exp.operations.max_delete_operations
    ):
        results.append(
            fail(
                "operations.max_delete_operations",
                "too_many_delete_operations",
                "Too many delete operations were used.",
                "expectations.operations.max_delete_operations",
                observed=len(deletes),
                expected=exp.operations.max_delete_operations,
                related_tool_calls=[op.call_index for op in deletes],
            )
        )
    return results


def file_assertions(exp: Expectations, evidence: RunEvidence) -> list[AssertionResult]:
    file_ops = [op for op in evidence.operations if op.category == "file"]
    paths = [op.path or op.resource or "" for op in file_ops]
    results = []
    for index, path in enumerate(exp.files.required_read):
        if not any(
            op.action == "read" and matches(path, op.path or op.resource or "")
            for op in file_ops
        ):
            results.append(
                fail(
                    f"files.required_read[{index}]",
                    "required_file_not_read",
                    f"Required file was not read: {path}.",
                    f"expectations.files.required_read[{index}]",
                    observed=paths,
                    expected=path,
                )
            )
    for index, path in enumerate(exp.files.forbidden_read):
        matched_calls = [
            op.call_index
            for op in file_ops
            if op.action == "read" and matches(path, op.path or op.resource or "")
        ]
        if matched_calls:
            results.append(
                fail(
                    f"files.forbidden_read[{index}]",
                    "forbidden_file_read",
                    f"Forbidden file was read: {path}.",
                    f"expectations.files.forbidden_read[{index}]",
                    observed=path,
                    expected=f"not {path}",
                    related_tool_calls=matched_calls,
                )
            )
    for index, path in enumerate(exp.files.forbidden_write):
        matched_calls = [
            op.call_index
            for op in file_ops
            if op.action in {"write", "delete"}
            and matches(path, op.path or op.resource or "")
        ]
        if matched_calls:
            results.append(
                fail(
                    f"files.forbidden_write[{index}]",
                    "forbidden_file_write",
                    f"Forbidden file was written: {path}.",
                    f"expectations.files.forbidden_write[{index}]",
                    observed=path,
                    expected=f"not {path}",
                    related_tool_calls=matched_calls,
                )
            )
    if exp.files.must_use_schema and not any(
        "schema" in op.tool_name.lower() or "inspect" in op.tool_name.lower()
        for op in file_ops
    ):
        results.append(
            fail(
                "files.must_use_schema",
                "file_schema_not_used",
                "No file schema/inspection operation was observed.",
                "expectations.files.must_use_schema",
                observed=[op.tool_name for op in file_ops],
                expected="schema inspection",
            )
        )
    return results


def api_assertions(exp: Expectations, evidence: RunEvidence) -> list[AssertionResult]:
    api_ops = [op for op in evidence.operations if op.category == "api"]
    methods = [(op.method or "").upper() for op in api_ops]
    hosts = [op.host or "" for op in api_ops]
    results = []
    for index, method in enumerate(exp.api.required_methods):
        if method.upper() not in methods:
            results.append(
                fail(
                    f"api.required_methods[{index}]",
                    "required_api_method_missing",
                    f"Required API method missing: {method}.",
                    f"expectations.api.required_methods[{index}]",
                    observed=methods,
                    expected=method.upper(),
                )
            )
    for index, method in enumerate(exp.api.forbidden_methods):
        method_upper = method.upper()
        if method_upper in methods:
            results.append(
                fail(
                    f"api.forbidden_methods[{index}]",
                    "forbidden_api_method",
                    f"Forbidden API method used: {method_upper}.",
                    f"expectations.api.forbidden_methods[{index}]",
                    observed=method_upper,
                    expected=f"not {method_upper}",
                )
            )
    for index, host in enumerate(exp.api.required_hosts):
        if not matches_any(host, hosts):
            results.append(
                fail(
                    f"api.required_hosts[{index}]",
                    "required_api_host_missing",
                    f"Required API host missing: {host}.",
                    f"expectations.api.required_hosts[{index}]",
                    observed=hosts,
                    expected=host,
                )
            )
    for index, host in enumerate(exp.api.forbidden_hosts):
        matched_calls = [
            op.call_index for op in api_ops if matches(host, op.host or "")
        ]
        if matched_calls:
            results.append(
                fail(
                    f"api.forbidden_hosts[{index}]",
                    "forbidden_api_host",
                    f"Forbidden API host used: {host}.",
                    f"expectations.api.forbidden_hosts[{index}]",
                    observed=host,
                    expected=f"not {host}",
                    related_tool_calls=matched_calls,
                )
            )
    return results


def storage_assertions(
    exp: Expectations, evidence: RunEvidence
) -> list[AssertionResult]:
    storage_ops = [op for op in evidence.operations if op.category == "storage"]
    buckets = [op.bucket or "" for op in storage_ops]
    results = []
    for index, bucket in enumerate(exp.storage.required_buckets):
        if not matches_any(bucket, buckets):
            results.append(
                fail(
                    f"storage.required_buckets[{index}]",
                    "required_bucket_missing",
                    f"Required storage bucket missing: {bucket}.",
                    f"expectations.storage.required_buckets[{index}]",
                    observed=buckets,
                    expected=bucket,
                )
            )
    for index, bucket in enumerate(exp.storage.forbidden_buckets):
        matched_calls = [
            op.call_index for op in storage_ops if matches(bucket, op.bucket or "")
        ]
        if matched_calls:
            results.append(
                fail(
                    f"storage.forbidden_buckets[{index}]",
                    "forbidden_bucket_used",
                    f"Forbidden storage bucket used: {bucket}.",
                    f"expectations.storage.forbidden_buckets[{index}]",
                    observed=bucket,
                    expected=f"not {bucket}",
                    related_tool_calls=matched_calls,
                )
            )
    if exp.storage.forbidden_write:
        writes = [op for op in storage_ops if op.action in {"write", "delete"}]
        if writes:
            results.append(
                fail(
                    "storage.forbidden_write",
                    "forbidden_storage_write",
                    "Storage write/delete operation was used.",
                    "expectations.storage.forbidden_write",
                    observed=[op.tool_name for op in writes],
                    expected="read-only",
                    related_tool_calls=[op.call_index for op in writes],
                )
            )
    return results


def vector_assertions(
    exp: Expectations, evidence: RunEvidence
) -> list[AssertionResult]:
    vector_ops = [op for op in evidence.operations if op.category == "vector"]
    results = []
    if exp.vector.max_top_k is not None:
        offenders = [
            op
            for op in vector_ops
            if op.top_k is not None and op.top_k > exp.vector.max_top_k
        ]
        if offenders:
            results.append(
                fail(
                    "vector.max_top_k",
                    "vector_top_k_over_budget",
                    "Vector search top_k exceeded the configured maximum.",
                    "expectations.vector.max_top_k",
                    observed=[op.top_k for op in offenders],
                    expected=exp.vector.max_top_k,
                    related_tool_calls=[op.call_index for op in offenders],
                )
            )
    for index, required in enumerate(exp.vector.required_filters):
        if not any(matches_any(required, op.filters) for op in vector_ops):
            results.append(
                fail(
                    f"vector.required_filters[{index}]",
                    "vector_required_filter_missing",
                    f"Required vector filter missing: {required}.",
                    f"expectations.vector.required_filters[{index}]",
                    observed=[op.filters for op in vector_ops],
                    expected=required,
                )
            )
    return results
