"""Build the record a backend receives after the read policy passed."""

from __future__ import annotations

from daita.domains.data.sql import CheckedRead

SOURCE_REVISION = "sha256:" + "3" * 64
RESOURCE_REVISION = "sha256:" + "2" * 64


def checked_read(
    sql: str,
    *,
    parameters: tuple[object, ...] = (),
    functions: tuple[str, ...] = (),
    resource_id: str = "resource-orders",
    name: str = "orders",
    sensitivity: str = "internal",
) -> CheckedRead:
    return CheckedRead(
        canonical_sql=sql,
        sql_fingerprint="sha256:" + "1" * 64,
        parameters=parameters,
        function_lookups=functions,
        resource_ids=(resource_id,),
        resource_names=((resource_id, name),),
        resource_revisions=((resource_id, RESOURCE_REVISION),),
        source_revision=SOURCE_REVISION,
        sensitivity_classes=(sensitivity,),
    )
