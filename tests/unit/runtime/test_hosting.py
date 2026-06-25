from daita.runtime import (
    HostRuntimeContext,
    current_host_runtime_context,
    host_runtime_context,
)


def test_host_runtime_context_scopes_and_resets():
    outer = HostRuntimeContext(
        surface="web_app",
        delivery_defaults=["in_app"],
        services={"notification": object()},
        runtime_extensions=["extension"],
        metadata={"organization_id": "org-1"},
    )
    inner = HostRuntimeContext(surface="worker")

    assert current_host_runtime_context() is None
    with host_runtime_context(outer):
        assert current_host_runtime_context() == outer
        assert outer.delivery_defaults == ("in_app",)
        assert outer.runtime_extensions == ("extension",)
        with host_runtime_context(inner):
            assert current_host_runtime_context() == inner
        assert current_host_runtime_context() == outer
    assert current_host_runtime_context() is None
