"""Lifecycle boundary shared by durable backend conformance tests."""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from daita.storage.protocols import StateStore

# Each call opens a separate handle to the same disposable agent state. Exiting
# closes that handle; a later call must observe previously committed records.
StateStoreFactory = Callable[[], AbstractAsyncContextManager[StateStore]]
