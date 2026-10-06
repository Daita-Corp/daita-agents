"""Transactional in-memory dependency for the real advisory document owners."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

from daita.storage.advisory import (
    AdvisoryCollection,
    AdvisoryDocument,
    AdvisoryTransaction,
)
from daita.storage.errors import StorageError


class MemoryAdvisoryStorage:
    def __init__(self) -> None:
        self.documents: dict[tuple[AdvisoryCollection, str], AdvisoryDocument] = {}
        self.lock = threading.RLock()

    @contextmanager
    def transaction(self, *, write: bool = False) -> Iterator[AdvisoryTransaction]:
        with self.lock:
            transaction = _Transaction(dict(self.documents), write)
            yield transaction
            if write:
                self.documents = transaction.documents


class _Transaction:
    def __init__(self, documents, write: bool) -> None:
        self.documents = documents
        self.write = write

    def get(self, collection, name, *, max_bytes):
        document = self.documents.get((collection, name))
        if document is not None and len(document.content) > max_bytes:
            raise StorageError("document exceeds read bound")
        return document

    def scan(self, collection, *, max_count, max_bytes):
        result = {
            name: item
            for (group, name), item in self.documents.items()
            if group == collection
        }
        if len(result) > max_count or any(
            len(item.content) > max_bytes for item in result.values()
        ):
            raise StorageError("collection exceeds read bound")
        return result

    def usage(self, collection):
        values = [
            item for (group, _), item in self.documents.items() if group == collection
        ]
        return len(values), sum(len(item.content) for item in values)

    def put(self, collection, name, content):
        assert self.write
        self.documents[(collection, name)] = AdvisoryDocument(content, uuid4().hex)

    def delete(self, collection, name):
        assert self.write
        self.documents.pop((collection, name), None)
