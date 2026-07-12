from typing import Any

class AsyncResult:
    async def data(self) -> list[dict[str, Any]]: ...

class AsyncSession:
    async def __aenter__(self) -> AsyncSession: ...
    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None: ...
    async def run(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncResult: ...

class AsyncDriver:
    async def verify_connectivity(self, **kwargs: Any) -> None: ...
    async def close(self) -> None: ...
    def session(
        self,
        *,
        database: str | None = None,
        **kwargs: Any,
    ) -> AsyncSession: ...

class AsyncGraphDatabase:
    @classmethod
    def driver(
        cls,
        uri: str,
        *,
        auth: tuple[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncDriver: ...
