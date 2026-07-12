from typing import Any, Literal

class TavilyClient:
    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None: ...
    def search(
        self,
        query: str,
        *,
        search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = ...,
        topic: Literal["general", "news", "finance"] = ...,
        days: int | None = ...,
        max_results: int = ...,
        include_answer: bool = ...,
        include_raw_content: bool = ...,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
