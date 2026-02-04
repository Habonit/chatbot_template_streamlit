from tavily import TavilyClient


class SearchService:
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)

    def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
    ) -> list[dict]:
        try:
            response = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
            )
            return response.get("results", [])
        except Exception:
            return []

    def format_for_llm(self, results: list[dict]) -> str:
        if not results:
            return ""

        lines = ["[웹 검색 결과]"]
        for i, result in enumerate(results, 1):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")
            lines.append(f"{i}. {title} - {url}")
            lines.append(f"   {content[:200]}...")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        try:
            client = TavilyClient(api_key=api_key)
            client.search("test", max_results=1)
            return True
        except Exception:
            return False
