"""Phase 03-3: LangChain @tool 데코레이터 기반 도구 정의

LangChain 표준 Tool Calling 패턴 사용:
- @tool 데코레이터로 도구 정의
- bind_tools()로 LLM에 바인딩
- ToolNode로 자동 실행
"""
from datetime import datetime
from typing import Any

import pytz
from langchain_core.tools import tool


def create_tools_with_services(
    search_service: Any = None,
    embedding_service: Any = None,
    embedding_repo: Any = None,
    session_id: str = "",
    llm: Any = None,
) -> list:
    """서비스가 주입된 도구 목록 생성

    Args:
        search_service: Tavily 검색 서비스
        embedding_service: 임베딩 서비스
        embedding_repo: 임베딩 저장소
        session_id: 세션 ID
        llm: LLM 인스턴스 (reasoning용)

    Returns:
        list: 서비스가 주입된 도구 목록
    """

    @tool
    def get_current_time() -> str:
        """현재 시각을 반환합니다. 시간, 날짜, 요일 관련 질문에 사용하세요."""
        kst = pytz.timezone("Asia/Seoul")
        now = datetime.now(kst)
        return now.strftime("%Y-%m-%d %H:%M:%S") + " (KST)"

    @tool
    def web_search(query: str) -> str:
        """웹에서 최신 정보를 검색합니다. 실시간 뉴스, 최신 동향, 현재 상황에 대한 질문에 사용하세요.

        Args:
            query: 검색할 쿼리
        """
        if search_service:
            results = search_service.search(query)
            return search_service.format_for_llm(results)
        return "웹 검색 서비스가 설정되지 않았습니다."

    @tool
    def search_pdf_knowledge(query: str) -> str:
        """업로드된 PDF 문서에서 관련 내용을 검색합니다. 문서 기반 질문에 사용하세요.

        Args:
            query: PDF에서 검색할 내용
        """
        if embedding_service and embedding_repo:
            query_embedding = embedding_service.embed_query(query)
            results = embedding_repo.search_similar(session_id, query_embedding, top_k=5)

            if results:
                formatted = []
                for i, r in enumerate(results, 1):
                    formatted.append(
                        f"[{i}] (유사도: {r['score']:.2f})\n{r['chunk'].normalized_text}"
                    )
                return "\n\n".join(formatted)
            return "PDF에서 관련 내용을 찾을 수 없습니다."
        return "PDF 검색 서비스가 설정되지 않았습니다."

    @tool
    def reasoning(question: str, context: str = "") -> str:
        """복잡한 문제를 단계별로 분석합니다. 비교, 분석, 추론이 필요한 질문에 사용하세요.

        Args:
            question: 분석할 질문
            context: 참고할 맥락 정보 (선택)
        """
        if llm:
            from langchain_core.messages import HumanMessage
            from prompt.tools.reasoning import get_prompt as get_reasoning_prompt

            prompt = get_reasoning_prompt(user_input=question, context=context)
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        return "추론 도구가 설정되지 않았습니다."

    return [
        get_current_time,
        web_search,
        search_pdf_knowledge,
        reasoning,
    ]
