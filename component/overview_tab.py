import streamlit as st


def get_langgraph_diagram() -> str:
    """LangGraph 워크플로우 마크다운 반환"""
    return """
| 순서 | 노드 | 역할 | 다음 단계 |
|:---:|------|------|----------|
| 1 | **START** | 그래프 진입점 | → summary_node |
| 2 | **summary_node** | Context Compression (3턴마다 대화 요약) | → llm_node |
| 3 | **llm_node** | LLM 추론 + `bind_tools()` | → tools_condition 분기 |
| 4a | **tool_node** | ToolNode 실행 (도구 호출 시) | → llm_node (재추론) |
| 4b | **END** | 그래프 종료 (도구 호출 없을 시) | - |

> **tools_condition**: llm_node의 응답에 tool_calls가 포함되면 → tool_node, 없으면 → END
"""


def get_architecture_diagram() -> str:
    """앱 전체 아키텍처 Mermaid 다이어그램 반환"""
    return """
graph TB
    subgraph UI["Streamlit UI"]
        overview["Overview Tab"]
        prompts["Prompts Tab"]
        chat["Chat Tab"]
        pdf["PDF Tab"]
        sidebar["Sidebar Settings"]
    end
    subgraph Service["Service Layer"]
        graph_builder["ReactGraphBuilder"]
        reasoning_detector["ReasoningDetector"]
        session_mgr["SessionManager"]
        llm_service["LLMService"]
        rag_service["RAGService"]
    end
    subgraph LangGraph["LangGraph ReAct Graph"]
        summary_node["summary_node"] --> llm_node["llm_node"]
        llm_node -->|tools_condition| tool_node["tool_node"]
        tool_node --> llm_node
        llm_node --> END_NODE["END"]
    end
    chat --> graph_builder
    sidebar --> graph_builder
    graph_builder --> LangGraph
    pdf --> rag_service
    chat --> reasoning_detector
"""


def get_concept_cards() -> list[dict]:
    """핵심 개념 카드 목록 반환"""
    return [
        {
            "title": "ReAct 패턴",
            "emoji": "🔄",
            "description": "LLM이 Reasoning + Acting을 반복하는 에이전트 패턴입니다.",
            "detail": "이 앱에서 `llm_node → tool_node → llm_node` 루프가 ReAct 패턴을 구현합니다. LLM이 도구 호출이 필요하다고 판단하면 `tool_node`로 라우팅되고, 도구 결과를 받아 다시 추론합니다. `tools_condition`이 이 분기를 제어합니다.",
        },
        {
            "title": "Tool Calling",
            "emoji": "🔧",
            "description": "`bind_tools()` + `ToolNode` + `tools_condition`으로 구현된 LangChain 표준 Tool Calling 패턴입니다.",
            "detail": "4개 도구가 LLM에 바인딩됩니다:\n- `get_current_time`: 현재 시각 (KST)\n- `switch_to_reasoning`: 추론 모드 전환 (gemini-2.5-pro)\n- `web_search`: Tavily 웹 검색\n- `search_pdf_knowledge`: PDF RAG 검색\n\nLLM은 자동으로 적합한 도구를 선택하여 호출합니다.",
        },
        {
            "title": "Context Compression",
            "emoji": "📋",
            "description": "`summary_node`에서 3턴마다 대화를 요약하여 컨텍스트를 압축합니다.",
            "detail": "normal 턴 4, 7, 10번째에서 직전 3개 normal 턴을 요약합니다. `compression_rate` 설정으로 요약 길이를 조절할 수 있습니다 (0.1~0.5). casual 턴은 요약 카운트에서 제외됩니다.",
        },
        {
            "title": "Streaming",
            "emoji": "⚡",
            "description": "`stream_mode=\"messages\"`를 사용한 실시간 토큰 스트리밍입니다.",
            "detail": "5종류의 스트리밍 이벤트:\n- `token`: 텍스트 토큰\n- `tool_call`: 도구 호출 시작\n- `tool_result`: 도구 실행 결과\n- `thought`: 사고 과정 (thinking mode)\n- `done`: 스트리밍 완료 + 메타데이터",
        },
        {
            "title": "Thinking Mode",
            "emoji": "🧠",
            "description": "`thinking_budget` 설정으로 모델의 사고 과정을 활성화합니다.",
            "detail": "지원 모델: gemini-2.5-pro, gemini-2.5-flash\n`include_thoughts=True` 시 사고 과정이 응답에 포함됩니다. UI에서 expander로 사고 과정을 확인할 수 있습니다.",
        },
        {
            "title": "Casual Detection",
            "emoji": "💬",
            "description": "`ReasoningDetector`가 패턴 매칭으로 casual/normal/reasoning 모드를 분류합니다.",
            "detail": "casual 모드는 그래프를 거치지 않고 직접 LLM 호출합니다 (casual_bypass). 인사, 감탄사, 짧은 입력 등이 casual로 분류됩니다. reasoning 모드는 복잡한 분석/비교/수학 등에 활성화됩니다.",
        },
        {
            "title": "Session & Checkpointing",
            "emoji": "💾",
            "description": "`SqliteSaver`로 그래프 상태를 자동 저장하고 세션을 관리합니다.",
            "detail": "LangGraph의 `SqliteSaver` checkpointer가 매 그래프 실행 시 상태를 자동 저장합니다. `thread_id` 기반으로 세션이 분리되며, 세션 전환 시 이전 대화를 복원할 수 있습니다.",
        },
    ]


def get_tool_info() -> list[dict]:
    """툴 정보 목록 반환"""
    return [
        {
            "name": "get_current_time",
            "description": "현재 시각 (KST)",
            "condition": "지금 몇 시, 오늘 날짜 등",
            "bind_method": "bind_tools()",
        },
        {
            "name": "switch_to_reasoning",
            "description": "추론 모드 전환 (gemini-2.5-pro)",
            "condition": "복잡한 분석, 비교, 수학 계산",
            "bind_method": "bind_tools()",
        },
        {
            "name": "web_search",
            "description": "Tavily 웹 검색",
            "condition": "최신 정보, 실시간 데이터 필요",
            "bind_method": "bind_tools()",
        },
        {
            "name": "search_pdf_knowledge",
            "description": "PDF RAG 검색",
            "condition": "업로드된 PDF 관련 질문",
            "bind_method": "bind_tools()",
        },
    ]


def get_tool_calling_markdown() -> str:
    """툴 콜링 구성 마크다운 반환"""
    tool_info = get_tool_info()
    lines = [
        "| 툴 | 설명 | 호출 조건 | 바인딩 |",
        "|-----|------|----------|--------|",
    ]
    for tool in tool_info:
        lines.append(
            f"| `{tool['name']}` | {tool['description']} | {tool['condition']} | `{tool['bind_method']}` |"
        )
    lines.append("")
    lines.append(
        "> LLM이 사용자 입력을 분석하여 적합한 도구를 **자동 선택**합니다. "
        "`llm_node`에서 `bind_tools()`로 바인딩된 도구 중 하나 이상이 호출되면 `tools_condition`이 `tool_node`로 라우팅합니다."
    )
    return "\n".join(lines)


def get_response_length_diagram() -> str:
    """응답 길이 규칙 Mermaid 다이어그램 반환"""
    return """
graph LR
    A{"사용된 툴"} -->|"추론/검색/RAG"| B["상세 답변"]
    A -->|"시각만 or 없음"| C["간결 답변"]
"""


def get_overview_content() -> dict:
    """Overview 탭에 표시할 콘텐츠 반환"""
    return {
        "introduction": """
## Gemini Hybrid Chatbot

이 앱은 **현대 AI 챗봇의 핵심 개념들**이 어떻게 구현되고 동작하는지 교육적으로 보여주는 데모입니다.

### 적용된 핵심 기술
- **ReAct 패턴**: LLM의 Reasoning + Acting 반복으로 복잡한 질문 처리
- **Tool Calling**: LangChain 표준 패턴으로 4개 도구 자동 선택 및 실행
- **Context Compression**: 3턴마다 대화 요약으로 장기 대화 지원
- **Streaming**: 실시간 토큰 스트리밍으로 응답 대기 시간 최소화
- **Thinking Mode**: 모델의 사고 과정 시각화
- **Casual Detection**: 입력 유형별 자동 모드 분류 (casual/normal/reasoning)
- **Session Checkpointing**: SqliteSaver 기반 자동 상태 저장
""",
        "quick_start": """
## 시작하기

### 1. API Key 설정
1. 사이드바의 **API Keys** 섹션을 엽니다
2. **Gemini API Key** 입력 (Google AI Studio에서 발급)
3. (선택) **Tavily API Key** 입력 (웹 검색 기능 사용 시)

### 2. 첫 대화 시작
1. Chat 탭으로 이동합니다
2. 하단의 입력창에 메시지를 입력합니다
3. Enter 키 또는 전송 버튼을 클릭합니다

### 3. PDF 문서 활용
1. PDF Preprocessing 탭으로 이동합니다
2. PDF 파일을 업로드합니다
3. "Process PDF" 버튼을 클릭하여 전처리를 시작합니다
4. 처리 완료 후 Chat 탭에서 PDF 관련 질문을 할 수 있습니다
""",
        "features": """
## 주요 기능

### Chat 기능
- **일반 대화**: 자연스러운 대화형 AI 응답
- **PDF 기반 Q&A**: 업로드된 PDF 문서에서 관련 정보를 검색하여 답변
- **웹 검색**: Tavily API를 통해 최신 정보를 검색하여 답변에 반영
- **자동 모델 전환**: 복잡한 질문 시 자동으로 추론 모델(Pro)로 전환

### PDF 전처리
- **텍스트 추출**: PDF에서 텍스트 추출
- **청킹**: 텍스트를 검색 가능한 청크로 분할
- **정규화**: LLM을 활용한 텍스트 정규화
- **임베딩**: 벡터 임베딩 생성 및 유사도 검색

### 세션 관리
- **새 세션 생성**: 새로운 대화 시작
- **세션 전환**: 이전 대화로 돌아가기
- **대화 다운로드**: CSV 형식으로 대화 내역 다운로드
""",
        "settings": """
## 설정 가이드

### Model Settings
| 설정 | 설명 | 범위 |
|------|------|------|
| **Chat Model** | 사용할 Gemini 모델 선택 | gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash |
| **Temperature** | 응답의 창의성/무작위성 조절 | 0.0 ~ 2.0 (기본: 0.7) |
| **Top-p** | 누적 확률 기반 토큰 선택 | 0.0 ~ 1.0 (기본: 0.9) |
| **Max Output Tokens** | 최대 출력 토큰 수 | 256 ~ 65,536 (기본: 8,192) |

### External Search
| 설정 | 설명 |
|------|------|
| **Enable Tavily Search** | 웹 검색 기능 활성화/비활성화 |
| **Search Depth** | 검색 깊이 (basic/advanced) |
| **Max Results** | 검색 결과 최대 개수 (1~10) |
""",
        "faq": """
## FAQ

### Q: API Key는 어디서 얻나요?
- **Gemini API Key**: [Google AI Studio](https://aistudio.google.com/)에서 발급
- **Tavily API Key**: [Tavily](https://tavily.com/)에서 발급

### Q: PDF 전처리는 왜 필요한가요?
PDF 전처리를 통해 문서의 내용을 벡터 임베딩으로 변환합니다. 이를 통해 질문과 관련된 문서 내용을 빠르게 검색할 수 있습니다.

### Q: 세션을 바꾸면 데이터가 사라지나요?
아니요. 각 세션의 대화 내역, 토큰 사용량, PDF 데이터 등은 모두 저장됩니다. 세션 전환 시 해당 세션의 데이터가 로드됩니다.

### Q: 토큰 제한은 어떻게 되나요?
환경 변수 `TOKEN_LIMIT_K`로 설정할 수 있습니다 (기본: 256K). 토큰 사용량이 80%를 초과하면 경고가 표시되며, 100% 초과 시 새 세션을 시작해야 합니다.

### Q: 어떤 모델을 선택해야 하나요?
- **gemini-2.5-flash**: 빠른 응답, 일반적인 대화에 적합 (권장)
- **gemini-2.5-pro**: 복잡한 추론, 분석 작업에 적합
- **gemini-2.0-flash**: 이전 버전 (2026년 3월 종료 예정)
""",
    }


def render_overview_tab() -> None:
    """Overview 탭 렌더링"""
    from streamlit_mermaid import st_mermaid

    content = get_overview_content()

    st.title("Gemini Hybrid Chatbot")
    st.caption("AI 챗봇 핵심 개념 교육 데모")

    # 1. 소개
    with st.expander("앱 소개", expanded=True):
        st.markdown(content["introduction"])

    # 2. 아키텍처 다이어그램
    with st.expander("🏗️ 앱 아키텍처", expanded=False):
        st.markdown("### 전체 구조")
        st.markdown("Streamlit UI → Service Layer → LangGraph ReAct Graph")
        st_mermaid(get_architecture_diagram())

    # 3. 핵심 개념 카드
    with st.expander("📚 핵심 개념", expanded=False):
        st.markdown("### AI 챗봇 핵심 기술")
        cards = get_concept_cards()
        for card in cards:
            with st.container(border=True):
                st.markdown(f"#### {card['emoji']} {card['title']}")
                st.markdown(card["description"])
                st.caption(card["detail"])

    # 4. LangGraph 워크플로우
    with st.expander("🔄 LangGraph 워크플로우", expanded=False):
        st.markdown("### ReAct 그래프 실행 흐름")
        st.markdown("사용자 입력 → summary_node → llm_node → (tool_node ↔ llm_node) → END")
        st.markdown(get_langgraph_diagram())

    # 5. 툴 콜링 구성
    with st.expander("🔧 툴 콜링 구성", expanded=False):
        st.markdown("### 사용 가능한 툴")
        st.markdown(get_tool_calling_markdown())

    # 6. 기존 섹션 유지
    with st.expander("시작하기 (Quick Start)", expanded=False):
        st.markdown(content["quick_start"])

    with st.expander("설정 가이드", expanded=False):
        st.markdown(content["settings"])

    with st.expander("FAQ", expanded=False):
        st.markdown(content["faq"])

    st.divider()
    st.caption("버전: 2.0.0 | Phase 04 | 마지막 업데이트: 2026-02-07")
