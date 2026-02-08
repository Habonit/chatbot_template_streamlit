import streamlit as st
from service.prompt_loader import PromptLoader

# Phase 02-5: .py 기반 프롬프트 임포트
from prompt.selector.tool_selector import TOOL_SELECTOR_PROMPT
from prompt.tools.reasoning import REASONING_PROMPT
from prompt.processor.result_processor import RESULT_PROCESSOR_PROMPT
from prompt.response.response_generator import RESPONSE_GENERATOR_PROMPT
from prompt.summary.summary_generator import SUMMARY_GENERATOR_PROMPT


def get_prompt_info() -> dict:
    """프롬프트 정보를 반환"""
    prompt_loader = PromptLoader()

    return {
        "tool_selector": {
            "title": "Tool Selector Prompt",
            "description": "시스템 프롬프트에 내장되어 LLM이 적절한 도구를 선택하도록 안내하는 프롬프트입니다.",
            "usage": "_build_system_prompt()에서 시스템 프롬프트에 내장됩니다.",
            "graph_node": "llm_node (시스템 프롬프트)",
            "status": "active",
            "content": TOOL_SELECTOR_PROMPT,
        },
        "reasoning_prompt": {
            "title": "Reasoning Tool Prompt",
            "description": "switch_to_reasoning 도구 내부에서 단계별 추론을 수행하는 프롬프트입니다.",
            "usage": "reasoning tool 실행 시 사용됩니다.",
            "graph_node": "tool_node (reasoning tool)",
            "status": "active",
            "content": REASONING_PROMPT,
        },
        "result_processor": {
            "title": "Result Processor Prompt",
            "description": "Phase 02 레거시 프롬프트입니다. Phase 03에서 LangChain tools_condition으로 대체되었습니다.",
            "usage": "현재 미사용 (레거시)",
            "graph_node": "N/A (레거시)",
            "status": "legacy",
            "content": RESULT_PROCESSOR_PROMPT,
        },
        "response_generator": {
            "title": "Response Generator Prompt",
            "description": "Phase 02 레거시 프롬프트입니다. Phase 03에서 LLM이 직접 최종 응답을 생성합니다.",
            "usage": "현재 미사용 (레거시)",
            "graph_node": "N/A (레거시)",
            "status": "legacy",
            "content": RESPONSE_GENERATOR_PROMPT,
        },
        "summary_prompt": {
            "title": "Summary Generator Prompt",
            "description": "대화 내용을 요약하기 위한 프롬프트입니다.",
            "usage": "summary_node에서 normal 턴 4, 7, 10번째에 사용됩니다.",
            "graph_node": "summary_node",
            "status": "active",
            "content": SUMMARY_GENERATOR_PROMPT,
        },
        "normalization_prompt": {
            "title": "PDF Normalization Prompt",
            "description": "PDF에서 추출한 텍스트를 검색에 최적화된 형태로 정규화하는 프롬프트입니다.",
            "usage": "PDF 전처리의 '정규화' 단계에서 각 청크마다 사용됩니다.",
            "graph_node": "N/A (PDF 전처리)",
            "status": "active",
            "content": prompt_loader.load("pdf", "normalization.txt"),
        },
        "description_prompt": {
            "title": "PDF Description Prompt",
            "description": "PDF 문서의 간단한 설명을 생성하는 프롬프트입니다.",
            "usage": "PDF 전처리의 '정규화' 단계 완료 후 사용됩니다.",
            "graph_node": "N/A (PDF 전처리)",
            "status": "active",
            "content": prompt_loader.load("pdf", "description.txt"),
        },
    }


def get_system_prompt_builder_info() -> dict:
    """_build_system_prompt() 로직 설명 반환"""
    return {
        "title": "System Prompt Builder",
        "description": "ReactGraphBuilder._build_system_prompt()가 실행 시점에 동적으로 시스템 프롬프트를 구성합니다.",
        "components": [
            {
                "name": "기본 지침",
                "content": "당신은 유용한 AI 어시스턴트입니다. 필요한 경우 도구를 사용하여 정확한 정보를 제공하세요.",
                "always_included": True,
            },
            {
                "name": "이전 대화 요약 (summary_history)",
                "content": "[이전 대화 요약] 섹션으로 포함",
                "always_included": False,
            },
            {
                "name": "PDF 설명 (pdf_description)",
                "content": "[업로드된 PDF] 섹션으로 포함",
                "always_included": False,
            },
        ],
        "flow": "[기본 지침] + [이전 대화 요약 (있을 경우)] + [PDF 설명 (있을 경우)]",
    }


def get_prompt_flow_diagram() -> str:
    """프롬프트 사용 흐름 Mermaid 다이어그램 반환"""
    return """
graph LR
    input["사용자 입력"] --> detector["ReasoningDetector"]
    detector -->|casual| casual_prompt["Casual Prompt<br/>(직접 LLM 호출)"]
    detector -->|normal/reasoning| system["System Prompt Builder"]
    system --> llm["llm_node<br/>LLM + bind_tools"]
    llm -->|tool_call| tools["Tool Prompts"]
    tools --> reasoning_prompt["reasoning tool"]
    summary_node["summary_node"] --> summary_prompt["Summary Prompt"]
    summary_node --> system
"""


def render_prompts_tab() -> None:
    """프롬프트 탭 렌더링"""
    st.title("Prompts")
    st.caption("프롬프트가 그래프 어디서 어떻게 사용되는지 교육적으로 보여줍니다")

    # 1. 프롬프트 흐름도
    from streamlit_mermaid import st_mermaid

    with st.expander("🗺️ 프롬프트 흐름도", expanded=True):
        st.markdown("### 프롬프트 사용 흐름")
        st.markdown("사용자 입력이 어떤 프롬프트를 거쳐 처리되는지 보여줍니다.")
        st_mermaid(get_prompt_flow_diagram())

    # 2. System Prompt Builder 설명
    with st.expander("🏗️ System Prompt Builder", expanded=False):
        builder_info = get_system_prompt_builder_info()
        st.markdown(f"### {builder_info['title']}")
        st.markdown(builder_info["description"])

        st.markdown(f"**구성:** `{builder_info['flow']}`")

        for comp in builder_info["components"]:
            badge = "✅ 항상 포함" if comp["always_included"] else "⚡ 조건부 포함"
            with st.container(border=True):
                st.markdown(f"**{comp['name']}** — {badge}")
                st.caption(comp["content"])

    # 3. 프롬프트 목록 (active / legacy 분리)
    prompt_info = get_prompt_info()

    active_prompts = {k: v for k, v in prompt_info.items() if v.get("status") == "active"}
    legacy_prompts = {k: v for k, v in prompt_info.items() if v.get("status") == "legacy"}

    st.markdown("### 📝 활성 프롬프트")
    for key, info in active_prompts.items():
        with st.expander(f"📝 {info['title']} — `{info['graph_node']}`", expanded=False):
            st.markdown(f"**설명**: {info['description']}")
            st.markdown(f"**사용 위치**: {info['usage']}")
            st.markdown(f"**그래프 노드**: `{info['graph_node']}`")
            st.divider()
            st.markdown("**프롬프트 내용:**")
            st.code(info["content"], language="text")

    if legacy_prompts:
        st.markdown("### 🗄️ 레거시 프롬프트")
        st.caption("Phase 02에서 사용되었으나 Phase 03에서 대체된 프롬프트입니다.")
        for key, info in legacy_prompts.items():
            with st.expander(f"🗄️ {info['title']} — `{info['graph_node']}`", expanded=False):
                st.markdown(f"**설명**: {info['description']}")
                st.markdown(f"**사용 위치**: {info['usage']}")
                st.divider()
                st.markdown("**프롬프트 내용:**")
                st.code(info["content"], language="text")

    st.divider()

    # 4. 컨텍스트 빌드 구조 (기존 유지 + 업데이트)
    with st.expander("📚 컨텍스트 빌드 구조", expanded=False):
        st.markdown("""
Phase 03-3 Tool Calling 패턴에서 컨텍스트는 다음과 같이 구성됩니다:

```
[System Prompt]
_build_system_prompt():
  - 기본 지침
  - + 이전 대화 요약 (summary_history)
  - + PDF 설명 (pdf_description, 있을 경우)

[이전 Raw 턴 메시지]
요약되지 않은 이전 완료 턴들

[현재 턴 메시지]
현재 진행 중인 턴 (HumanMessage + tool call/result 포함)
```

이 구조로 장기 대화에서도 컨텍스트 윈도우를 효율적으로 관리합니다.
        """)

    st.caption("프롬프트 파일 위치: `prompt/` 디렉토리")
