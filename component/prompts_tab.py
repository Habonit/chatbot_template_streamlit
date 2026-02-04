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
        # Phase 02-5: ReAct 그래프 노드별 프롬프트 (.py 기반)
        "tool_selector": {
            "title": "Tool Selector Prompt",
            "description": "ReAct 그래프에서 다음에 실행할 툴을 선택하는 프롬프트입니다.",
            "usage": "tool_selector 노드에서 사용됩니다.",
            "content": TOOL_SELECTOR_PROMPT,
        },
        "reasoning_prompt": {
            "title": "Reasoning Tool Prompt",
            "description": "단계별 추론을 수행하는 프롬프트입니다.",
            "usage": "reasoning_tool 노드에서 사용됩니다.",
            "content": REASONING_PROMPT,
        },
        "result_processor": {
            "title": "Result Processor Prompt",
            "description": "툴 실행 결과를 분석하고 추가 툴 필요 여부를 판단하는 프롬프트입니다.",
            "usage": "result_processor 노드에서 사용됩니다.",
            "content": RESULT_PROCESSOR_PROMPT,
        },
        "response_generator": {
            "title": "Response Generator Prompt",
            "description": "최종 응답을 생성하는 프롬프트입니다.",
            "usage": "response_generator 노드에서 사용됩니다.",
            "content": RESPONSE_GENERATOR_PROMPT,
        },
        "summary_prompt": {
            "title": "Summary Generator Prompt",
            "description": "대화 내용을 요약하기 위한 프롬프트입니다.",
            "usage": "summary_node에서 대화가 3턴 이상일 때 사용됩니다.",
            "content": SUMMARY_GENERATOR_PROMPT,
        },
        # 레거시 .txt 기반 프롬프트 (PDF 처리용)
        "normalization_prompt": {
            "title": "PDF Normalization Prompt",
            "description": "PDF에서 추출한 텍스트를 검색에 최적화된 형태로 정규화하는 프롬프트입니다.",
            "usage": "PDF 전처리의 '정규화' 단계에서 각 청크마다 사용됩니다.",
            "content": prompt_loader.load("pdf", "normalization.txt"),
        },
        "description_prompt": {
            "title": "PDF Description Prompt",
            "description": "PDF 문서의 간단한 설명을 생성하는 프롬프트입니다.",
            "usage": "PDF 전처리의 '정규화' 단계 완료 후 사용됩니다.",
            "content": prompt_loader.load("pdf", "description.txt"),
        },
    }


def render_prompts_tab() -> None:
    """프롬프트 탭 렌더링"""
    st.title("Prompts")
    st.caption("앱에서 사용되는 프롬프트 안내")

    st.info(
        "이 탭에서는 Gemini Hybrid Chatbot에서 사용되는 모든 프롬프트를 확인할 수 있습니다. "
        "프롬프트는 `prompt/` 디렉토리에서 관리됩니다."
    )

    prompt_info = get_prompt_info()

    for key, info in prompt_info.items():
        with st.expander(f"📝 {info['title']}", expanded=(key == "system_prompt")):
            st.markdown(f"**설명**: {info['description']}")
            st.markdown(f"**사용 시점**: {info['usage']}")

            st.divider()

            st.markdown("**프롬프트 내용:**")
            st.code(info["content"], language="text")

    st.divider()

    # 컨텍스트 빌드 설명
    with st.expander("📚 컨텍스트 빌드 구조", expanded=False):
        st.markdown("""
전체 프롬프트는 다음과 같은 구조로 구성됩니다:

```
[System Prompt]
기본 시스템 프롬프트 + (PDF Extension - PDF 업로드 시)

[누적 요약문]
이전 대화의 요약 (3턴 이상일 때)
---

[최근 대화 원문]
- User: ...
- Assistant: ...
---

[현재 사용자 입력]
사용자의 현재 질문/메시지
```

이 구조를 통해 긴 대화에서도 컨텍스트를 효율적으로 관리합니다.
        """)

    st.caption("프롬프트 파일 위치: `prompt/` 디렉토리")
