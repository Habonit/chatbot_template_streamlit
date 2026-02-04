import streamlit as st
from service.prompt_loader import PromptLoader


def get_prompt_info() -> dict:
    """프롬프트 정보를 반환"""
    prompt_loader = PromptLoader()

    return {
        "system_prompt": {
            "title": "System Prompt",
            "description": "AI의 동작 모드를 정의하고 툴 사용 지시사항을 포함하는 기본 프롬프트입니다.",
            "usage": "모든 대화에서 사용됩니다. PDF가 업로드된 경우 PDF 관련 지시사항이 추가됩니다.",
            "content": prompt_loader.load("system", "base.txt"),
        },
        "pdf_extension": {
            "title": "PDF Extension Prompt",
            "description": "PDF가 업로드된 경우 시스템 프롬프트에 추가되는 지시사항입니다.",
            "usage": "PDF가 업로드되고 처리된 세션에서만 사용됩니다.",
            "content": prompt_loader.load("system", "pdf_extension.txt"),
        },
        "summary_prompt": {
            "title": "Summary Prompt",
            "description": "대화 내용을 요약하기 위한 프롬프트입니다.",
            "usage": "대화 턴이 3턴을 초과할 때 자동으로 사용됩니다.",
            "content": prompt_loader.load("summary", "summary.txt"),
        },
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
