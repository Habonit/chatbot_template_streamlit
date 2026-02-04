import streamlit as st


def get_overview_content() -> dict:
    """Overview 탭에 표시할 콘텐츠 반환"""
    return {
        "introduction": """
## Gemini Hybrid Chatbot

Gemini Hybrid Chatbot은 Google의 Gemini API를 활용한 하이브리드 AI 챗봇입니다.

### 주요 특징
- **하이브리드 AI**: 일반 대화는 빠른 Flash 모델, 복잡한 추론은 Pro 모델을 자동으로 선택
- **RAG (Retrieval-Augmented Generation)**: PDF 문서를 업로드하여 문서 기반 질의응답 가능
- **웹 검색 통합**: Tavily API를 통한 실시간 웹 검색 지원
- **세션 관리**: 대화 히스토리를 세션별로 분리하여 관리
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
    content = get_overview_content()

    st.title("Gemini Hybrid Chatbot")
    st.caption("하이브리드 AI 챗봇 사용 가이드")

    with st.expander("앱 소개", expanded=True):
        st.markdown(content["introduction"])

    with st.expander("시작하기 (Quick Start)", expanded=False):
        st.markdown(content["quick_start"])

    with st.expander("주요 기능", expanded=False):
        st.markdown(content["features"])

    with st.expander("설정 가이드", expanded=False):
        st.markdown(content["settings"])

    with st.expander("FAQ", expanded=False):
        st.markdown(content["faq"])

    st.divider()
    st.caption("버전: 1.0.0 | 마지막 업데이트: 2026-02-04")
