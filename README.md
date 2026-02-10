# Gemini Hybrid Chatbot

> **2026.02.23 ~ 02.25 고급 챗봇 개발 교육** 실습용 Streamlit 애플리케이션

Google Gemini API와 LangChain/LangGraph를 활용한 교육용 AI 챗봇입니다. API 호출 기초부터 ReAct 에이전트, RAG, 스트리밍, 추론(Thinking)까지 — 교육 노트북에서 학습한 개념이 실제 애플리케이션에서 어떻게 통합되는지 직접 확인할 수 있습니다.

**배포 URL**: https://habonit-chat-template.streamlit.app/

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **ReAct 에이전트** | LangGraph StateGraph 기반 `llm_node ↔ tool_node` 루프로 도구 호출을 자동 처리 |
| **라우터 노드** | 사용자 입력을 분석하여 casual/normal 모드로 자동 분기 |
| **컨텍스트 요약** | 대화가 길어지면 요약 노드가 자동 압축하여 토큰 효율 관리 |
| **스트리밍 응답** | 실시간 토큰 스트리밍으로 TTFT(Time To First Token) 최소화 |
| **추론 도구 Thinking** | Gemini thinking_budget을 활용한 별도 추론 LLM으로 깊은 사고 과정 지원 |
| **PDF RAG** | PDF 업로드 → 텍스트 추출 → 청킹 → 임베딩 → 벡터 검색 파이프라인 |
| **웹 검색** | Tavily API 연동으로 실시간 웹 검색 도구 제공 |
| **교육 UI** | 각 턴마다 그래프 실행 경로, 모드, 프롬프트, 토큰 사용량 등 메타데이터 표시 |
| **LangSmith 평가** | 자동화된 LLM 평가 파이프라인 (evaluator 5종) |

---

## 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                       │
│  ┌──────────┬──────────┬──────────┬───────────────┐  │
│  │ Overview │ Prompts  │   Chat   │ PDF Preprocess│  │
│  └──────────┴──────────┴──────────┴───────────────┘  │
├─────────────────────────────────────────────────────┤
│                     app.py                           │
│          (세션 관리 · 핸들러 · 팩토리)                  │
├─────────────────────────────────────────────────────┤
│              LangGraph ReAct Graph                   │
│  START → summary_node → router_node                  │
│            ├─ casual → casual_node → END             │
│            └─ normal → llm_node ↔ tool_node → END    │
├─────────────────────────────────────────────────────┤
│  Service Layer                                       │
│  react_graph · tools · search · embedding · rag      │
├─────────────────────────────────────────────────────┤
│  Repository Layer                                    │
│  embedding_repo · pdf_extractor · SQLite (LangGraph) │
└─────────────────────────────────────────────────────┘
```

---

## 프로젝트 구조

```
.
├── app.py                    # 메인 오케스트레이터
├── component/                # Streamlit UI 컴포넌트
│   ├── chat_tab.py           #   채팅 탭 (스트리밍 · 메타데이터 표시)
│   ├── overview_tab.py       #   아키텍처 · 개념 카드 교육 탭
│   ├── prompts_tab.py        #   프롬프트 흐름 교육 탭
│   ├── pdf_tab.py            #   PDF 전처리 탭
│   ├── sidebar.py            #   사이드바 설정
│   ├── education_tips.py     #   컨텍스트 교육 팁 생성
│   └── styles.py             #   커스텀 CSS
├── service/                  # 비즈니스 로직
│   ├── react_graph.py        #   LangGraph ReAct 그래프 빌더
│   ├── tools.py              #   도구 정의 (웹 검색 · RAG · 추론)
│   ├── reasoning_detector.py #   casual/normal 모드 감지
│   ├── search_service.py     #   Tavily 웹 검색
│   ├── embedding_service.py  #   Gemini 임베딩
│   ├── rag_service.py        #   RAG 청킹 · 검색
│   ├── llm_service.py        #   LLM 직접 호출
│   ├── prompt_loader.py      #   프롬프트 템플릿 로더
│   └── session_manager.py    #   세션 관리 (SQLite)
├── domain/                   # 도메인 모델
│   ├── message.py            #   Message 데이터클래스
│   ├── session.py            #   Session 모델
│   └── chunk.py              #   PDF 청크 모델
├── repository/               # 데이터 접근
│   ├── embedding_repo.py     #   임베딩 저장/로드
│   └── pdf_extractor.py      #   PDF 텍스트 추출
├── prompt/                   # 프롬프트 템플릿
│   ├── pdf/                  #   PDF 관련 프롬프트
│   ├── summary/              #   요약 프롬프트
│   ├── selector/             #   도구 선택 프롬프트
│   ├── response/             #   응답 생성 프롬프트
│   ├── processor/            #   결과 처리 프롬프트
│   └── tools/                #   추론 도구 프롬프트
├── evaluation/               # LangSmith 평가
│   ├── evaluators.py         #   평가 함수 5종
│   ├── datasets.py           #   테스트 데이터셋 생성
│   ├── runner.py             #   평가 실행기
│   └── run_evaluation.py     #   CLI 진입점
├── tests/                    # 테스트 (585+)
├── .streamlit/config.toml    # Streamlit 테마 설정
├── pyproject.toml            # 프로젝트 설정
└── material/                 # 교육 노트북 (14개)
```

---

## 교육 커리큘럼과의 연결

이 앱은 `material/` 디렉토리의 14개 Colab 노트북에서 학습한 개념을 통합 구현한 결과물입니다.

| Phase | 노트북 | 앱에서의 구현 |
|-------|--------|--------------|
| **1. 기초** | API 호출, 프롬프트 설계, 멀티턴 | `llm_service.py`, 프롬프트 템플릿, SQLite 대화 저장 |
| **2. 제어** | 토큰/비용, 생성 파라미터, 스트리밍 | 토큰 사용량 추적, 사이드바 파라미터, `stream()` 메서드 |
| **3. 실전** | 컨텍스트 관리, 구조화 출력, 도구 호출, Thinking, ReAct | `summary_node`, Pydantic 스키마, `tool_node`, `thinking_budget`, StateGraph |
| **4. 지식 확장** | 임베딩, RAG | `embedding_service.py`, PDF 전처리 파이프라인, 벡터 검색 |
| **5. 품질 관리** | 평가 | `evaluation/` 패키지, LangSmith 연동 |

---

## 설치 및 실행

### 사전 요구사항

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (권장) 또는 pip

### 설치

```bash
# 저장소 클론
git clone <repository-url>
cd <프로젝트 이름>

# uv로 설치
uv sync

```

### 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 API 키를 입력합니다:

```env
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key          # 웹 검색 (선택)
LANGSMITH_API_KEY=your_langsmith_api_key    # 트레이싱 (선택)
TOKEN_LIMIT_K=256                           # 토큰 상한 (천 단위)
```

### 실행

```bash
uv run streamlit run app.py
```

### 테스트

```bash
uv run pytest
```

---

## 사용 모델

| 모델 | 용도 |
|------|------|
| `gemini-2.0-flash` | 기본 대화 모델 (사이드바에서 변경 가능) |
| `gemini-2.5-flash` / `gemini-2.5-pro` | Thinking 지원 모델 |
| `gemini-embedding-001` | PDF 임베딩 |

---

## 기술 스택

| 카테고리 | 기술 |
|----------|------|
| **UI** | Streamlit, streamlit-mermaid |
| **LLM** | Google Gemini API (langchain-google-genai) |
| **에이전트** | LangGraph (StateGraph, ToolNode) |
| **벡터 검색** | FAISS (faiss-cpu) |
| **웹 검색** | Tavily API |
| **PDF 처리** | pdfplumber |
| **데이터 저장** | SQLite (langgraph-checkpoint-sqlite) |
| **평가** | LangSmith |
| **테스트** | pytest (585+ 테스트) |

---

## 라이선스

교육 목적으로 제작되었습니다.
