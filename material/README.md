# Gemini 챗봇 교육 노트북

Gemini API와 LangChain/LangGraph를 활용하여 대화형 AI 챗봇을 처음부터 설계하고 구현하는 과정을 다루는 교육 자료입니다. API 호출 기초부터 ReAct 에이전트, RAG, 평가까지 14개 노트북으로 구성되어 있습니다.

---

## 실행 환경

모든 노트북은 **Google Colab** 환경에서 실행하도록 설계되어 있습니다.

- **런타임**: Python 3 (기본 Colab 런타임)
- **GPU**: 불필요 (LLM 추론은 API 호출 방식)
- **API 키**: Google AI Studio에서 발급한 Gemini API 키가 필요합니다
  - Colab의 `userdata`(보안 비밀)에 `GOOGLE_API_KEY`로 등록하여 사용
- **LangSmith** (노트북 2, 14): LangSmith API 키를 추가로 등록하면 트레이싱 및 평가 기능을 사용할 수 있습니다

각 노트북의 첫 번째 셀에서 필요한 패키지를 `!pip install -q`로 자동 설치합니다.

---

## 사용 모델

| 모델 | 용도 |
|------|------|
| `gemini-2.5-flash` | 전 노트북 공통 — 기본 생성 모델 |
| `gemini-2.5-pro` | 노트북 10 — Thinking 비교 대상 |
| `text-embedding-004` | 노트북 12, 13 — 임베딩 모델 |

---

## 외부 라이브러리

### 핵심 패키지

| 패키지 | 설명 | 사용 노트북 |
|--------|------|------------|
| `google-genai` | Gemini 공식 SDK (1차 인터페이스) | 전체 (1-14) |
| `langchain-google-genai` | LangChain용 Gemini 래퍼 | 전체 (1-14) |
| `langchain` | LangChain 코어 (체인, 리트리버, 문서 로더) | 7, 13 |
| `langchain-text-splitters` | 문서 청크 분할 유틸리티 | 13 |
| `langgraph` | LangGraph 상태 그래프 프레임워크 | 3, 7, 9, 11 |
| `langsmith` | LLM 트레이싱 및 평가 플랫폼 | 2, 14 |
| `pydantic` | 데이터 검증 및 구조화 출력 스키마 정의 | 8, 9, 10, 14 |

### 벡터 스토어 및 데이터 처리

| 패키지 | 설명 | 사용 노트북 |
|--------|------|------------|
| `chromadb` | Chroma 벡터 스토어 (SQLite 기반 영속성) | 12, 13 |
| `faiss-cpu` | FAISS 벡터 스토어 (인메모리, 고속 검색) | 12, 13 |
| `numpy` | 유사도 계산, 벡터 연산 | 12, 13, 14 |
| `networkx` | GraphRAG용 그래프 구축 및 탐색 | 13 |
| `matplotlib` | 유사도 매트릭스 시각화, 그래프 시각화 | 12, 13, 14 |

### Colab 내장 (별도 설치 불필요)

`os`, `json`, `time`, `asyncio`, `sqlite3`, `csv`, `datetime`, `typing` 등 표준 라이브러리와 `google.colab.userdata`는 Colab 환경에 기본 포함되어 있습니다.

---

## 학습 경로

```
Phase 1 — 기초: LLM과 대화하는 법
  1. API 호출       →  2. 프롬프트 설계   →  3. 멀티턴 대화

Phase 2 — 제어: 모델 동작을 원하는 대로 다루기
  4. 토큰과 비용    →  5. 생성 파라미터   →  6. 스트리밍

Phase 3 — 실전 기법: 챗봇을 똑똑하게 만드는 기술
  7. 컨텍스트 관리  →  8. 구조화 출력     →  9. 도구 호출
  → 10. 추론 모델   → 11. ReAct 에이전트

Phase 4 — 지식 확장: 외부 문서를 활용하는 법
  12. 임베딩        → 13. RAG

Phase 5 — 품질 관리: 만든 챗봇을 평가하는 법
  14. 평가
```

---

## 노트북 목록

### Phase 1 — 기초

| # | 노트북 | 핵심 내용 |
|---|--------|----------|
| 1 | `note_01_api_call.ipynb` | google-genai SDK 직접 호출과 LangChain 래핑 호출을 비교하고 LCEL 체인 패턴을 처음 체험합니다 |
| 2 | `note_02_prompt_and_langsmith.ipynb` | 메시지 역할(system/user/assistant)의 차이를 이해하고 LangSmith로 프롬프트 동작을 추적합니다 |
| 3 | `note_03_multi_turn.ipynb` | LLM의 stateless 특성을 확인하고 대화 이력 관리 전략(InMemory, SQLite, LangGraph)을 비교합니다 |

### Phase 2 — 제어

| # | 노트북 | 핵심 내용 |
|---|--------|----------|
| 4 | `note_04_token_and_cost.ipynb` | 토큰의 실체를 이해하고 컨텍스트 윈도우, 비용 구조, Long Context vs RAG 트레이드오프를 분석합니다 |
| 5 | `note_05_generation_params.ipynb` | Temperature, Top-p, Top-k 등 생성 파라미터의 원리를 이해하고 용도별 최적 조합을 실험합니다 |
| 6 | `note_06_streaming.ipynb` | 스트리밍 응답의 구현 방식(google-genai, LangChain, LCEL)과 TTFT 측정 방법을 학습합니다 |

### Phase 3 — 실전 기법

| # | 노트북 | 핵심 내용 |
|---|--------|----------|
| 7 | `note_07_context_management.ipynb` | Sliding Window, 토큰 트리밍, 요약 압축, 하이브리드 전략을 비교하여 대화 컨텍스트를 관리합니다 |
| 8 | `note_08_structured_output.ipynb` | Pydantic 모델로 LLM 출력을 구조화하고 google-genai와 LangChain 양쪽의 접근법을 비교합니다 |
| 9 | `note_09_tool_calling.ipynb` | Tool Calling의 의도-실행-결과 루프를 이해하고 LangGraph ToolNode로 자동화하는 방법을 학습합니다 |
| 10 | `note_10_thinking.ipynb` | 추론 모델의 작동 원리, thinking_budget 제어, 토큰 과금 구조를 이해하고 비용-정확도 최적점을 찾습니다 |
| 11 | `note_11_react_agent.ipynb` | LangGraph StateGraph로 ReAct 에이전트를 설계하고 llm_node-tool_node 루프를 구현합니다 |

### Phase 4 — 지식 확장

| # | 노트북 | 핵심 내용 |
|---|--------|----------|
| 12 | `note_12_embedding.ipynb` | 텍스트 임베딩의 원리와 Cosine Similarity를 이해하고 Chroma/FAISS 벡터 스토어를 비교합니다 |
| 13 | `note_13_rag.ipynb` | RAG 파이프라인(Retrieve-Augment-Generate)을 구축하고 GraphRAG로 복잡한 질문에 대응합니다 |

### Phase 5 — 품질 관리

| # | 노트북 | 핵심 내용 |
|---|--------|----------|
| 14 | `note_14_evaluation.ipynb` | LLM-as-Judge, Pairwise 비교, RAG Triad 등 평가 기법을 학습하고 LangSmith로 자동화합니다 |

---

## 노트북 구조

모든 노트북은 동일한 3파트 구조를 따릅니다.

| 파트 | 내용 | 비고 |
|------|------|------|
| **Part 1 — 이론** | 개념 설명 + 최소 코드 예시 | 강사가 진행 |
| **Part 2 — 실습** | TODO 기반 코딩 실습 | 학습자가 직접 작성 |
| **Part 3 — 챌린지** | 심화 문제 (선택) | 자율 학습 |

각 노트북에 대응하는 `.md` 파일(`note_XX_제목.md`)은 Part 1과 Part 2의 핵심 개념을 종합한 레퍼런스 문서로, 노트북을 열지 않고도 해당 주제를 이해할 수 있도록 작성되어 있습니다.

---

## 파일 구조

```
material/
├── README.md                          ← 이 파일
├── _00_load_map.md                    ← 교육 로드맵 (전체 설계)
├── _01_style.md                       ← 노트북 작성 스타일 가이드
├── _02_markdown_style.md              ← 마크다운 문서 작성 스타일 가이드
├── note_01_api_call.ipynb             ← 노트북
├── note_01_api_call.md                ← 요약 문서
├── note_02_prompt_and_langsmith.ipynb
├── note_02_prompt_and_langsmith.md
├── ...
├── note_14_evaluation.ipynb
└── note_14_evaluation.md
```
