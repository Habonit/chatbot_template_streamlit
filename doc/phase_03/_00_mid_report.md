# 중간 점검 리포트: 적용 현황 분석

> load_map.md의 개념들을 현재 앱 구현 상태와 대조 분석

---

## 요약

| 구분 | 개수 |
|------|------|
| ✅ 적용됨 | 17개 |
| ❌ 미적용 | 7개 |

---

## ✅ 적용된 개념들

### 노트북 1: Gemini 직접 호출 vs LangChain 호출

| 개념 | 적용 위치 | 설명 |
|------|-----------|------|
| google-genai SDK 직접 호출 | `service/llm_service.py` | `google.generativeai` 사용 |
| LangChain 래핑 | `service/react_graph.py` | `ChatGoogleGenerativeAI` 사용 |

### 노트북 2: System Prompt

| 개념 | 적용 위치 | 설명 |
|------|-----------|------|
| 메시지 역할 체계 | `service/react_graph.py` | SystemMessage, HumanMessage, AIMessage 구분 |
| System Prompt 설계 | `prompt/` 폴더 | 역할별 프롬프트 템플릿 분리 |

### 노트북 3: Multi-turn + 대화 저장

| 개념 | 적용 위치 | 설명 |
|------|-----------|------|
| Multi-turn 대화 | `service/react_graph.py` | MessagesState로 대화 이력 관리 |
| 대화 저장소 (SQLite) | `service/session_manager.py` | SqliteSaver로 체크포인팅 |

### 노트북 4: 토큰

| 개념 | 적용 위치 | 설명 |
|------|-----------|------|
| 토큰 추적 | `service/llm_service.py` | input_tokens, output_tokens 추적 및 UI 표시 |

### 노트북 7: 컨텍스트 매니지먼트

| 개념 | 적용 위치 | 설명 |
|------|-----------|------|
| 요약 기반 압축 | `service/react_graph.py` | Turn 4, 7, 10... 마다 이전 3턴 자동 요약 |
| 하이브리드 (요약 + Window) | `service/react_graph.py` | 요약본 + 최근 3턴 원문 조합 |

### 노트북 9: LangGraph

| 개념 | 적용 위치 | 설명 |
|------|-----------|------|
| LangGraph StateGraph | `service/react_graph.py` | 노드 + 조건부 엣지로 워크플로우 구성 |
| 도구 분기 (조건부 엣지) | `service/react_graph.py` | tool_selector → 4개 도구 분기 |

### 노트북 11: Embedding

| 개념 | 적용 위치 | 설명 |
|------|-----------|------|
| Gemini Embedding API | `service/embedding_service.py` | gemini-embedding-001 (768차원) |
| 벡터 스토어 (FAISS) | `repository/embedding_repo.py` | IndexFlatIP + pickle 저장 |
| Cosine Similarity | `repository/embedding_repo.py` | 정규화 후 Inner Product |

### 노트북 12: RAG

| 개념 | 적용 위치 | 설명 |
|------|-----------|------|
| 문서 로딩 (PDF) | `repository/pdf_extractor.py` | pdfplumber 사용 |
| 청크 분할 | `service/rag_service.py` | RecursiveCharacterTextSplitter 방식 (1024자, 256 overlap) |
| RAG 체인 | `service/react_graph.py` | search_pdf_knowledge 도구에서 검색 → 프롬프트 주입 |

### 노트북 15: 종합 실습

| 개념 | 적용 위치 | 설명 |
|------|-----------|------|
| Streamlit UI | `app.py`, `component/` | 4개 탭 구성 |

---

## ❌ 미적용 개념들 (향후 적용 대상)

### 노트북 2: LangSmith

| 개념 | 설명 |
|------|------|
| LangSmith 트레이싱 | 모든 LLM 호출 자동 로깅, 프롬프트 전문 확인, 토큰/비용 추적 |
| LangSmith Hub | 프롬프트 버전 관리 및 팀 공유 |

> LangSmith를 통한 관제 시스템 구축 필요

### 노트북 5: 생성 파라미터

| 개념 | 현재 상태 | 필요 작업 |
|------|-----------|-----------|
| temperature | ✅ 사이드바에 있음 | - |
| top_p | ✅ 사이드바에 있음 | - |
| max_output_tokens | ✅ 사이드바에 있음 | - |
| seed | ❌ 없음 | 사이드바에 추가 필요 |

> seed 파라미터를 사이드바에 추가하여 재현성 제어 가능하도록 해야 함

### 노트북 6: Streaming

| 개념 | 설명 |
|------|------|
| 스트리밍 응답 | `.stream()` / `.astream()` 으로 토큰 단위 실시간 출력 |

> 현재 `st.spinner("Thinking...")` 후 응답 전체를 한 번에 표시하는 방식. 토큰 단위 스트리밍 출력으로 변경 필요

### 노트북 8: Structured Output

| 개념 | 설명 |
|------|------|
| Pydantic 구조화 출력 | `model.with_structured_output(Pydantic)` 패턴 |

> 현재 JSON 문자열 파싱 방식으로 불안정. Pydantic 모델 바인딩으로 안정적인 구조화 출력 구현 필요

### 노트북 9: Tool Calling

| 개념 | 설명 |
|------|------|
| @tool 데코레이터 | LangChain 표준 도구 정의 방식 |
| bind_tools() | 모델에 도구 스키마 바인딩 |
| ToolNode | 도구 실행 자동화 노드 |

> 현재 수동 도구 선택 (LLM이 도구명 문자열 출력 → 조건 분기) 방식. `@tool` 데코레이터 + `bind_tools()` + `ToolNode`로 자동화 필요, ReAct 구조와 연결하는 것이 필요

### 노트북 10: 추론 모델

| 개념 | 설명 |
|------|------|
| thinking_budget 제어 | Gemini 2.5의 추론 토큰 예산 설정 |
| include_thoughts=True | 모델의 추론 과정 확인 옵션 |

> 현재 reasoning_mode 토글은 있으나 모델 레벨의 추론 옵션 미적용. `include_thoughts=True` 등을 사용해서 모델에게도 추론 모드를 전달해야 함, 물론 현재의 프롬프트 방식도 적용하면서. 

### 노트북 14: 평가 시스템

| 개념 | 설명 |
|------|------|
| LangSmith Evaluation | Dataset + Evaluator 자동 평가 |

> LangSmith 기반 자동 평가 시스템 구축 필요

---

## 참고: 노트북별 매핑 표

| 노트북 | 주제 | 상태 |
|--------|------|------|
| 1 | Gemini 직접 vs LangChain | ✅ 적용 |
| 2 | System Prompt + LangSmith | ⚠️ LangSmith 미적용 |
| 3 | Multi-turn + 저장 | ✅ 적용 |
| 4 | 토큰/컨텍스트 윈도우 | ✅ 적용 |
| 5 | 생성 파라미터 | ❌ seed 미적용 |
| 6 | Streaming | ❌ 미적용 |
| 7 | 컨텍스트 매니지먼트 | ✅ 적용 |
| 8 | Structured Output | ❌ 미적용 |
| 9 | Tool Calling + LangGraph | ⚠️ LangGraph만 적용 |
| 10 | 추론 모델 | ❌ 미적용 |
| 11 | Embedding | ✅ 적용 |
| 12 | RAG | ✅ 적용 |
| 13 | Guardrails | - (적용 안함) |
| 14 | 평가 | ❌ 미적용 |
| 15 | 종합 실습 | ✅ 대부분 적용 |
