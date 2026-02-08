# Phase 03: 고급 기능 적용

> load_map.md 기반 미적용 개념들을 단계별로 구현

---

## 개요

| 구분 | 내용 |
|------|------|
| 기반 문서 | `_00_mid_report.md` |
| 총 단계 | 6단계 |
| 예상 범위 | 서비스 레이어 + UI 컴포넌트 + 평가 시스템 |

---

## 단계별 요약

| 단계 | 파일 | 주제 | 핵심 변경 |
|------|------|------|----------|
| 1 | `_01.md` | **기반 설정** | LangSmith 트레이싱 + seed 파라미터 |
| 2 | `_02.md` | **Structured Output** | Pydantic `with_structured_output()` |
| 3 | `_03.md` | **Tool Calling** | `@tool` + `bind_tools()` + `ToolNode` |
| 4 | `_04.md` | **Streaming** | 토큰 단위 스트리밍 응답 |
| 5 | `_05.md` | **추론 모델** | `thinking_budget` + `include_thoughts` |
| 6 | `_06.md` | **평가 시스템** | LangSmith Evaluation |

---

## 의존성 순서

```
Phase 03-1 (LangSmith + seed)
    ↓
Phase 03-2 (Structured Output)  ← Phase 03-3 선행 조건
    ↓
Phase 03-3 (Tool Calling)
    ↓
Phase 03-4 (Streaming)  ← 03-3 이후 권장
    ↓
Phase 03-5 (추론 모델)  ← 03-4 이후 권장 (Streaming + Thinking 통합)
    ↓
Phase 03-6 (평가 시스템)  ← 03-1 LangSmith 필요, invoke() 기반 독립 실행
```

---

## 파일 변경 요약

### 신규 생성

| 파일 | 단계 | 설명 |
|------|------|------|
| `domain/llm_output.py` | 2 | Pydantic 출력 모델 |
| `service/tools.py` | 3 | @tool 기반 도구 정의 |
| `evaluation/` | 6 | 평가 시스템 패키지 |

### 주요 수정

| 파일 | 단계 | 변경 내용 |
|------|------|----------|
| `.env` | 1 | LangSmith 환경 변수 추가 |
| `component/sidebar.py` | 1,5 | seed, thinking_budget 추가 |
| `service/react_graph.py` | 2,3,4,5 | 대폭 리팩토링 |
| `component/chat_tab.py` | 4 | 스트리밍 UI |
| `app.py` | 1,4 | 파라미터 전달, 스트리밍 핸들러 |

### 삭제

| 파일/코드 | 단계 | 이유 |
|-----------|------|------|
| `prompt/selector/tool_selector.py` | 3 | bind_tools로 대체 |
| JSON 파싱 로직 (~15줄) | 2 | Pydantic으로 대체 |
| 수동 도구 노드들 (~150줄) | 3 | ToolNode로 대체 |

---

## 기술 스택 변경

### 추가되는 의존성

**없음** — `langchain-google-genai` 4.2.0이 `thinking_budget` 네이티브 지원.

### 사용되는 새 패턴

| 패턴 | 단계 | 설명 |
|------|------|------|
| `model.with_structured_output(Pydantic)` | 2 | 구조화 출력 |
| `model.bind_tools([...])` | 3 | 도구 바인딩 |
| `ToolNode(tools)` | 3 | 자동 도구 실행 |
| `tools_condition` | 3 | 조건부 엣지 |
| `graph.stream(stream_mode="messages")` | 4 | 동기 토큰 스트리밍 (Streamlit 호환) |
| `ChatGoogleGenerativeAI(thinking_budget=...)` | 5 | 네이티브 추론 설정 |
| `langsmith.evaluation.evaluate()` | 6 | 자동 평가 |

---

## 그래프 구조 변화

### Before (Phase 02)

```
START → summary_node → tool_selector → [4개 도구] → result_processor → response_generator → END
                            ↑                              ↓
                            └────── continue ──────────────┘
```

**노드 수**: 8개
**조건부 엣지**: 2개 (tool_selector, result_processor)

### After (Phase 03-3: Tool Calling)

```
START → summary_node → llm_node → tool_node → llm_node → ... → END
                           ↓          ↑
                           └──────────┘
```

**노드 수**: 3개
**조건부 엣지**: 1개 (tools_condition)

### After (Phase 03-5: Thinking 활성화 시)

```
START → summary_node → llm_node ⇄ tool_node → ... → END
```

**그래프 구조 변경 없음** — `ChatGoogleGenerativeAI`가 `thinking_budget`을 네이티브 지원하므로 별도 노드 불필요.
**노드 수**: 3개 (03-3과 동일)
**조건부 엣지**: 1개 (tools_condition)

---

## 품질 개선 효과

| 항목 | Before | After |
|------|--------|-------|
| 도구 선택 안정성 | 문자열 파싱 (불안정) | bind_tools (안정) |
| 결과 파싱 | JSON 수동 파싱 | Pydantic 자동 |
| 응답 UX | 전체 대기 | 토큰 스트리밍 |
| 추론 품질 | 프롬프트만 | thinking_budget |
| 모니터링 | 없음 | LangSmith 트레이싱 |
| 품질 측정 | 수동 | 자동 평가 |

---

## 참고 문서

- [LangSmith Docs](https://docs.smith.langchain.com/)
- [LangChain Structured Output](https://docs.langchain.com/oss/python/langchain/models)
- [LangGraph Prebuilt](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [Google GenAI Thinking](https://github.com/googleapis/python-genai)
