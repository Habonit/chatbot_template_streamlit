# Phase 03-4, 5, 6 비판적 리뷰

> 작성일: 2026-02-06
> 대상: `_04.md` (Streaming), `_05.md` (Thinking), `_06.md` (평가 시스템)
> 기준: 현재 코드베이스 (Phase 03-3 완료 상태) 대비 개발 가능성 및 아키텍처 정합성

---

## 현재 코드 상태 요약

Phase 03-3까지 완료:
- `bind_tools()` + `ToolNode` + `tools_condition` 패턴 적용 완료
- Casual Mode 분기 + `normal_turn_ids` 기반 요약 관리 완료
- `extract_text_from_content` 등 버그 수정 완료
- `react_graph.py` 약 693줄, `tools.py` 98줄

---

## 1. _04.md (Streaming) 리뷰

### 긍정적 평가

- `stream_mode="messages"` 선택 적절. `astream_events` 대비 Streamlit 동기 호환성 확보, LangGraph 1.0.7 지원 확인됨
- `invoke()`와 `stream()`의 시그니처 동일 유지 설계 합리적
- `done` 이벤트에 metadata 일괄 반환하는 구조 깔끔
- fallback 전략(`use_streaming=False`) 포함

### 문제점

#### 1.1 `stream()`과 `invoke()`의 코드 중복 심각

`stream()` 메서드의 초기화 로직(casual 분기, 메시지 변환, initial_state 구성, normal_turn_ids 처리)이 `invoke()`와 거의 완전히 동일하다. 이 상태로 구현하면 **둘 중 하나를 수정할 때 반드시 다른 쪽도 동기화해야 하는 유지보수 지옥**이 된다.

**영향 범위**: Phase 03-3-2 casual mode 도입 때 이미 경험한 문제의 반복

**제안**: 공통 전처리 로직을 `_prepare_invocation()` 같은 private 메서드로 추출하고, `invoke()`와 `stream()` 모두 이를 호출하게 해야 한다.

```python
# 제안 구조
def _prepare_invocation(self, user_input, session_id, messages, ...):
    """invoke()와 stream() 공통 전처리"""
    # casual 분기 판단
    # 메시지 변환
    # initial_state 구성
    # normal_turn_ids 업데이트
    return mode, initial_state, config, updated_normal_turn_ids

def invoke(self, ...):
    mode, state, config, turn_ids = self._prepare_invocation(...)
    if mode == "casual":
        return self._invoke_casual(...)
    result = self._graph.invoke(state, config)
    return self._parse_invoke_result(result, turn_ids, ...)

def stream(self, ...):
    mode, state, config, turn_ids = self._prepare_invocation(...)
    if mode == "casual":
        yield from self._stream_casual(...)
        return
    for chunk, metadata in self._graph.stream(state, config, stream_mode="messages"):
        ...
```

#### 1.2 `_stream_casual`에서 토큰 추적이 불완전

```python
for chunk in self._llm.stream([HumanMessage(content=casual_prompt)]):
    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
        total_tokens["input"] += chunk.usage_metadata.get("input_tokens", 0)
```

LangChain의 `stream()`에서 `usage_metadata`는 **마지막 청크에만** 포함되는 경우가 대부분이다. 매 청크마다 누적 합산하면 중복 카운팅 또는 0으로 남을 위험이 있다. 현재 `invoke()`의 `_invoke_llm_with_token_tracking`과 동작이 다르다.

**제안**: 마지막 청크의 usage_metadata만 사용하거나, LangSmith 트레이싱에 위임

#### 1.3 `app.py`의 `handle_stream_message`가 `handle_chat_message`와 중복

서비스 구성(search_service, embedding_service, graph_builder 생성) 로직이 통째로 복사된다. 서비스 구성이 변경될 때마다 두 곳을 수정해야 한다.

**제안**: `_create_graph_builder(settings, embed_repo)` 팩토리 함수를 만들어 공유

```python
def _create_graph_builder(settings, embed_repo):
    """handle_chat_message와 handle_stream_message 공통 팩토리"""
    search_service = None
    embedding_service = None
    # ... 서비스 구성 로직
    return ReactGraphBuilder(...)
```

#### 1.4 `_parse_message_chunk`에서 tool_call 이벤트 처리 불완전

```python
if chunk.tool_call_chunks:
    for tc in chunk.tool_call_chunks:
        name = tc.get("name")
        if name:
            tool_calls_buffer.append({"name": name})
            return {"type": "tool_call", "name": name}  # 첫 번째만 처리
```

- `tool_call_chunks`는 청크 단위로 오므로, 한 도구 호출의 `name`이 여러 청크에 걸쳐 올 수 있음
- `return`이 첫 번째 tool_call만 처리하고 나머지를 버림
- 멀티 도구 호출 시 이벤트 누락

**제안**: tool_call_chunks를 누적 버퍼로 관리하고, 완성된 tool_call만 yield

#### 1.5 현재 턴 메시지 추출 로직 중복

`invoke()` 라인 649-661의 현재 턴 시작점 찾기 로직이 `_extract_tool_results()`에 그대로 복사된다. 공통 함수로 추출 필요.

---

## 2. _05.md (추론 모델 / Thinking) 리뷰

### 긍정적 평가

- Thinking은 최종 응답에만 적용, Tool Calling 단계는 기존 LLM 유지하는 판단 합리적
- 모델 체크 로직(`THINKING_SUPPORTED_MODELS`) 포함
- 추가 의존성 없음 (`google-genai` 1.61.0 이미 설치됨)

### 문제점

#### 2.1 [치명적] LLM 이중 호출 — 비용/시간 2배 낭비

현재 설계:
```
llm_node (ChatGoogleGenerativeAI) → tool_calls 없음 → thinking_node (google-genai SDK) → END
```

`llm_node`가 이미 최종 응답을 생성한 후, `thinking_node`에서 **다시 한번** 전체 컨텍스트로 LLM을 호출한다.

| 문제 | 영향 |
|------|------|
| 토큰 비용 2배 증가 | LLM 2회 호출 (llm_node + thinking_node) |
| 응답 시간 2배 증가 | 직렬 실행 |
| llm_node의 첫 응답 완전히 버려짐 | 생성 토큰 낭비 |
| 두 응답 간 일관성 보장 불가 | 다른 내용이 나올 수 있음 |

**문서 원문**: "기존 LLM 응답을 thinking 응답으로 교체" — 이전 응답 생성에 든 토큰을 완전히 낭비한다는 뜻

**제안 (3가지 대안)**:

```
대안 A: llm_node 자체를 조건부로 교체
- thinking_budget > 0이면 llm_node 내부에서 google-genai SDK 사용
- bind_tools가 안 되므로 도구 판단은 별도 로직 필요
- 장점: 호출 1회, 단점: 복잡도 증가

대안 B: thinking_node에 도구 결과만 전달 (llm_node 최종 응답 생략)
- llm_node에서 tool_calls가 없을 때 응답 생성을 건너뛰고
  바로 thinking_node에 대화 컨텍스트 + 도구 결과만 전달
- 장점: 호출 1회, 단점: llm_node 분기 복잡화

대안 C: ReAct 루프 전체를 google-genai SDK로 교체 (thinking 모드 전용 그래프)
- thinking 활성화 시 완전히 다른 그래프 사용
- 장점: 깔끔한 분리, 단점: 코드 중복
```

#### 2.2 SDK 이원화로 인한 기술 부채

`langchain-google-genai` (ChatGoogleGenerativeAI)와 `google-genai` (genai.Client)를 **동시에 사용**:

| 항목 | langchain-google-genai | google-genai |
|------|----------------------|--------------|
| 인증 | `google_api_key` 파라미터 | `genai.Client(api_key=)` |
| 메시지 | `HumanMessage`, `AIMessage` | `types.Content(role=, parts=)` |
| 토큰 추적 | `response.usage_metadata` | `response.usage_metadata.prompt_token_count` |
| 스트리밍 | `.stream()` → `AIMessageChunk` | `.generate_content_stream()` → 자체 청크 |

이로 인한 비용:
- `_convert_messages_to_genai_contents()` 변환 함수 필요 (36줄)
- 두 SDK 버전 관리 복잡화
- 토큰 추적 방식 불일치

#### 2.3 메시지 변환에서 ToolMessage → user role 변환이 위험

```python
elif hasattr(msg, "type") and msg.type == "tool":
    tool_summary = f"[{msg.name} 결과] {str(msg.content)[:300]}"
    contents.append(
        types.Content(role="user", parts=[types.Part(text=tool_summary)])
    )
```

- 모델이 사용자가 도구 결과를 직접 입력한 것으로 오해할 수 있음
- 연속된 user role 메시지가 생겨 Gemini API 에러 발생 가능
- 도구 결과 300자 잘림으로 정보 손실

#### 2.4 Streaming + Thinking 통합이 사실상 미완성

문서 자체에서 인정:
> "thinking_node는 google-genai SDK를 직접 사용하므로, stream_mode="messages"에서는 완성된 AIMessage가 한 번에 옵니다."

도구 호출까지는 실시간 스트리밍이되, thinking_node에서 **다시 긴 대기**가 발생한다. Phase 03-4의 "Thinking... 대기 제거" 목표와 모순된다.

#### 2.5 `thought_process`를 ChatState에 저장하는 것은 낭비

`thought_process`를 ChatState에 넣으면 SqliteSaver가 매 체크포인트마다 저장한다. 사고 과정은 일시적 디스플레이 정보인데, DB에 누적 저장하는 것은 저장 공간 낭비.

**제안**: `invoke()` 반환값에만 포함하고, ChatState에서는 제외

#### 2.6 `build()` 분기 로직 복잡화

```python
if self.thinking_budget > 0 and self._genai_client:
    # thinking 버전 그래프
else:
    # 기존 그래프
```

이후 Phase에서 기능이 추가될 때마다 이 분기가 더 복잡해진다.

**제안**: 그래프 구조를 동적으로 조합하는 빌더 패턴 적용

---

## 3. _06.md (평가 시스템) 리뷰

### 긍정적 평가

- `evaluation/` 폴더를 독립적으로 구성하여 기존 코드에 영향 없음
- 평가자 함수들이 현재 `invoke()` 반환 구조의 필드명과 일치 (`tool_history`, `text`, `total_tokens`)
- `_target_function` 래퍼가 LangSmith `evaluate()` 인터페이스에 적합

### 문제점

#### 3.1 _04, _05 완성 후 인터페이스 변경 미반영

`run_evaluation.py`가 `ReactGraphBuilder`를 직접 호출하는데, _05 구현 후 시그니처가 변경됨 (`thinking_budget`, `show_thoughts` 추가). 문서에서 이를 반영하지 않았다.

```python
# _06 문서의 코드 — _05 파라미터 없음
graph_builder = ReactGraphBuilder(
    api_key=api_key,
    model=DEFAULT_MODEL,
    temperature=DEFAULT_TEMPERATURE,
    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
)
```

#### 3.2 [치명적] 세션 격리 미흡

모든 평가 케이스가 **같은 session_id("evaluation")를 공유**:

```python
def chatbot_fn(question: str) -> dict:
    turn_counter["count"] += 1
    return graph_builder.invoke(
        user_input=question,
        session_id="evaluation",  # 모든 테스트 공유
        turn_count=turn_counter["count"],
    )
```

| 문제 | 영향 |
|------|------|
| SqliteSaver 체크포인트 누적 | 이전 평가의 대화 이력이 다음 평가에 영향 |
| 요약 트리거 의도치 않게 발동 | 7개 테스트 실행 시 turn 4, 7에서 요약 생성 |
| 평가 간 독립성 보장 안 됨 | 결과 재현성 없음 |

**제안**:
```python
def chatbot_fn(question: str) -> dict:
    unique_session = f"eval-{uuid.uuid4().hex[:8]}"
    return graph_builder.invoke(
        user_input=question,
        session_id=unique_session,
        turn_count=1,  # 항상 첫 턴
    )
```
또는 `db_path=":memory:"` 사용

#### 3.3 `_summarize_results`의 결과 파싱이 LangSmith SDK 버전 의존적

```python
for r in results_list:
    eval_results = r.get("evaluation_results", {}).get("results", [])
```

`langsmith.evaluation.evaluate()`는 `ExperimentResults` 객체를 반환하며, 이터레이션 시 `ExperimentResultRow`를 yield한다. 이 필드명이 문서의 가정과 다를 수 있다.

**제안**: LangSmith SDK의 실제 반환 타입을 확인하고, 타입 힌트 추가

#### 3.4 `tool_usage_correct` 평가자의 한계

```python
score = 1 if expected_tool in actual_tools else 0
```

| 미검출 케이스 | 설명 |
|-------------|------|
| 기대 도구 + 불필요한 추가 도구 사용 | 만점 처리됨 |
| 도구 순서 오류 | 감지 불가 |
| 도구 인자(query) 부적절 | 평가 안 함 |

#### 3.5 테스트 데이터가 너무 적고 하드코딩

7개 테스트 케이스로 챗봇을 평가하는 것은 통계적으로 불충분. 기대값이 하드코딩되어 있어서 프롬프트나 도구 변경 시 데이터셋도 수정 필요.

**제안**: 외부 JSON/CSV 파일에서 로드하는 방식으로 변경

#### 3.6 RAG 평가가 사실상 빈 껍데기

`runner.py`에서 `EmbeddingService`, `EmbeddingRepository`를 주입하지 않으므로, `search_pdf_knowledge` 도구가 항상 "설정되지 않았습니다"를 반환한다. PDF 업로드 → 임베딩 생성 → 평가 실행의 전체 파이프라인이 설계되지 않았다.

---

## 4. 교차 의존성 문제 (4-5-6 간)

| 문제 | 설명 |
|------|------|
| **4→5 순서 강제** | _05의 `_parse_message_chunk` 확장은 _04의 코드 위에 구현됨. _04 없이 _05 불가 |
| **5→6 미반영** | _06이 _05의 `thinking_budget` 파라미터를 인지하지 못함 |
| **4+5 동시 적용 시 `stream()` 중단** | _04의 `stream()` + _05의 `thinking_node` 조합 시 스트리밍 중단 구간 발생 |
| **코드 중복 3중화** | `invoke()`, `stream()`, `chatbot_fn()`(평가)에서 동일한 초기화/결과 파싱 로직 3번 작성 |

---

## 5. 권장 구현 순서

| 순서 | 작업 | 이유 |
|------|------|------|
| **0** | `invoke()` 리팩토링 | 공통 전처리/후처리 추출. _04, _05, _06 모두의 기반 |
| **1** | _04 (Streaming) | `stream()`이 리팩토링된 공통 로직을 재사용 |
| **2** | _05 (Thinking) 재설계 | 이중 호출 문제 해결 후 구현. _04의 스트리밍 기반 위에 확장 |
| **3** | _06 (평가) | _04, _05 완성 후 최종 형태로 평가 |

### 선행 리팩토링 상세 (순서 0)

`react_graph.py`에서 추출할 공통 로직:

```python
# 1. 전처리 (invoke, stream 공통)
def _prepare_invocation(self, user_input, session_id, messages, ...):
    """casual 분기 판단, 메시지 변환, initial_state 구성, normal_turn_ids 업데이트"""
    ...

# 2. 현재 턴 메시지 추출 (invoke, stream, 평가 공통)
def _extract_current_turn_messages(self, result_messages, turn_count):
    """turn_id 기반으로 현재 턴 시작점 찾기"""
    ...

# 3. 결과 파싱 (invoke, stream done 이벤트 공통)
def _parse_result(self, result_messages, turn_count, updated_normal_turn_ids, ...):
    """final_text, tool_history, tool_results 추출"""
    ...
```

`app.py`에서 추출할 공통 로직:

```python
# 4. GraphBuilder 팩토리 (handle_chat_message, handle_stream_message 공통)
def _create_graph_builder(settings, embed_repo):
    """서비스 구성 + ReactGraphBuilder 생성"""
    ...
```

---

## 6. 결론

### _04 (Streaming)
기술적 방향 올바름. **코드 중복이 핵심 문제**. 구현 전 `invoke()` 공통 로직 추출 리팩토링이 선행되어야 한다. 이것 없이 진행하면 _05 추가 시 유지보수가 극도로 어려워진다.

### _05 (Thinking)
**설계에 근본적 결함** 존재. LLM 이중 호출(llm_node → thinking_node)은 비용/시간 2배 낭비이며, SDK 이원화로 인한 기술 부채가 크다. thinking을 별도 노드가 아닌 llm_node의 조건부 동작으로 재설계해야 한다. 현재 형태로 구현하면 나중에 반드시 뜯어고쳐야 한다.

### _06 (평가)
가장 독립적이지만, **세션 격리 미흡**과 **RAG 평가 미완성**이 문제. _04, _05 구현 후 인터페이스가 변경될 것이므로 지금 구현하면 수정 불가피.

### 총평
3개 문서 모두 방향은 맞지만, _04와 _05를 현재 형태 그대로 순차 구현하면 `react_graph.py`가 **500줄 이상의 중복 코드 덩어리**가 된다. 공통 로직 추출 리팩토링을 _04 이전에 수행하는 것을 강력히 권한다.
