# Phase 03-3-2: Casual Mode와 요약 턴 관리

## 1. 개요

Casual 모드(일상 대화)는 요약 대상에서 제외하고, 턴 카운트에서도 배제한다.

### 1.1 배경

- Casual 모드: "안녕", "고마워" 등 단순 인사/감사 표현
- 이런 대화는 맥락 유지에 중요하지 않음
- 요약 토큰 낭비 방지

### 1.2 턴 분류

| 모드 | 예시 | 요약 대상 | 요약 턴 카운트 |
|------|------|----------|---------------|
| normal | "LangChain이 뭐야?" | ✅ 포함 | ✅ 포함 |
| casual | "안녕", "고마워" | ❌ 제외 | ❌ 제외 |

### 1.3 모드별 처리 현황

| 모드 | 의도된 동작 | 현재 구현 | 비고 |
|------|------------|----------|------|
| casual | Fast-path, 그래프 우회 | ✅ 구현됨 | 본 문서 범위 |
| reasoning | gemini-2.5-pro + thinking | ❌ 미구현 | Phase 03-5 예정 |
| normal | 기본 모델로 그래프 실행 | ✅ 구현됨 | - |

**중요**: 현재 `reasoning`과 `normal`은 동일하게 처리됩니다.

```python
# 현재 invoke() 분기 구조
mode = detect_reasoning_need(user_input)

if mode == "casual":
    # ✅ Fast-path: 그래프 우회, 요약 제외
    return casual_response

# reasoning, normal 모두 동일 경로
# ❌ reasoning 전용 처리 없음 (Phase 03-5에서 구현 예정)
result = self._graph.invoke(...)
```

**Phase 03-5 예정 사항** (`doc/phase_03/_05.md` 참조):
- `thinking_budget` 파라미터로 추론 토큰 예산 설정
- `google-genai` SDK의 `ThinkingConfig` 사용
- 사고 과정(`thought`) UI 표시

---

## 2. 현재 구현 상태 (AS-IS)

### 2.1 현재 코드 구조

```python
# service/react_graph.py - ChatState
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    turn_count: int           # 전체 턴 (casual 포함)
    compression_rate: float
    summary_history: list
    # normal_turn_count 없음
    # normal_turn_ids 없음

# service/react_graph.py - invoke()
def invoke(self, user_input, session_id, messages, turn_count, ...):
    mode = detect_reasoning_need(user_input)
    if mode == "casual":
        return {...}  # Fast-path, 그래프 스킵
    # normal_turn_ids 파라미터 없음

# app.py
# normal_turn_ids session_state 없음
```

### 2.2 현재 메시지 구조

```python
# LangChain 메시지에 turn_id 정보 없음
user_message = HumanMessage(content=user_input)
# turn_id를 알 수 없음 → 비연속 턴 추출 불가
```

### 2.3 현재 문제점

| 문제 | 설명 |
|------|------|
| casual 턴 포함 | casual도 turn_count에 포함되어 요약 트리거 타이밍 불정확 |
| 메시지 turn_id 부재 | HumanMessage에 turn_id 메타데이터 없어 특정 턴 추출 불가 |
| 세션 복원 미지원 | normal_turn_ids 상태 저장/복원 없음 |

---

## 3. 설계 결정 사항

### 3.1 메시지에 turn_id 추가 방식

**선택: Option A - additional_kwargs 사용**

```python
# 변경 전
user_message = HumanMessage(content=user_input)

# 변경 후
user_message = HumanMessage(
    content=user_input,
    additional_kwargs={"turn_id": turn_count, "mode": "normal"}
)
```

**이유:**
- LangChain 메시지 구조 유지
- SqliteSaver 호환성 유지 (additional_kwargs는 자동 직렬화됨)
- 기존 코드 영향 최소화

**대안 비교:**

| 옵션 | 장점 | 단점 |
|------|------|------|
| A. additional_kwargs | 간단, 호환성 좋음 | 메시지마다 메타데이터 추가 필요 |
| B. 별도 매핑 테이블 | 메시지 구조 변경 없음 | 동기화 복잡, 에러 가능성 |
| C. 기존 방식 유지 | 변경 없음 | 비연속 턴 추출 불가 |

### 3.2 summary_history 구조

**선택: 연속 인덱스 + 실제 요약 턴 분리**

```json
{
  "thread_id": "session_123",
  "turns": [1, 2, 3],              // 전체 턴 범위 (UI 표시용)
  "summarized_turns": [1, 3],      // 실제 요약된 normal 턴
  "excluded_turns": [2],           // casual로 제외된 턴
  "turn_length": 3,
  "original_chars": 500,
  "summary_chars": 150,
  "compression_rate": 0.3,
  "summary": "요약 내용"
}
```

**이유:**
- UI에서 "Turn 1-3 요약" 표시 가능 (혼란 방지)
- 실제 요약 대상 턴 추적 가능 (디버깅)
- excluded_turns로 casual 턴 명시

### 3.3 연속 casual 처리 정책

**문제 시나리오:**
```
Turn 1: normal → normal_count = 1
Turn 2: casual → normal_count = 1
Turn 3: casual → normal_count = 1
Turn 4: casual → normal_count = 1
Turn 5: normal → normal_count = 2
... (casual이 계속되면 요약이 무한정 지연)
```

**정책: Fallback 트리거 추가**

```python
# 요약 트리거 조건
def should_summarize(normal_turn_count: int, total_turn_count: int) -> bool:
    # 기본: normal 턴 4, 7, 10...
    if normal_turn_count >= 4 and (normal_turn_count - 1) % 3 == 0:
        return True
    # Fallback: 전체 턴 10개마다 강제 요약 (토큰 관리)
    if total_turn_count >= 10 and total_turn_count % 10 == 0:
        return True
    return False
```

---

## 4. 변경 사항 (TO-BE)

### 4.1 ChatState 스키마 변경

```python
# service/react_graph.py

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    turn_count: int                    # 전체 턴 (UI 표시용)
    normal_turn_count: int             # [신규] normal 턴 카운트
    normal_turn_ids: list[int]         # [신규] normal 턴 ID 목록
    compression_rate: float
    summary_history: list
    # ... 기타 필드
```

### 4.2 invoke() 시그니처 변경

```python
def invoke(
    self,
    user_input: str,
    session_id: str,
    messages: list = None,
    summary: str = "",
    pdf_description: str = "",
    turn_count: int = 0,
    summary_history: list = None,
    compression_rate: float = 0.3,
    normal_turn_ids: list = None,        # [신규]
) -> dict:
```

### 4.3 invoke() 로직 변경

```python
def invoke(self, ...):
    mode = detect_reasoning_need(user_input)

    if normal_turn_ids is None:
        normal_turn_ids = []

    if mode == "casual":
        # casual: normal_turn_ids 변경 없이 반환
        casual_response = self._generate_casual_response(user_input)
        return {
            "text": casual_response,
            "normal_turn_ids": normal_turn_ids,        # 그대로 유지
            "normal_turn_count": len(normal_turn_ids), # 변화 없음
            "summary_history": summary_history or [],
            "is_casual": True,
            ...
        }

    # normal: turn_id 추가
    updated_normal_turn_ids = normal_turn_ids + [turn_count]
    normal_turn_count = len(updated_normal_turn_ids)

    # 메시지에 turn_id 메타데이터 추가
    user_message = HumanMessage(
        content=user_input,
        additional_kwargs={"turn_id": turn_count, "mode": "normal"}
    )

    # 그래프 실행
    result = self._graph.invoke({
        "messages": [user_message],
        "turn_count": turn_count,
        "normal_turn_count": normal_turn_count,
        "normal_turn_ids": updated_normal_turn_ids,
        ...
    })

    return {
        "normal_turn_ids": updated_normal_turn_ids,
        "normal_turn_count": normal_turn_count,
        ...
    }
```

### 4.4 메시지 추출 함수

```python
def extract_messages_by_turn_ids(messages: list, turn_ids: list[int]) -> list:
    """특정 turn_id에 해당하는 메시지만 추출

    Args:
        messages: 전체 메시지 리스트 (additional_kwargs에 turn_id 포함)
        turn_ids: 추출할 턴 ID 목록 (예: [1, 3, 4])

    Returns:
        해당 턴의 메시지만 포함한 리스트
    """
    result = []
    for msg in messages:
        turn_id = msg.additional_kwargs.get("turn_id")
        if turn_id in turn_ids:
            result.append(msg)
    return result
```

### 4.5 _summary_node() 변경

```python
def _summary_node(self, state: ChatState) -> dict:
    normal_turn_count = state.get("normal_turn_count", 0)
    total_turn_count = state.get("turn_count", 0)
    normal_turn_ids = state.get("normal_turn_ids", [])
    messages = state.get("messages", [])

    # 요약 불필요
    if not should_summarize(normal_turn_count, total_turn_count):
        return {"summary_history": summary_history}

    # 요약할 normal 턴 ID (최근 3개)
    turns_to_summarize = normal_turn_ids[-3:]

    # 전체 턴 범위 계산
    if turns_to_summarize:
        start_turn = turns_to_summarize[0]
        end_turn = turns_to_summarize[-1]
        all_turns_in_range = list(range(start_turn, end_turn + 1))
        excluded_turns = [t for t in all_turns_in_range if t not in turns_to_summarize]
    else:
        all_turns_in_range = []
        excluded_turns = []

    # 해당 턴의 메시지만 추출
    messages_to_summarize = extract_messages_by_turn_ids(messages, turns_to_summarize)

    # 요약 생성
    summary_text = self._generate_summary(messages_to_summarize, compression_rate)

    summary_history.append({
        "thread_id": session_id,
        "turns": all_turns_in_range,           # [1, 2, 3] - UI 표시용
        "summarized_turns": turns_to_summarize, # [1, 3] - 실제 요약 턴
        "excluded_turns": excluded_turns,       # [2] - casual 턴
        "turn_length": len(turns_to_summarize),
        "original_chars": original_chars,
        "summary_chars": len(summary_text),
        "compression_rate": compression_rate,
        "summary": summary_text,
    })

    return {"summary_history": summary_history}
```

### 4.6 app.py 변경

```python
# init_session_state()에 추가
def init_session_state():
    # ... 기존 코드
    if "normal_turn_ids" not in st.session_state:
        st.session_state.normal_turn_ids = []

# handle_chat_message()에서 invoke 호출
result = graph_builder.invoke(
    user_input=user_input,
    session_id=session_id,
    messages=st.session_state.messages[:-1],
    turn_count=turn_count,
    normal_turn_ids=st.session_state.normal_turn_ids,  # [신규]
    ...
)

# 결과 업데이트
if "normal_turn_ids" in result:
    st.session_state.normal_turn_ids = result["normal_turn_ids"]

# load_session_data()에서 복원
def load_session_data(session_id, session_manager, embed_repo):
    # ... 기존 코드
    metadata = session_manager.get_session_metadata(session_id)
    st.session_state.normal_turn_ids = metadata.get("normal_turn_ids", [])
```

### 4.7 SqliteSaver 저장 (자동)

ChatState에 `normal_turn_ids` 필드가 있으면 SqliteSaver가 자동으로 저장/복원합니다.

```python
# 별도 코드 불필요 - ChatState 필드로 선언하면 자동 처리
class ChatState(TypedDict):
    normal_turn_ids: list[int]  # SqliteSaver가 자동 직렬화
```

### 4.8 UI 변경 (chat_tab.py)

```python
def format_summary_card(summary_entry: dict) -> str:
    """요약 히스토리 카드 포맷팅"""
    turns = summary_entry.get("turns", [])
    excluded = summary_entry.get("excluded_turns", [])

    if turns:
        # 범위 표시: "Turn 1-3"
        turns_str = f"{min(turns)}-{max(turns)}" if len(turns) > 1 else str(turns[0])
    else:
        turns_str = "?"

    summary = summary_entry.get("summary", "")

    # excluded 턴이 있으면 표시
    if excluded:
        excluded_str = f"\n*({', '.join(map(str, excluded))}턴 제외)*"
    else:
        excluded_str = ""

    return f"**Turn {turns_str}**{excluded_str}\n\n{summary}"
```

---

## 5. 시퀀스 다이어그램

```
User Input → detect_reasoning_need()
              │
              ├─ casual → Fast-path 응답
              │           - normal_turn_ids 유지
              │           - normal_turn_count 유지
              │           - 그래프 스킵
              │
              └─ normal → HumanMessage(turn_id=N, mode="normal")
                          - normal_turn_ids.append(turn_count)
                          - normal_turn_count = len(normal_turn_ids)
                          │
                          └─ 그래프 실행
                              │
                              └─ should_summarize(normal_turn_count, turn_count)?
                                  │
                                  ├─ True → summary_node 실행
                                  │         - extract_messages_by_turn_ids()
                                  │         - summary_history 업데이트
                                  │
                                  └─ False → summary_node 스킵
```

---

## 6. 시나리오 예시

### 6.1 기본 시나리오

```
Turn 1: "LangChain이 뭐야?" (normal) → normal_ids=[1], count=1
Turn 2: "고마워" (casual)            → normal_ids=[1], count=1 (변화 없음)
Turn 3: "Docker 설명해줘" (normal)   → normal_ids=[1,3], count=2
Turn 4: "React vs Vue" (normal)     → normal_ids=[1,3,4], count=3
Turn 5: "Kubernetes란?" (normal)    → normal_ids=[1,3,4,5], count=4 → 요약 트리거!

요약 결과:
{
  "turns": [1, 2, 3, 4],           # 전체 범위 (UI: "Turn 1-4")
  "summarized_turns": [1, 3, 4],   # 실제 요약된 턴
  "excluded_turns": [2],           # casual 제외
  "summary": "LangChain, Docker, React vs Vue 설명..."
}
```

### 6.2 연속 casual 시나리오 (Fallback)

```
Turn 1: normal  → count=1
Turn 2: casual  → count=1
Turn 3: casual  → count=1
...
Turn 9: casual  → count=1
Turn 10: casual → count=1, total=10 → Fallback 요약 트리거!

요약 결과:
{
  "turns": [1, 2, ..., 10],
  "summarized_turns": [1],         # normal은 1개뿐
  "excluded_turns": [2,3,4,5,6,7,8,9,10],
  "summary": "Turn 1의 내용 요약..."
}
```

---

## 7. 테스트 계획

### 7.1 테스트 파일: tests/test_casual_mode.py

```python
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from service.react_graph import (
    ReactGraphBuilder,
    extract_messages_by_turn_ids,
    should_summarize,
)


class TestMessageTurnIdMetadata:
    """메시지에 turn_id 메타데이터 추가 테스트"""

    def test_human_message_has_turn_id(self):
        """HumanMessage에 turn_id 포함"""
        msg = HumanMessage(
            content="테스트",
            additional_kwargs={"turn_id": 1, "mode": "normal"}
        )
        assert msg.additional_kwargs["turn_id"] == 1
        assert msg.additional_kwargs["mode"] == "normal"


class TestExtractMessagesByTurnIds:
    """turn_id 기반 메시지 추출 테스트"""

    def test_extract_single_turn(self):
        """단일 턴 추출"""
        messages = [
            HumanMessage(content="Q1", additional_kwargs={"turn_id": 1}),
            AIMessage(content="A1"),
            HumanMessage(content="Q2", additional_kwargs={"turn_id": 2}),
            AIMessage(content="A2"),
        ]
        result = extract_messages_by_turn_ids(messages, [1])
        assert len(result) == 1
        assert result[0].content == "Q1"

    def test_extract_non_consecutive_turns(self):
        """비연속 턴 추출 [1, 3]"""
        messages = [
            HumanMessage(content="Q1", additional_kwargs={"turn_id": 1}),
            AIMessage(content="A1"),
            HumanMessage(content="Q2", additional_kwargs={"turn_id": 2}),
            AIMessage(content="A2"),
            HumanMessage(content="Q3", additional_kwargs={"turn_id": 3}),
            AIMessage(content="A3"),
        ]
        result = extract_messages_by_turn_ids(messages, [1, 3])
        assert len(result) == 2
        assert result[0].content == "Q1"
        assert result[1].content == "Q3"


class TestShouldSummarizeWithFallback:
    """요약 트리거 조건 테스트 (Fallback 포함)"""

    def test_normal_trigger_at_4(self):
        """normal_count=4에서 트리거"""
        assert should_summarize(4, 4) is True

    def test_no_trigger_at_3(self):
        """normal_count=3에서 트리거 안함"""
        assert should_summarize(3, 3) is False

    def test_fallback_trigger_at_total_10(self):
        """total=10에서 Fallback 트리거"""
        assert should_summarize(1, 10) is True


class TestCasualModeIntegration:
    """Casual 모드 통합 테스트"""

    @pytest.fixture
    def api_key(self):
        import os
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            pytest.skip("GEMINI_API_KEY 환경 변수 필요")
        return key

    def test_casual_does_not_change_normal_turn_ids(self, api_key):
        """casual 입력이 normal_turn_ids를 변경하지 않음"""
        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        result = builder.invoke(
            user_input="안녕",  # casual
            session_id="test_casual",
            turn_count=2,
            normal_turn_ids=[1],
        )

        assert result["normal_turn_ids"] == [1]
        assert result["is_casual"] is True

    def test_normal_appends_to_normal_turn_ids(self, api_key):
        """normal 입력이 normal_turn_ids에 추가됨"""
        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        result = builder.invoke(
            user_input="Docker 설명해줘",  # normal
            session_id="test_normal",
            turn_count=3,
            normal_turn_ids=[1],
        )

        assert result["normal_turn_ids"] == [1, 3]

    def test_summary_excludes_casual_turns(self, api_key):
        """요약에서 casual 턴 제외"""
        builder = ReactGraphBuilder(api_key=api_key, db_path=":memory:")
        builder.build()

        # 시뮬레이션: Turn 1(N), 2(C-제외), 3(N), 4(N)
        messages = [
            HumanMessage(content="Q1", additional_kwargs={"turn_id": 1, "mode": "normal"}),
            AIMessage(content="A1", tool_calls=[]),
            # Turn 2 (casual)는 메시지에 포함되지 않음 (Fast-path)
            HumanMessage(content="Q3", additional_kwargs={"turn_id": 3, "mode": "normal"}),
            AIMessage(content="A3", tool_calls=[]),
            HumanMessage(content="Q4", additional_kwargs={"turn_id": 4, "mode": "normal"}),
            AIMessage(content="A4", tool_calls=[]),
        ]

        result = builder.invoke(
            user_input="Q5",  # normal, count=4 → 트리거
            session_id="test_exclude_casual",
            messages=messages,
            turn_count=5,
            normal_turn_ids=[1, 3, 4],  # Turn 2 없음
        )

        if result.get("summary_history"):
            summary = result["summary_history"][0]
            assert 2 not in summary.get("summarized_turns", [])
```

---

## 8. Tool History 누적 버그 수정

### 8.1 현재 문제점

**문제**: 툴 사용 정보가 현재 턴만이 아닌 이전 턴까지 누적되어 표시됨

```
Turn 1: web_search 사용 → tool_history: ["web_search"]
Turn 2: 툴 미사용      → tool_history: ["web_search"]  ← 이전 턴 잔존!
Turn 3: reasoning 사용 → tool_history: ["web_search", "reasoning"]  ← 누적!
```

**원인**: `invoke()` 반환 시 `result_messages`가 전체 대화 히스토리를 포함

```python
# 현재 코드 (AS-IS) - react_graph.py:506
for msg in result_messages:  # 전체 메시지 순회 → 이전 턴 포함
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            tool_history.append(tc["name"])  # 이전 턴 도구도 추가됨
```

### 8.2 수정 방안

**방안**: 현재 턴 메시지에서만 tool_history 추출

```python
# 수정 코드 (TO-BE)
def invoke(self, ...):
    ...
    result = self._graph.invoke(...)
    result_messages = result.get("messages", [])

    # 현재 턴 메시지만 추출
    current_turn_messages = extract_current_turn(result_messages)

    # 현재 턴에서만 tool_history 추출
    tool_history = []
    tool_results = {}
    for msg in current_turn_messages:  # 현재 턴만!
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_history.append(tc["name"])
        if hasattr(msg, "type") and msg.type == "tool":
            tool_results[msg.name] = msg.content

    return {
        "tool_history": tool_history,  # 현재 턴만
        "tool_results": tool_results,  # 현재 턴만
        ...
    }
```

### 8.3 기대 동작

```
Turn 1: web_search 사용 → tool_history: ["web_search"]
Turn 2: 툴 미사용      → tool_history: []  ✅ 빈 배열
Turn 3: reasoning 사용 → tool_history: ["reasoning"]  ✅ 현재 턴만
```

### 8.4 UI 영향

`chat_tab.py`의 "🔧 툴 사용 정보" Expander가 현재 턴의 도구만 표시:

```python
# 변경 불필요 - invoke() 반환값만 수정하면 됨
if msg.function_calls or msg.tool_results:
    with st.expander("🔧 툴 사용 정보", expanded=False):
        # function_calls = 현재 턴만 포함됨
```

---

## 9. 구현 체크리스트

| # | 항목 | 파일 | 상태 |
|---|------|------|------|
| 1 | ChatState에 normal_turn_count, normal_turn_ids 추가 | react_graph.py | ✅ 완료 |
| 2 | HumanMessage에 turn_id 메타데이터 추가 | react_graph.py | ✅ 완료 |
| 3 | extract_messages_by_turn_ids() 함수 구현 | react_graph.py | ✅ 완료 |
| 4 | should_summarize() Fallback 조건 추가 | react_graph.py | ✅ 완료 |
| 5 | invoke() 시그니처 및 로직 변경 | react_graph.py | ✅ 완료 |
| 6 | _summary_node() 변경 | react_graph.py | ✅ 완료 |
| 7 | app.py normal_turn_ids 초기화/전달/업데이트 | app.py | ✅ 완료 |
| 8 | load_session_data() normal_turn_ids 복원 | app.py | ✅ 완료 |
| 9 | format_summary_card() UI 변경 | chat_tab.py | ✅ 완료 |
| 10 | 테스트 파일 작성 | test_casual_mode.py | ✅ 완료 |
| 11 | tool_history 현재 턴만 추출 | react_graph.py | ✅ 완료 |
| 12 | tool_results 현재 턴만 추출 | react_graph.py | ✅ 완료 |

---

## 10. 정리

| 항목 | AS-IS | TO-BE |
|------|-------|-------|
| 턴 카운트 기준 | turn_count (casual 포함) | normal_turn_count (normal만) |
| 요약 트리거 | turn_count 기반 | normal_turn_count + Fallback |
| 메시지 turn_id | 없음 | additional_kwargs로 저장 |
| 요약 대상 | 연속 3턴 | normal 3턴 (비연속 가능) |
| summary_history.turns | [1,2,3] | [1,2,3] + summarized_turns + excluded_turns |
| 세션 복원 | normal_turn_ids 없음 | ChatState에서 자동 복원 |
| tool_history | 전체 턴 누적 | 현재 턴만 |
| tool_results | 전체 턴 누적 | 현재 턴만 |
