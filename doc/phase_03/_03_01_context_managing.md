# Phase 03-3: Context Managing (컨텍스트 관리)

## 개요

대화가 길어질수록 LLM에 전달되는 컨텍스트가 커지는 문제를 해결한다.
3턴 단위로 요약을 생성하고, 요약 + 최근 raw 턴만 컨텍스트로 전달하여 토큰 사용량을 최적화한다.

## 목표

1. 컨텍스트 크기를 일정하게 유지
2. 대화 품질 유지 (중요 정보 손실 방지)
3. 사용자가 요약 상태를 UI에서 확인 가능

## 컨텍스트 구성 방식

### Turn별 Context 구성

| Turn | Context 구성 | 설명 |
|------|-------------|------|
| 1 | `[Turn1]` | Raw 메시지 |
| 2 | `[Turn1, Turn2]` | Raw 메시지 |
| 3 | `[Turn1, Turn2, Turn3]` | Raw 메시지 |
| 4 | `[Summary(1-3), Turn4]` | 1-3턴 요약 생성 |
| 5 | `[Summary(1-3), Turn4, Turn5]` | 요약 + Raw |
| 6 | `[Summary(1-3), Turn4, Turn5, Turn6]` | 요약 + Raw |
| 7 | `[Summary(1-3), Summary(4-6), Turn7]` | 4-6턴 요약 추가 |
| 8 | `[Summary(1-3), Summary(4-6), Turn7, Turn8]` | 요약들 + Raw |
| ... | ... | ... |
| 50 | `[Summary(1-3), ..., Summary(46-48), Turn49, Turn50]` | 16개 요약 + 2개 Raw |

### 요약 생성 규칙

- **트리거 시점**: Turn 4, 7, 10, 13, 16, ... (4부터 시작, 3턴 간격)
- **요약 범위**: 직전 3턴의 raw 대화 (user + assistant 메시지)
- **요약 저장**: `summary_history` 리스트에 누적

### Context 구성 공식

```
Context = [모든 요약들] + [요약되지 않은 최근 raw 턴들]
```

- 요약된 턴 수: `len(summary_history) * 3` (요약 1개당 3턴 커버)
- Raw 턴 수: `turn_count - 요약된_턴_수`

예시:
- Turn 7, summary_history 2개 → 요약된 턴 = 6, raw 턴 = 1

## 턴(Turn)의 정의

**1턴 = 사용자 질의 + 중간 사고 과정(Tool Calling) + AI 최종 응답**

```
┌─ Turn 1 ────────────────────────────────────────┐
│ [User] "파이썬 3.13 새 기능 검색해서 분석해줘"    │
│ [Tool Call] web_search("파이썬 3.13 새 기능")    │
│ [Tool Result] {...검색 결과...}                  │
│ [Tool Call] reasoning("검색 결과 분석")          │
│ [Tool Result] {...분석 결과...}                  │
│ [AI] "파이썬 3.13의 주요 새 기능은..."           │
└─────────────────────────────────────────────────┘
```

- Tool Calling이 많을수록 턴의 길이가 길어짐
- 요약 길이는 턴의 실제 내용 복잡도에 비례해야 함

## 데이터 구조

### summary_history 구조 (JSON)

```json
[
    {
        "thread_id": "session_abc123",
        "turns": [1, 2, 3],
        "turn_length": 3,
        "original_chars": 1500,
        "summary_chars": 450,
        "compression_rate": 0.3,
        "summary": "사용자가 인사 후 파이썬 3.13 새 기능을 검색 요청했고, AI가 웹 검색과 분석 도구를 활용하여 주요 기능(no-GIL, JIT 컴파일러 등)을 설명함. 이어서 사용자가 JIT 컴파일러의 성능 향상에 대해 추가 질문했고, AI가 벤치마크 결과를 인용하여 상세히 답변함."
    },
    {
        "thread_id": "session_abc123",
        "turns": [4, 5, 6],
        "turn_length": 3,
        "original_chars": 100,
        "summary_chars": 30,
        "compression_rate": 0.3,
        "summary": "사용자가 현재 시간을 물어봄. AI가 시간 도구로 응답."
    }
]
```

### 요약 길이 원칙

**공식: `int(original_chars * compression_rate)`**

- `original_chars`: 요약 대상 3턴의 전체 텍스트 길이 (글자 수)
- `compression_rate`: 사이드바에서 설정 (0.1 ~ 0.5, 기본값 0.3)
- `target_chars`: 요약문 목표 길이 (글자 수)

| 예시 | 원본 길이 | 압축률 | 요약 목표 |
|------|----------|--------|----------|
| 단순 대화 | 100자 | 0.3 | 30자 |
| 검색 + 분석 | 500자 | 0.3 | 150자 |
| 복잡한 다중 도구 | 1500자 | 0.3 | 450자 |

### ChatState 필드

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # raw 메시지 (HumanMessage, AIMessage, ToolMessage 포함)
    session_id: str  # thread_id로 사용
    summary_history: list[dict]  # 요약 히스토리 (JSON 형태로 누적)
    turn_count: int  # 현재 턴 번호 (user 메시지 수신 시 +1, invoke() 호출 전 app.py에서 증가)
    compression_rate: float  # 압축률 (0.1 ~ 0.5, 사이드바에서 설정, 기본값 0.3)
    # ... 기타 필드
```

### summary_history 각 항목 스키마

```python
{
    "thread_id": str,          # 세션 ID (어떤 대화 세션인지)
    "turns": list[int],        # 요약된 턴 번호들 [1,2,3] 또는 [4,5,6]
    "turn_length": int,        # 요약된 턴 개수 (항상 3)
    "original_chars": int,     # 원본 텍스트 길이 (글자 수)
    "summary_chars": int,      # 요약 텍스트 길이 (글자 수)
    "compression_rate": float, # 압축률 (0.1 ~ 0.5, 사이드바에서 설정)
    "summary": str,            # 요약 내용
}
```

### Sidebar 설정: 압축률 조절

```python
# component/sidebar.py
compression_rate = st.slider(
    "요약 압축률",
    min_value=0.1,
    max_value=0.5,
    value=0.3,
    step=0.05,
    help="낮을수록 짧게 요약, 높을수록 상세하게 요약"
)
```

| 압축률 | 설명 | 용도 |
|--------|------|------|
| 0.1 | 매우 짧은 요약 | 토큰 절약 최대화 |
| 0.3 | 균형 (기본값) | 일반적 대화 |
| 0.5 | 상세한 요약 | 중요한 맥락 유지 |

## 실행 순서 (Sequence Diagram)

### 전체 흐름

```
┌──────────┐    ┌─────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────┐    ┌──────────┐
│  User    │    │ app.py  │    │ invoke()     │    │summary_node│    │ llm_node │    │tool_node │
└────┬─────┘    └────┬────┘    └──────┬───────┘    └─────┬──────┘    └────┬─────┘    └────┬─────┘
     │               │                │                  │                │               │
     │ 메시지 입력    │                │                  │                │               │
     │──────────────>│                │                  │                │               │
     │               │                │                  │                │               │
     │               │ turn_count +1  │                  │                │               │
     │               │ (user 수신 시) │                  │                │               │
     │               │────────────────>                  │                │               │
     │               │                │                  │                │               │
     │               │                │ should_summarize │                │               │
     │               │                │ (turn_count)     │                │               │
     │               │                │─────────────────>│                │               │
     │               │                │                  │                │               │
     │               │                │                  │ [Turn 4,7,10..]│               │
     │               │                │                  │ 요약 생성      │               │
     │               │                │                  │────┐           │               │
     │               │                │                  │    │           │               │
     │               │                │                  │<───┘           │               │
     │               │                │                  │                │               │
     │               │                │                  │ summary_history│               │
     │               │                │                  │ 업데이트       │               │
     │               │                │<─────────────────│                │               │
     │               │                │                  │                │               │
     │               │                │ Context 구성     │                │               │
     │               │                │ (요약 + raw턴)   │                │               │
     │               │                │─────────────────────────────────>│               │
     │               │                │                  │                │               │
     │               │                │                  │                │ [tool_calls?] │
     │               │                │                  │                │──────────────>│
     │               │                │                  │                │               │
     │               │                │                  │                │<──────────────│
     │               │                │                  │                │               │
     │               │                │                  │                │ 최종 응답     │
     │               │                │<─────────────────────────────────│               │
     │               │                │                  │                │               │
     │<──────────────│<───────────────│                  │                │               │
     │   응답 표시   │                │                  │                │               │
```

### Turn N의 생명주기

```
┌─ Turn N 시작 ─────────────────────────────────────────────────────────────┐
│                                                                           │
│  [1] User 메시지 수신 → turn_count = N (app.py에서 증가)                   │
│      state["messages"] = [...이전 메시지들..., HumanMessage(현재 질문)]    │
│                                                                           │
│  [2] invoke() 호출                                                        │
│      - casual 대화면 fast-path로 바로 응답                                 │
│      - 아니면 그래프 실행                                                  │
│                                                                           │
│  [3] summary_node 실행                                                    │
│      - should_summarize(N) == True면 (Turn 4, 7, 10...)                   │
│        → 직전 3턴 (N-3, N-2, N-1) 요약 생성                               │
│        → summary_history에 추가                                           │
│      - False면 패스                                                       │
│                                                                           │
│  [4] llm_node 실행                                                        │
│      - Context 구성: [요약들] + [raw 턴들] + [현재 user 메시지]            │
│      - LLM 호출 (tool_calls 가능)                                         │
│                                                                           │
│  [5] tool_node 실행 (tool_calls 있으면)                                   │
│      - 도구 실행 → ToolMessage 추가                                       │
│      - llm_node로 다시 (반복 가능)                                        │
│                                                                           │
│  [6] 최종 AIMessage 생성 (tool_calls 없음)                                │
│      → Turn N 완료                                                        │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 각 시점의 데이터 상태 (Turn 4 예시)

| 시점 | turn_count | summary_history | messages 상태 |
|------|------------|-----------------|---------------|
| Turn 4 시작 | 4 | [] | [Turn1완료, Turn2완료, Turn3완료, User4] |
| summary_node 후 | 4 | [{turns:[1,2,3]}] | 동일 |
| llm_node Context | 4 | [{turns:[1,2,3]}] | System(요약) + User4 |
| Turn 4 완료 | 4 | [{turns:[1,2,3]}] | [..., User4, AI4(최종)] |

## 구현 상세

### 1. should_summarize() 함수

```python
def should_summarize(turn_count: int) -> bool:
    """요약 생성 여부 판단

    Turn 4, 7, 10, 13, ... 에서 True 반환
    """
    if turn_count < 4:
        return False
    return (turn_count - 1) % 3 == 0
```

### 2. generate_summary() 함수

```python
def generate_summary(messages: list, target_length: int) -> str:
    """LLM을 사용하여 메시지들을 요약

    Args:
        messages: 요약할 메시지 리스트 (HumanMessage, AIMessage, ToolMessage 포함)
        target_length: 목표 글자 수 (int(original_chars * compression_rate))

    Returns:
        요약 문자열 (target_length 근처 길이)

    Note:
        prompt/summary/summary_generator.py의 get_prompt() 활용
    """
    prompt = get_summary_prompt(
        messages=messages,
        target_length=target_length,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()
```

### 3. summary_node 동작

```python
def _summary_node(self, state: ChatState) -> dict:
    turn_count = state["turn_count"]
    session_id = state["session_id"]
    compression_rate = state.get("compression_rate", 0.3)  # 사이드바에서 설정

    if not should_summarize(turn_count):
        return {}  # 변경 없음

    # 요약할 메시지 추출 (직전 3턴의 모든 메시지)
    messages_to_summarize = extract_last_n_turns(state["messages"], n=3)

    # 원본 텍스트 길이 계산
    original_text = "".join(
        getattr(msg, "content", "") for msg in messages_to_summarize
    )
    original_chars = len(original_text)

    # LLM으로 요약 생성 (압축률 적용)
    target_chars = int(original_chars * compression_rate)
    new_summary = generate_summary(
        messages_to_summarize,
        target_length=target_chars
    )
    summary_chars = len(new_summary)

    # summary_history에 JSON 형태로 추가
    summary_history = state["summary_history"].copy()
    start_turn = turn_count - 3
    summary_history.append({
        "thread_id": session_id,
        "turns": [start_turn, start_turn + 1, start_turn + 2],
        "turn_length": 3,
        "original_chars": original_chars,
        "summary_chars": summary_chars,
        "compression_rate": compression_rate,
        "summary": new_summary,
    })

    return {"summary_history": summary_history}
```

### 턴 추출 로직

Tool Calling 패턴에서 1턴에 포함되는 메시지:
```
HumanMessage (사용자 질의)
├── AIMessage (tool_calls 포함, content 비어있음)
├── ToolMessage (도구 실행 결과)
├── AIMessage (tool_calls 포함, 추가 도구 호출 시)
├── ToolMessage (추가 도구 결과)
└── AIMessage (최종 응답, tool_calls 없음)
```

```python
def extract_last_n_turns(messages: list, n: int) -> list:
    """마지막 n턴의 완료된 메시지를 추출

    1턴의 끝 = AIMessage이면서 tool_calls가 없거나 빈 리스트인 경우

    Returns:
        완료된 턴들의 메시지 리스트 (진행 중인 턴은 제외)
    """
    turns = []
    current_turn = []

    for msg in messages:
        current_turn.append(msg)

        # 턴 종료 조건: AIMessage이고 tool_calls가 없거나 빈 리스트
        # (None이거나 [] 모두 턴 종료로 처리)
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            turns.append(current_turn)
            current_turn = []

    # current_turn에 남은 메시지 = 아직 완료되지 않은 진행 중인 턴
    # 이는 별도로 처리해야 함 (llm_node에서 추가)

    # 마지막 n턴 반환 (완료된 턴만)
    return [msg for turn in turns[-n:] for msg in turn]
```

### 4. llm_node에서 Context 구성

```python
def _llm_node(self, state: ChatState) -> dict:
    turn_count = state["turn_count"]
    summary_history = state["summary_history"]
    all_messages = state["messages"]

    # 요약되지 않은 raw 턴 수 계산
    # 예: Turn 4, summary_history=[{turns:[1,2,3]}]
    #     summarized_turns = 1 * 3 = 3
    #     raw_turn_count = 4 - 3 = 1 (현재 진행 중인 Turn 4)
    summarized_turns = len(summary_history) * 3
    raw_turn_count = turn_count - summarized_turns

    # Context 구성: System Prompt 먼저
    system_parts = ["당신은 유용한 AI 어시스턴트입니다."]

    # 1) 모든 요약을 System Prompt에 포함
    if summary_history:
        system_parts.append("\n[이전 대화 요약]")
        for s in summary_history:
            turns_str = f"{s['turns'][0]}-{s['turns'][-1]}턴"
            system_parts.append(f"[{turns_str}] {s['summary']}")

    system_prompt = SystemMessage(content="\n".join(system_parts))

    # 2) 최근 raw 턴의 메시지만 추출
    # raw_turn_count - 1: 완료된 이전 턴들 (extract_last_n_turns는 완료된 턴만 추출)
    completed_raw_turns = max(0, raw_turn_count - 1)
    recent_completed = extract_last_n_turns(all_messages, n=completed_raw_turns) if completed_raw_turns > 0 else []

    # 3) 현재 진행 중인 턴의 메시지 추출 (아직 완료되지 않은 턴)
    # 마지막 HumanMessage부터 끝까지 = 현재 턴
    current_turn_messages = extract_current_turn(all_messages)

    # 최종 Context = System Prompt + 완료된 raw 턴들 + 현재 턴 메시지들
    context_messages = [system_prompt] + recent_completed + current_turn_messages

    # LLM 호출
    response = self._llm_with_tools.invoke(context_messages)
    return {"messages": [response]}


def extract_current_turn(messages: list) -> list:
    """현재 진행 중인 턴의 메시지 추출

    마지막 완료된 턴 이후의 모든 메시지를 반환
    (마지막 HumanMessage부터 끝까지)
    """
    # 뒤에서부터 마지막 완료된 턴 찾기
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        # 완료된 턴의 끝 = AIMessage이고 tool_calls 없음
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            # 그 다음부터 끝까지가 현재 턴
            return messages[i + 1:]

    # 완료된 턴이 없으면 전체가 현재 턴
    return messages
```

### Context 예시 (Turn 7)

```
[SystemMessage]
  당신은 유용한 AI 어시스턴트입니다.

  [이전 대화 요약]
  [1-3턴] 사용자가 인사 후 파이썬 3.13 새 기능을 검색 요청...
  [4-6턴] 사용자가 간단히 현재 시간을 물어봄...

[HumanMessage] "JIT 컴파일러 성능 벤치마크 더 자세히 알려줘"
```

## UI 표시 (Streamlit)

### 사이드바: 압축률 설정

```
┌─────────────────────────────────────┐
│ ⚙️ 고급 설정                        │
├─────────────────────────────────────┤
│ 요약 압축률                         │
│ ○───────●───────○                  │
│ 0.1    0.3    0.5                  │
│ (짧게)  (균형)  (상세)              │
└─────────────────────────────────────┘
```

### 채팅 탭 오른쪽 패널

```
┌─────────────────────────────────────┐
│ 📝 대화 요약 히스토리                │
│ 현재 압축률: 0.3                    │
├─────────────────────────────────────┤
│ [1-3턴 요약] 압축률 0.3             │
│ 1500자 → 450자                     │
│ 사용자가 인사하고 파이썬 3.13...    │
├─────────────────────────────────────┤
│ [4-6턴 요약] 압축률 0.3             │
│ 100자 → 30자                       │
│ 사용자가 현재 시간을 물어봄...       │
├─────────────────────────────────────┤
│ [7-9턴 요약] 압축률 0.3             │
│ 800자 → 240자                      │
│ 저녁 약속에 대해 이야기했으며...     │
└─────────────────────────────────────┘
│                                     │
│ 📊 요약 통계                        │
│ 총 요약: 3개 (9턴 압축)             │
│ 원본: 2400자 → 요약: 720자         │
│ 전체 압축률: 70% 절감               │
└─────────────────────────────────────┘
```

사용자는 이 패널을 통해:
- **압축률 확인**: 각 요약에 적용된 compression_rate
- **압축 효과 확인**: 원본 → 요약 글자 수 변화
- **컨텍스트 구성 파악**: 어떤 턴들이 요약되어 들어가는지
- **실시간 조절**: 사이드바에서 압축률 변경 시 다음 요약부터 적용

## 테스트 케이스

### 단위 테스트

1. `should_summarize()` 함수 테스트
   - Turn 1, 2, 3 → False
   - Turn 4, 7, 10 → True
   - Turn 5, 6, 8, 9 → False

2. Context 구성 테스트
   - Turn 4: 1개 요약 + 1턴 raw
   - Turn 7: 2개 요약 + 1턴 raw
   - Turn 10: 3개 요약 + 1턴 raw

3. 시퀀스 테스트
   - Turn 4 시작 시: `summary_history`가 비어있다가 `summary_node` 후 1개 추가되는지
   - Turn 7 시작 시: 기존 요약 1개 유지 + 새 요약 1개 추가되어 총 2개인지
   - Turn 10 시작 시: 기존 요약 2개 유지 + 새 요약 1개 추가되어 총 3개인지

### 통합 테스트

1. 10턴 연속 대화 시뮬레이션
2. 요약 품질 검증 (정보 손실 여부)
3. 토큰 사용량 측정

## 구현 순서

1. [ ] `ChatState` 수정: `compression_rate` 필드 추가
2. [ ] `component/sidebar.py` 수정: 압축률 슬라이더 추가
3. [ ] `extract_last_n_turns()` 함수 구현: 완료된 턴 단위 메시지 추출
4. [ ] `extract_current_turn()` 함수 구현: 현재 진행 중인 턴 메시지 추출
5. [ ] `_summary_node` 수정: 요약 생성 로직 개선 (압축률 적용, JSON 구조)
6. [ ] `_llm_node` 수정: Context 구성 로직 추가 (요약 + 완료된 raw 턴 + 현재 턴)
7. [ ] `invoke()` 수정: compression_rate 전달
8. [ ] 테스트 작성 및 실행
9. [ ] Streamlit UI: 요약 히스토리 패널 추가 (압축 통계 포함)

## 참고

- LangGraph `add_messages` 리듀서: 메시지 누적 관리
- SqliteSaver: 세션별 상태 저장 (summary_history 포함)
