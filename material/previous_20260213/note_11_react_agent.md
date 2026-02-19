# 노트북 11. LangGraph 상태 그래프와 ReAct 에이전트 설계

> Phase 3 — 실전 기법

노트북 9에서 도구 하나를 호출하는 법을 배웠습니다.
이제 여러 노드를 연결하고, 조건에 따라 분기하는 '에이전트 그래프'를 설계할 차례입니다.

**학습 목표**
- ReAct(Reasoning + Acting) 패턴의 반복 루프 원리를 이해한다
- LangGraph의 StateGraph, 노드, 엣지를 활용하여 에이전트를 구성한다
- llm_node와 tool_node 사이의 조건부 분기를 구현한다
- Checkpointer를 통해 대화 상태를 영속적으로 관리한다

---

## ReAct 패턴이란

**ReAct**는 Reasoning(추론)과 Acting(행동)을 결합한 에이전트 설계 패턴입니다. 모델이 다음 과정을 **충분한 결과가 나올 때까지 반복**합니다.

1. **Reasoning** -- 현재 상황을 분석하고 다음 행동을 추론합니다.
2. **Acting** -- 도구를 호출하여 외부 정보를 가져옵니다.
3. **Observation** -- 도구 결과를 관찰하고, 추가 행동이 필요한지 판단합니다.

예를 들어 "서울 날씨에 맞는 옷을 추천해줘"라는 질문에 대해, 모델은 먼저 날씨 도구를 호출하고, 그 결과를 바탕으로 옷 추천 도구를 다시 호출한 뒤 최종 답변을 생성합니다. 단순 Tool Calling이 한 번의 호출로 끝나는 것과 달리, ReAct는 필요한 만큼 루프를 돌 수 있습니다.

### Tool Calling(1회) vs ReAct(반복 루프) 비교

| 구분 | Tool Calling (노트북 9) | ReAct (노트북 11) |
|------|------------------------|-------------------|
| 도구 호출 | 1회 또는 병렬 1회 | 필요할 때까지 반복 |
| 분기 로직 | 없음 (단순 루프) | 조건부 엣지(Conditional Edge)로 분기 |
| 상태 관리 | messages 리스트 수동 관리 | StateGraph가 자동 관리 |
| 확장성 | 단일 루프 | 노드 추가로 유연하게 확장 |
| 적합한 경우 | 간단한 단일 도구 호출 | 복잡한 다단계 작업 |

---

## LangGraph StateGraph 기초

**LangGraph**는 에이전트를 상태 그래프(State Graph)로 모델링하는 프레임워크입니다. 핵심 구성 요소는 세 가지입니다.

| 구성 요소 | 역할 | 예시 |
|-----------|------|------|
| **State** | 그래프 전체에서 공유하는 데이터 | `{"messages": [...], "turn_count": 0}` |
| **Node** | 상태를 받아 처리하고 변경분을 반환하는 함수 | `llm_node`, `tool_node` |
| **Edge** | 노드 간 연결 (무조건 엣지 또는 조건부 엣지) | `llm -> tools` 또는 `llm -> END` |

```python
from langgraph.graph import StateGraph, MessagesState, START, END

builder = StateGraph(MessagesState)
builder.add_node("llm", llm_func)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)
graph = builder.compile()
```

> 노드 함수는 전체 상태가 아니라 **변경할 부분만** 반환합니다.
> `{"messages": [new_msg]}`를 반환하면 reducer가 기존 리스트에 추가합니다.

### MessagesState와 Reducer

**MessagesState**는 LangGraph가 제공하는 기본 상태 클래스로, `messages` 필드 하나를 가집니다.

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

`add_messages`는 **reducer** 함수입니다. 노드가 반환한 메시지를 기존 리스트에 **추가**(append)하며, 같은 `id`를 가진 메시지를 반환하면 추가 대신 **교체**합니다. 이를 활용하면 이전 메시지를 요약본으로 대체하는 것도 가능합니다.

### State 채널 유형

| 유형 | 정의 방식 | 동작 |
|------|----------|------|
| **Reducer** 채널 | `Annotated[T, func]` | 함수가 이전 값과 새 값을 결합 |
| **Overwrite** 채널 | `T` (일반 타입) | 새 값이 이전 값을 덮어씀 |

`messages`는 reducer 채널이므로 값이 누적되고, `turn_count: int`처럼 일반 타입으로 선언한 필드는 반환 시 덮어쓰기됩니다.

### 커스텀 State 정의

추가 필드가 필요하면 `TypedDict`로 커스텀 상태를 정의합니다.

```python
class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turn_count: int
    user_name: str
```

> `messages` 필드는 반드시 `Annotated[list[AnyMessage], add_messages]`로 정의해야 LangGraph가 메시지를 올바르게 누적합니다.

---

## 조건부 엣지

그래프의 핵심 기능은 **조건에 따라 다른 노드로 분기**하는 것입니다. `add_conditional_edges()`에 라우팅 함수를 전달하면, 해당 함수의 반환 값에 따라 다음 노드가 결정됩니다.

```python
def route_by_length(state: MessagesState):
    last_msg = state["messages"][-1].content
    if len(last_msg) > 20:
        return "long"
    return "short"

builder.add_conditional_edges(START, route_by_length)
```

라우팅 함수는 상태를 받아 **노드 이름 문자열**을 반환합니다. 이 메커니즘이 ReAct 루프의 분기 조건, 라우터 노드의 경로 결정 등 모든 동적 흐름의 기반이 됩니다.

---

## ReAct 에이전트 구현

llm_node와 tool_node를 조건부 엣지로 연결하면 ReAct 루프가 완성됩니다.

```
START -> llm_node -> [tool_calls 존재?] -> Yes -> tool_node -> llm_node -> ...
                                         -> No  -> END
```

```python
from langgraph.prebuilt import ToolNode, tools_condition

builder = StateGraph(MessagesState)
builder.add_node("llm", llm_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")

react_graph = builder.compile()
```

**tools_condition**은 LangGraph가 제공하는 내장 라우팅 함수입니다. 마지막 AIMessage에 `tool_calls`가 있으면 `"tools"` 노드로 분기하고, 없으면 `"__end__"`(END)로 분기합니다. 도구가 바인딩되어 있어도 모델이 도구 없이 답변할 수 있다고 판단하면 tool_calls 없이 직접 텍스트를 반환하여 END로 향합니다.

> `tools_condition`이 ReAct 루프의 핵심입니다. "도구 호출 결과를 확인한 뒤 추가 호출이 필요하면 다시 LLM으로" 라는 반복 구조가 이 조건부 엣지 하나로 구현됩니다.

### create_react_agent: 프리빌트 방식

LangGraph는 위 패턴을 한 줄로 생성하는 `create_react_agent()`도 제공합니다.

```python
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(llm, tools)
```

| 방식 | 장점 | 단점 |
|------|------|------|
| 수동 구성 | 완전한 제어, 커스텀 노드 추가 가능 | 코드가 길어짐 |
| `create_react_agent` | 간결, 빠른 프로토타이핑 | 커스터마이징 제한적 |

동작 원리를 이해한 후 `create_react_agent()`를 사용하는 것이 권장됩니다.

---

## Checkpointer: 대화 상태 영속화

**Checkpointer**를 추가하면 대화 상태(메시지 이력, 도구 호출 결과 등)를 `thread_id` 단위로 저장하고 복원할 수 있습니다.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"messages": [HumanMessage(content="서울 날씨")]}, config=config)
```

같은 `thread_id`를 사용하면 이전 대화 이력을 유지하여 멀티턴 대화가 가능하고, 다른 `thread_id`는 완전히 독립된 대화입니다.

| Checkpointer | 저장 방식 | 적합한 경우 |
|-------------|----------|------------|
| **MemorySaver** | 인메모리 | 개발 및 테스트 |
| **SqliteSaver** | SQLite 파일 | 프로덕션 (프로세스 재시작 후에도 유지) |

> 비동기 환경(FastAPI, Streamlit 등)에서는 `AsyncSqliteSaver`를 사용해야 이벤트 루프 블로킹을 방지할 수 있습니다.

---

## 스트리밍과 디버깅

`graph.stream()`을 사용하면 각 노드가 완료될 때마다 이벤트를 실시간으로 받을 수 있습니다.

- `stream_mode="updates"` -- 각 노드의 상태 변경분만 수신합니다. 어떤 노드가 어떤 메시지를 생성했는지 추적하기에 적합합니다.
- `stream_mode="values"` -- 각 단계의 **전체 상태 스냅샷**을 수신합니다. 그래프가 예상대로 동작하는지 디버깅할 때 유용합니다.

---

## 커스텀 노드로 그래프 확장

ReAct 그래프에 커스텀 노드를 추가하여 기능을 확장할 수 있습니다. 예를 들어 로깅 노드를 llm_node 앞에 배치하거나, 요약 노드로 긴 대화를 압축하거나, 라우터 노드로 질문 유형에 따라 다른 경로로 분기하는 구조가 가능합니다.

```python
builder.add_edge(START, "log")
builder.add_edge("log", "llm")       # 로깅 후 LLM으로
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")
```

### 무한 루프 방지

ReAct 루프에서 모델이 도구를 무한히 호출하는 상황을 방지하려면, ToolMessage 수를 세어 상한에 도달하면 도구 바인딩 없이 LLM을 호출하여 루프를 강제로 종료하는 방법을 사용할 수 있습니다.

---

## 에이전트 설계 패턴

| 패턴 | 구조 | 적합한 경우 |
|------|------|------------|
| 단일 LLM | START -> llm -> END | 간단한 질의응답 |
| ReAct | llm <-> tools (루프) | 도구 활용 에이전트 |
| 라우터 | router -> node_a / node_b -> END | 질문 유형별 분기 |
| 파이프라인 | node_a -> node_b -> node_c -> END | 순차 처리 |
| 하이브리드 | router -> ReAct 또는 단일 LLM | 복합 에이전트 |

> 노드는 **하나의 책임**만 가지도록 설계하고, 상태에는 필요한 정보만 포함합니다.
> 조건부 엣지의 분기 조건은 명확해야 하며, 무한 루프 방지를 위해 도구 호출 횟수에 상한을 두는 것이 좋습니다.

### 자주 하는 실수

| 실수 | 증상 | 해결 |
|------|------|------|
| `add_messages` 누락 | 메시지가 덮어씌워짐 | `Annotated[list, add_messages]` 사용 |
| `bind_tools()` 누락 | LLM이 도구를 호출하지 못함 | `llm.bind_tools(tools)` 필수 |
| Checkpointer 없이 멀티턴 | 이전 대화를 기억하지 못함 | `compile(checkpointer=...)` |
| 노드에서 전체 state 반환 | 불필요한 상태 덮어쓰기 | 변경된 필드만 반환 |

---

## 정리
- **ReAct** 패턴은 "추론 -> 행동 -> 관찰 -> 다시 추론"의 반복 루프로, 단일 Tool Calling과 달리 모델이 필요할 때까지 도구를 반복 호출할 수 있습니다.
- **StateGraph**는 State(공유 데이터), Node(처리 함수), Edge(연결)로 구성되며, `MessagesState`의 `add_messages` reducer가 메시지 누적을 자동으로 관리합니다.
- **tools_condition** 조건부 엣지가 llm_node와 tool_node 사이의 ReAct 루프를 구현하는 핵심이며, tool_calls 유무에 따라 분기합니다.
- **Checkpointer**(MemorySaver, SqliteSaver)를 `compile()`에 전달하면 thread_id 단위로 대화 상태를 저장하고 복원하여 멀티턴 대화를 지원합니다.
- 커스텀 노드와 조건부 엣지를 조합하면 라우터, 로깅, 요약 등 다양한 에이전트 설계 패턴으로 확장할 수 있습니다.
