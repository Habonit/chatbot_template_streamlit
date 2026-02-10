# 노트북 3. Single-turn vs Multi-turn + 대화 저장 전략

> Phase 1 — 기초

LLM에는 '기억'이 없습니다. 매번 전체 대화 내역을 통째로 보내야 합니다. 그 대화를 어디에 저장하느냐가 아키텍처 결정입니다.

**학습 목표**
- Single-turn 호출의 본질적 한계를 이해하고 설명할 수 있다
- Multi-turn 대화의 작동 원리(messages 리스트 누적)를 구현할 수 있다
- google-genai, LangChain, LangGraph 각 프레임워크에서 멀티턴 대화를 구성할 수 있다
- 대화 저장소의 특성과 선택 기준을 판단할 수 있다

---

## Stateless한 LLM API

LLM API 호출은 본질적으로 **Stateless**(무상태)입니다. 매 호출은 독립적이며, 이전 호출의 내용을 전혀 기억하지 못합니다. 이것을 **Single-turn**(단일 턴) 호출이라 합니다.

```python
resp1 = client.models.generate_content(model="gemini-2.5-flash", contents="제 이름은 김철수입니다.")
resp2 = client.models.generate_content(model="gemini-2.5-flash", contents="제 이름이 뭐라고 했죠?")
# resp2: "이름을 알려주시겠어요?" — 두 호출은 완전히 별개이므로 기억하지 못함
```

이 한계는 SDK나 프레임워크에 무관합니다. LangChain의 `model.invoke()`도 동일하게 매번 독립적인 호출입니다. "아까 말한 거"라고 하면 "무엇을 말씀하셨는지 모르겠습니다"가 돌아옵니다.

> 핵심: LLM이 대화를 기억하는 것처럼 보이는 모든 서비스는, 실제로는 클라이언트가 이전 대화를 매번 다시 보내주는 것입니다.

---

## Multi-turn의 작동 원리

**Multi-turn**(멀티턴) 대화의 원리는 단순합니다. 클라이언트에서 **messages 리스트를 누적 관리**하고, 매 호출 시 **전체 대화 이력을 통째로 전송**합니다.

```
턴 1 전송: [user: "이름은 김철수"]
턴 1 응답:                         → [assistant: "반갑습니다"]

턴 2 전송: [user: "이름은 김철수", assistant: "반갑습니다", user: "내 이름이 뭐였죠?"]
턴 2 응답:                         → [assistant: "김철수님이시죠"]
```

모델은 받은 메시지 리스트를 보고 "하나의 긴 대화"로 이해할 뿐입니다. 기억하는 것이 아니라, 우리가 매번 전체 대화를 다시 보내주는 것입니다.

### 토큰 누적 문제

멀티턴에서는 매 호출마다 전체 이력이 입력으로 들어갑니다. 대화가 길어질수록 입력 토큰 수가 선형적으로 증가합니다.

```
턴  1: ~70 토큰
턴  2: ~115 토큰
턴  3: ~160 토큰
...
턴 20: 수천 토큰
```

이 비용 문제를 해결하는 전략(요약, 윈도우 등)은 컨텍스트 관리에서 별도로 다룹니다.

---

## google-genai에서의 멀티턴

google-genai SDK는 두 가지 방법을 제공합니다.

### contents 리스트 수동 관리

`Content` 객체 리스트를 직접 구성하여 `contents` 매개변수에 전달합니다. 각 메시지에 `role="user"` 또는 `role="model"`을 지정합니다.

```python
from google.genai.types import Content, Part

contents = []
contents.append(Content(role="user", parts=[Part(text="제 이름은 이영희입니다.")]))
resp = client.models.generate_content(
    model="gemini-2.5-flash", config={"system_instruction": "..."}, contents=contents,
)
contents.append(Content(role="model", parts=[Part(text=resp.text)]))
# 턴 2에서 contents에 새 user 메시지를 추가하고 다시 호출하면 이전 맥락이 유지됨
```

### client.chats.create() 세션

SDK가 제공하는 **채팅 세션** 객체를 사용하면 대화 이력 관리를 위임할 수 있습니다.

```python
chat = client.chats.create(
    model="gemini-2.5-flash",
    config={"system_instruction": "친절하게 답변하세요."},
)
resp1 = chat.send_message("제 이름은 박민수입니다.")
resp2 = chat.send_message("제 이름이 뭐였죠?")  # 세션이 이전 대화를 자동 포함
```

> `client.chats.create()`는 편리하지만 내부적으로 하는 일은 동일합니다. 대화 이력을 리스트에 쌓고, 매 호출 시 전체를 전송합니다. 프로세스가 종료되면 세션도 사라집니다.

---

## LangChain에서의 멀티턴

### List[BaseMessage] 수동 관리

LangChain의 메시지 타입(`HumanMessage`, `AIMessage`, `SystemMessage`)으로 리스트를 구성하고, `model.invoke()`에 전달하는 가장 기본적인 방법입니다.

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

conversation = [SystemMessage(content="당신은 친절한 도우미입니다.")]
conversation.append(HumanMessage(content="제 이름은 김철수입니다."))
resp = model.invoke(conversation)
conversation.append(resp)  # AI 응답도 이력에 추가
```

### ChatMessageHistory

`InMemoryChatMessageHistory`는 이력 관리 전용 클래스로, `add_message`, `clear` 등 편의 메서드를 제공합니다. 대화 함수를 감싸는 패턴으로 깔끔하게 사용할 수 있습니다.

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

history = InMemoryChatMessageHistory()
history.add_message(SystemMessage(content="당신은 여행 가이드입니다."))

def chat_turn(user_input: str, history) -> str:
    history.add_message(HumanMessage(content=user_input))
    response = model.invoke(history.messages)
    history.add_message(response)
    return response.content
```

---

## LangGraph의 MessagesState와 add_messages

**LangGraph**는 그래프 기반으로 LLM 애플리케이션의 흐름을 정의하는 프레임워크입니다. 멀티턴 대화를 관리하는 세 가지 핵심 개념이 있습니다.

| 개념 | 역할 |
|------|------|
| **MessagesState** | 메시지 리스트를 상태로 관리하는 미리 정의된 타입 |
| **add_messages** | 새 메시지를 기존 이력에 append하는 **Reducer**(리듀서) |
| **Checkpointer** | 상태를 저장하는 백엔드 (MemorySaver, SqliteSaver 등) |

**Reducer**(리듀서)란, 새로운 값이 들어왔을 때 기존 상태와 어떻게 합칠지를 결정하는 함수입니다. `add_messages`는 "새 메시지를 기존 리스트에 append"하는 reducer입니다.

### 기본 구조: StateGraph + Checkpointer

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver

def chat_node(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}  # add_messages reducer가 기존 이력에 append

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chat", chat_node)
graph_builder.add_edge(START, "chat")
graph_builder.add_edge("chat", END)

graph = graph_builder.compile(checkpointer=MemorySaver())
```

### thread_id로 세션 분리

`thread_id`를 다르게 지정하면 별도의 대화 세션으로 분리됩니다. 여러 사용자의 대화를 동시에 독립적으로 관리할 수 있습니다.

```python
config_a = {"configurable": {"thread_id": "user-alice"}}
config_b = {"configurable": {"thread_id": "user-bob"}}

graph.invoke({"messages": [HumanMessage(content="저는 Alice입니다.")]}, config=config_a)
graph.invoke({"messages": [HumanMessage(content="저는 Bob입니다.")]}, config=config_b)
# 각 세션은 서로의 대화를 알지 못함
```

---

## 대화 저장소 선택

지금까지 다룬 모든 InMemory 방식은 프로세스 종료 시 대화 이력이 사라집니다. 실제 서비스에서는 영구 저장이 필요할 수 있으며, 저장소 선택은 다음 기준으로 판단합니다.

| 저장소 | 영속성 | 속도 | 용도 |
|--------|--------|------|------|
| **InMemory** | 프로세스 종료 시 소멸 | 가장 빠름 | 프로토타입, 테스트 |
| **SQLite** | 파일로 영구 저장 | 빠름 | 단일 서버, 소규모 |
| **Redis** | TTL로 자동 만료 가능 | 매우 빠름 | 다중 서버, 세션 관리 |
| **PostgreSQL** | 영구 보존 + 검색/분석 | 보통 | 운영 환경, 대규모 |

> 선택 기준: 대화를 얼마나 오래 보관해야 하는가? 서버가 몇 대인가? 대화 이력을 나중에 분석해야 하는가?

### LangGraph의 Checkpointer 교체

LangGraph는 **Checkpointer만 교체하면** 저장소를 바꿀 수 있습니다. 그래프 코드는 동일하게 유지됩니다.

```python
graph = graph_builder.compile(checkpointer=MemorySaver())        # InMemory — 프로토타입용
graph = graph_builder.compile(checkpointer=SqliteSaver(conn))    # SQLite — 단일 서버 운영용
```

| 구분 | MemorySaver | SqliteSaver |
|------|-------------|-------------|
| 저장 위치 | 메모리 (RAM) | SQLite 파일 |
| 영속성 | 프로세스 종료 시 소멸 | 파일로 영구 저장 |
| 속도 | 매우 빠름 | 빠름 |
| 용도 | 프로토타입, 테스트 | 단일 서버 운영 |
| 설정 복잡도 | 없음 | DB 연결 설정 필요 |

프로토타입에서는 MemorySaver로 시작하고, 운영 환경에서 SqliteSaver나 PostgreSQL로 전환하는 것이 일반적인 패턴입니다.

---

## 프레임워크별 멀티턴 구현 비교

| 프레임워크 | 방식 | 이력 관리 주체 | 세션 분리 | 저장소 교체 |
|-----------|------|---------------|----------|------------|
| google-genai (수동) | contents 리스트 직접 관리 | 개발자 | 직접 구현 | 직접 구현 |
| google-genai (세션) | `client.chats.create()` | SDK 세션 객체 | 세션 객체 단위 | 불가 (InMemory 고정) |
| LangChain (수동) | `List[BaseMessage]` 직접 관리 | 개발자 | 직접 구현 | 직접 구현 |
| LangChain (히스토리) | `InMemoryChatMessageHistory` | 히스토리 객체 | 객체 단위 | 클래스 교체 |
| LangGraph | `MessagesState` + Checkpointer | 프레임워크 (reducer) | `thread_id` | Checkpointer 교체 |

LangGraph로 갈수록 프레임워크가 더 많은 부분을 관리해주며, 특히 세션 분리와 저장소 교체가 설정 수준에서 해결됩니다.

---

## 정리

- LLM API는 **Stateless**합니다. 호출 간에 아무것도 기억하지 않으며, 멀티턴 대화는 클라이언트가 전체 이력을 매번 다시 전송하는 방식으로 구현됩니다.
- 멀티턴의 핵심 패턴은 **messages 리스트 누적**이며, google-genai의 세션이든 LangChain의 히스토리든 내부적으로 하는 일은 동일합니다.
- 대화가 길어질수록 **입력 토큰이 선형 증가**합니다. 이 비용 문제는 컨텍스트 관리 전략(요약, 윈도우 등)으로 해결합니다.
- LangGraph의 **MessagesState + add_messages reducer + Checkpointer** 조합은 세션 분리(`thread_id`)와 저장소 교체를 선언적으로 처리할 수 있어, 프로덕션까지 확장 가능한 패턴입니다.
- 저장소 선택은 **영속성 요구사항과 서버 구조**에 따라 결정합니다. InMemory에서 시작하여 SQLite, Redis, PostgreSQL로 점진적으로 전환하는 것이 일반적입니다.
