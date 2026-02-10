# 노트북 7. 컨텍스트 매니지먼트 전략

> Phase 3 — 실전 기법

대화가 길어지면 토큰은 폭발하고 비용은 치솟고 품질은 떨어집니다. 이걸 관리하는 전략이 챗봇의 실전 품질을 결정합니다.

**학습 목표**
- 멀티턴 대화에서 토큰이 폭발적으로 증가하는 원리를 이해한다
- Sliding Window, Token 기반 트리밍, 요약 압축, 하이브리드 전략을 구현할 수 있다
- LangChain의 `trim_messages()` 유틸리티를 활용할 수 있다
- 전략별 토큰 사용량과 정보 보존율의 트레이드오프를 설명할 수 있다

## 멀티턴 토큰 폭발 문제

LLM은 매 호출 시 **전체 대화 이력**을 입력으로 전송합니다. 대화가 이어질수록 입력 토큰이 선형으로 누적되며, 전체 대화에 걸친 총 비용은 이차(quadratic)로 증가합니다.

```
턴 1: system + user1                              → 입력 ~50 토큰
턴 5: system + user1~5 + ai1~4                     → 입력 ~650 토큰
턴 20: system + user1~20 + ai1~19                  → 입력 ~2,930 토큰
```

20턴 대화 한 건의 총 입력 토큰(1턴부터 20턴까지의 합계)은 수만 토큰에 달합니다. 100명의 사용자가 각각 20턴씩 대화하면 비용이 빠르게 누적됩니다.

### Lost in the Middle 현상

비용만 문제가 아닙니다. 모델은 긴 컨텍스트의 **중간 부분**에 위치한 정보를 잘 활용하지 못하는 경향이 있습니다. 이를 **Lost in the Middle** 현상이라 합니다.

```
[잘 기억] 처음 부분 ... [잘 못 기억] 중간 부분 ... [잘 기억] 마지막 부분
```

대화 초반에 언급한 이름이나 중간에 합의한 조건을 모델이 잊어버리는 상황이 발생할 수 있습니다.

> 컨텍스트 매니지먼트가 해결하는 세 가지 문제:
> - **비용 폭발**: 턴이 늘수록 입력 토큰이 선형 증가, 총 비용은 이차 증가
> - **품질 저하**: Lost in the Middle로 중간 정보 누락 위험
> - **지연 증가**: 입력 토큰이 많을수록 TTFT(첫 토큰 도착 시간)도 증가

## Sliding Window

가장 간단한 전략입니다. **최근 N개의 메시지만 유지**하고 오래된 메시지를 삭제합니다. 시스템 메시지는 모델의 역할과 제약을 유지하기 위해 항상 보존합니다.

```python
def sliding_window(messages, window_size, keep_system=True):
    if keep_system and isinstance(messages[0], SystemMessage):
        system = [messages[0]]
        rest = messages[1:]
    else:
        system = []
        rest = messages
    return system + rest[-window_size:]
```

Window 크기가 작으면 토큰 절약은 크지만 맥락 손실이 커지고, 크기가 크면 맥락 보존은 좋지만 절약 효과가 적습니다. 실무에서는 4~8개 메시지(2~4턴)를 Window로 많이 사용합니다.

## LangChain trim_messages()

LangChain은 수동 구현보다 더 세밀한 제어가 가능한 `trim_messages()` 유틸리티를 제공합니다.

```python
from langchain_core.messages import trim_messages

trimmed = trim_messages(
    messages,
    max_tokens=5,
    token_counter=len,       # 메시지 개수 기준 (len) 또는 실제 토큰 카운터 (llm)
    strategy="last",         # "last" = 최근 유지
    include_system=True,     # 시스템 메시지 항상 보존
    start_on="human",        # 결과가 HumanMessage로 시작하도록 보장
)
```

`start_on="human"`이 중요한 이유는 AIMessage로 시작하는 대화를 모델에 보내면 혼란을 줄 수 있기 때문입니다.

| 파라미터 | 설명 | 기본값 |
|---------|------|-------|
| `max_tokens` | 최대 허용 토큰(또는 메시지) 수 | 필수 |
| `token_counter` | 토큰 계산 방법 (`len`, `llm`, 커스텀 함수) | 필수 |
| `strategy` | `"last"` (최근 유지) 또는 `"first"` (처음 유지) | `"last"` |
| `include_system` | 시스템 메시지 항상 포함 여부 | `False` |
| `start_on` | 결과가 시작해야 할 메시지 타입 | `None` |
| `allow_partial` | 메시지를 잘라서라도 토큰 한도에 맞출지 여부 | `False` |

## Token 기반 트리밍

Sliding Window는 메시지 **개수**로 자르지만, 실제 비용은 **토큰 수**에 비례합니다. 짧은 메시지 10개와 긴 메시지 2개는 토큰 수가 크게 다릅니다.

Token 기반 트리밍은 `token_counter`에 실제 토큰 카운터를 전달하여 정확한 토큰 예산 내에서 메시지를 유지합니다.

```python
# LLM 모델을 token_counter로 사용
trimmed = trim_messages(
    messages,
    max_tokens=100,
    token_counter=llm,      # 실제 토큰 계산
    strategy="last",
    include_system=True,
    start_on="human",
)
```

커스텀 토큰 카운터도 사용할 수 있습니다. google-genai의 `count_tokens` API를 활용하면 됩니다.

> 토큰 예산 설정 가이드:
> 토큰 예산 = 컨텍스트 윈도우 - 예상 출력 토큰 - 안전 마진.
> 실무에서는 500~2,000 토큰 수준의 작은 예산을 설정하여 비용을 제어합니다.
> 예산이 너무 작으면 맥락을 잃고, 너무 크면 비용이 증가합니다.

## 요약 기반 압축

Sliding Window와 Token 트리밍은 오래된 메시지를 **삭제**합니다. 요약 기반 압축은 오래된 대화를 LLM으로 **요약**하여 핵심 정보를 보존합니다.

**전략 흐름**: 대화가 임계값을 초과하면 오래된 메시지를 LLM으로 요약하고, 요약본을 시스템 프롬프트에 포함한 뒤 최근 메시지만 유지합니다.

```python
def compress_with_summary(messages, llm, keep_recent=4):
    system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
    non_system = messages[1:] if system_msg else messages

    if len(non_system) <= keep_recent:
        return messages

    old_messages = non_system[:-keep_recent]
    recent_messages = non_system[-keep_recent:]

    summary = summarize_conversation(old_messages, llm)
    base = system_msg.content if system_msg else "당신은 AI 비서입니다."
    new_system = SystemMessage(f"{base}\n\n[이전 대화 요약]\n{summary}")
    return [new_system] + recent_messages
```

### 요약 프롬프트 설계

요약의 품질은 프롬프트에 크게 의존합니다. **보존해야 할 정보의 종류**를 명시하면 품질이 개선됩니다.

- **고객 상담**: 이름, 주문번호, 문의 내용, 해결 상태
- **교육**: 학생 이름, 학습 주제, 이해도, 다음 단계
- **기술 지원**: 에러 내용, 시도한 해결 방법, 환경 정보

> 요약 전략 주의사항:
> - 요약 자체에 LLM 호출 비용이 발생합니다
> - 요약 과정에서 세부 수치나 코드 조각이 손실될 수 있습니다
> - 대화가 짧을 때는 요약 비용이 절약 비용보다 클 수 있으므로 임계값을 설정해야 합니다

## 하이브리드 전략

실전에서 가장 많이 사용되는 전략입니다. **요약 + Sliding Window**를 결합하여 각각의 장점을 취합니다.

```
대화 길이 <= 임계값  → 그대로 전달 (트리밍 불필요)
대화 길이 > 임계값   → 오래된 부분 요약 + 최근 N턴 유지
```

| 임계값 | keep_recent | 적합한 상황 |
|-------|------------|----------|
| 6~8 메시지 | 4 | 일반 챗봇 (빠른 응답 중요) |
| 12~16 메시지 | 6~8 | 상담/교육 (맥락 중요) |
| 500~1000 토큰 | 300 토큰 | 토큰 기반 정밀 제어 |

메시지 수 기준 분기가 간편하지만, 토큰 수 기준으로 분기하는 것이 더 정밀합니다. 또한 이전 요약 위에 새 요약을 누적하는 **점진적 요약(incremental summarization)** 패턴도 가능합니다.

## 전략별 비교

| 전략 | 토큰 절약 | 정보 보존 | 구현 복잡도 | 추가 비용 |
|------|---------|---------|----------|--------|
| Sliding Window | 높음 | 낮음 (초기 정보 손실) | 매우 낮음 | 없음 |
| Token 트리밍 | 높음 (정밀) | 낮음 | 낮음 | 없음 |
| 요약 압축 | 중간 | 높음 (핵심 보존) | 중간 | 요약 LLM 호출 |
| 하이브리드 | 중간~높음 | 높음 | 중간 | 조건부 LLM 호출 |

동일한 대화에 각 전략을 적용하고 "사용자 이름"과 "이전에 안내한 세부 정보"를 기억하는지 테스트하면, Sliding Window와 Token 트리밍은 초기 정보를 잃는 반면, 요약 기반 전략은 핵심 정보를 보존하는 경향을 보입니다.

> 전략 선택 가이드:
> - **빠른 응답, 비용 절감이 최우선** -- Sliding Window 또는 Token 트리밍
> - **맥락 보존이 중요** (상담, 교육) -- 요약 압축 또는 하이브리드
> - **실전 서비스** -- 하이브리드 (짧을 때 패스, 길 때 요약)

## LangGraph에서의 컨텍스트 관리

LangGraph의 `StateGraph`에서는 그래프 노드 안에서 컨텍스트 관리를 적용합니다. **MessagesState**에 저장된 메시지 이력을 LLM에 전달하기 전에 트리밍하는 것이 핵심 패턴입니다.

### 노드 내부 트리밍

상태에는 전체 이력이 유지되면서 LLM에는 트리밍된 메시지만 전달합니다. 이렇게 하면 `MemorySaver`나 `SqliteSaver`를 통해 대화 이력 자체는 보존하면서 비용은 제어할 수 있습니다.

```python
from langgraph.graph import StateGraph, START, END, MessagesState

def chatbot_node(state: MessagesState):
    trimmed = trim_messages(
        state["messages"],
        max_tokens=300,
        token_counter=llm,
        strategy="last",
        include_system=True,
        start_on="human",
    )
    response = llm.invoke(trimmed)
    return {"messages": [response]}
```

### 요약 노드 패턴

더 고급 패턴으로, 조건부 요약 노드를 그래프에 추가할 수 있습니다. `ChatState`에 `summary` 필드를 추가하고, 메시지 수가 임계값을 초과하면 요약 노드로 분기합니다.

```python
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str

def should_summarize(state: ChatState):
    if len(state["messages"]) > SUMMARY_THRESHOLD:
        return "summarize"
    return "respond"
```

요약 노드는 오래된 메시지를 요약하여 `summary` 필드에 저장하고, 응답 노드는 요약이 있으면 시스템 프롬프트에 포함하여 LLM을 호출합니다. 요약 타이밍과 전략을 유연하게 제어할 수 있다는 점이 이 패턴의 강점입니다.

---

## 정리

- 멀티턴 대화에서 매 호출마다 전체 이력을 전송하므로, 비용은 이차적으로 증가하고 Lost in the Middle 현상으로 품질도 저하됩니다
- Sliding Window는 구현이 간단하고 토큰 절약이 크지만 초기 맥락을 잃으며, Token 기반 트리밍은 이를 토큰 단위로 정밀하게 제어합니다
- 요약 기반 압축은 추가 LLM 호출 비용이 들지만 핵심 정보를 보존하며, 하이브리드 전략이 실전에서 가장 많이 사용됩니다
- LangChain의 `trim_messages()`는 전략, 시스템 메시지 보존, 시작 메시지 타입 보장 등 세밀한 제어를 지원합니다
- LangGraph에서는 상태에 전체 이력을 유지하면서 노드 내부에서만 트리밍을 적용하는 패턴이 일반적입니다
