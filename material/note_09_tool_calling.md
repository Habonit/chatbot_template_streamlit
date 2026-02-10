# 노트북 9. Tool Calling + LangChain/LangGraph 연계

> Phase 3 — 실전 기법

LLM은 학습 데이터에 없는 실시간 정보를 알지 못하고, 정확한 계산도 못 하며, 외부 API도 호출할 수 없습니다. **Tool Calling**은 이러한 한계를 극복하기 위해 LLM에게 "손과 발"을 달아주는 메커니즘입니다.

**학습 목표**
- Tool Calling의 본질(의도 출력 → 클라이언트 실행 → 결과 주입 → 최종 답변)을 이해한다
- google-genai의 수동/자동 Function Calling을 구현할 수 있다
- LangChain의 `@tool`, `bind_tools()`, `ToolMessage` 패턴을 활용할 수 있다
- LangGraph `ToolNode`로 자동화된 도구 실행 그래프를 구성할 수 있다

## Tool Calling의 본질

LLM은 도구를 **직접 실행하지 않습니다**. 모델은 "이 함수를 이 인자로 호출해달라"는 **의도(intent)**를 JSON 형태로 출력할 뿐이며, 실제 실행은 클라이언트(개발자 코드)가 담당합니다. 이 구조를 이해하는 것이 Tool Calling의 핵심입니다.

### 4단계 루프

```
1. 사용자 질문 + 도구 목록 → 모델에 전달
2. 모델이 function_call(name, args)을 JSON으로 출력 (의도만 표현)
3. 클라이언트가 실제 함수를 실행 → 결과를 모델에 다시 전달
4. 모델이 함수 결과를 바탕으로 최종 자연어 답변 생성
```

> 모델이 도구를 직접 실행하지 않는 설계는 보안(위험한 함수 실행 전 검증 가능), 유연성(실행 환경과 모델의 분리), 비용(불필요한 호출 차단) 측면에서 큰 이점이 있습니다.

### Tool Calling이 해결하는 문제

| 상황 | Tool Calling 없이 | Tool Calling 있으면 |
|------|------------------|-------------------|
| "서울 날씨 알려줘" | 학습 데이터 기반 추측 | `get_weather("서울")` → 실시간 데이터 |
| "127 x 389는?" | 근사값 (오류 가능) | `calculator("127*389")` → 49403 (정확) |
| "이메일 보내줘" | "보낼 수 없습니다" | `send_email(...)` → 실제 발송 |

## google-genai: FunctionDeclaration과 Function Calling

google-genai SDK에서는 **FunctionDeclaration**으로 도구의 이름, 설명, 파라미터를 JSON Schema 형식으로 정의합니다.

```python
get_weather_func = types.FunctionDeclaration(
    name="get_weather",
    description="지정된 도시의 현재 날씨를 조회합니다",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "도시 이름"}},
        "required": ["city"],
    },
)
tools = types.Tool(function_declarations=[get_weather_func])
```

### 수동 Function Calling

수동 방식은 4단계 루프를 개발자가 직접 제어합니다. 모델 응답에서 `function_call`을 파싱하고, 함수를 실행한 뒤, `FunctionResponse`로 결과를 되돌려 최종 답변을 요청합니다.

```python
fc = response.candidates[0].content.parts[0].function_call  # function_call 파싱
result = available_functions[fc.name](**dict(fc.args))       # 실제 함수 실행

function_response = types.Part.from_function_response(       # FunctionResponse로 결과 전달
    name=fc.name, response=result,
)
```

수동 방식은 코드가 길어지지만, 함수 실행 전 인자 검증이나 로깅, 사용자 확인 등 세밀한 제어가 가능합니다. 결제 API처럼 실행 전 확인이 필요한 경우에 적합합니다.

### 자동 Function Calling

파이썬 함수를 `tools` 매개변수에 직접 전달하면 SDK가 스키마 추출, `function_call` 파싱, 함수 실행, 결과 주입을 모두 자동으로 처리합니다. 함수의 **이름**, **docstring**, **타입 힌트**가 FunctionDeclaration으로 자동 변환됩니다.

```python
def get_weather_auto(city: str) -> dict:
    """지정된 도시의 현재 날씨 정보를 조회합니다.
    Args:
        city: 도시 이름 (예: 서울, 부산, 제주)
    """
    return weather_data.get(city, {})

response = client.models.generate_content(
    model=MODEL, contents="부산 날씨 알려줘",
    config=types.GenerateContentConfig(tools=[get_weather_auto]),
)  # 자동으로 함수 실행 → 결과 주입 → 최종 답변까지 완료
```

> 자동 Function Calling은 빠른 프로토타이핑에 적합하며, 최대 10회 루프로 무한 반복을 방지합니다. 여러 함수를 리스트로 전달하면 다중 도구 자동 호출도 가능합니다.

## LangChain: @tool과 bind_tools 패턴

### @tool 데코레이터

LangChain에서는 `@tool` 데코레이터로 도구를 정의합니다. **함수 이름**이 도구의 `name`, **docstring 첫 줄**이 `description`, **타입 힌트**가 파라미터 스키마로 자동 변환됩니다.

```python
from langchain_core.tools import tool

@tool
def get_weather_lc(city: str) -> str:
    """지정된 도시의 현재 날씨 정보를 조회합니다."""
    return json.dumps({"temp": 15, "condition": "맑음"})
```

복잡한 도구의 경우 Pydantic **BaseModel**을 `args_schema`로 지정하면, `Field(description=...)`을 통해 각 파라미터에 상세한 설명과 기본값, 제약 조건을 추가할 수 있습니다.

### bind_tools + ToolMessage 루프

`bind_tools()`로 모델에 도구를 바인딩하면, 모델은 도구 호출 의도를 `AIMessage.tool_calls`에 담아 반환합니다. 개발자는 이를 파싱하여 함수를 실행하고 결과를 **ToolMessage**로 되돌립니다.

```python
llm_with_tools = llm.bind_tools([get_weather_lc, calculator_lc])
ai_message = llm_with_tools.invoke("제주 날씨 알려줘")
# ai_message.tool_calls → [{"name": "get_weather_lc", "args": {"city": "제주"}, "id": "call_abc"}]

for tc in ai_message.tool_calls:
    result = tool_map[tc["name"]].invoke(tc["args"])
    tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
```

> `ToolMessage`의 `tool_call_id`는 반드시 해당 `tool_call`의 `id`와 일치해야 합니다. 불일치하면 모델이 결과를 올바르게 매칭하지 못합니다.

## LangGraph: ToolNode와 조건부 엣지

### ToolNode

LangChain의 수동 루프를 자동화한 것이 LangGraph의 **ToolNode**입니다. `AIMessage.tool_calls`에서 도구 이름과 인자를 파싱하고, 실행하고, 결과를 `ToolMessage`로 감싸는 과정을 모두 자동 처리합니다.

### 도구 실행 그래프

`ToolNode`를 `StateGraph`에 통합하면, 조건부 엣지(**tools_condition**)를 통해 도구 호출 여부에 따라 자동으로 분기하는 완전한 루프를 구성할 수 있습니다.

```python
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

builder = StateGraph(MessagesState)
builder.add_node("llm", llm_node)
builder.add_node("tools", ToolNode([get_weather_lc, calculator_lc]))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)  # tool_calls 있으면 → tools, 없으면 → END
builder.add_edge("tools", "llm")                       # 도구 실행 후 다시 LLM으로
graph = builder.compile()
```

그래프의 메시지 흐름은 다음과 같습니다:

```
HumanMessage → llm_node → AIMessage(tool_calls) → tools_condition → tool_node
→ ToolMessage(결과) → llm_node → AIMessage(최종 답변) → tools_condition → END
```

도구가 바인딩되어 있어도 모델이 도구 없이 답할 수 있다고 판단하면 `tool_calls` 없이 직접 텍스트를 반환합니다. 이 판단은 도구의 `description`과 사용자 질문의 의미적 유사도에 기반합니다.

### Checkpointer와 도구 호출 상태

LangGraph의 **checkpointer**(예: `MemorySaver`)를 추가하면, 도구 호출 이력을 포함한 전체 대화 상태가 유지됩니다. 이를 통해 "아까 검색한 결과에서..."와 같은 후속 질문에도 정확하게 답할 수 있습니다.

## 다중 도구와 병렬 호출

모델은 한 번의 응답에서 **여러 도구를 동시에** 호출할 수 있습니다. 예를 들어 "서울과 부산 날씨를 비교해줘"라는 질문에 `get_weather("서울")`과 `get_weather("부산")`을 동시에 요청합니다. `ToolNode`는 병렬로 도착한 `tool_calls`를 모두 실행한 뒤 결과를 한 번에 반환하므로, 네트워크 요청이 필요한 도구의 경우 응답 시간이 단축됩니다.

> 단, 도구 간 의존성(A의 결과가 B의 입력)이 있으면 모델이 순차적으로 호출해야 합니다. 이 경우 루프가 여러 번 반복됩니다.

## 도구 설명(description)의 중요성

모델이 어떤 도구를 선택할지는 전적으로 **description**에 달려 있습니다. 설명이 부정확하면 모델은 잘못된 도구를 선택하거나, 도구를 전혀 사용하지 않습니다.

**좋은 description 작성 원칙**:
- **무엇을 하는지** 명확히 서술: "날씨를 조회합니다" (O) vs "도구입니다" (X)
- **입력 조건**을 명시: "한국 도시만 지원" → 모델이 범위를 판단
- **반환값**을 설명: "온도, 상태, 습도를 포함하는 JSON" → 모델이 후처리를 계획
- **제한사항**을 명시: "최대 10개 결과" → 모델이 기대치를 조절

## 구현 방식 비교

| 항목 | google-genai (수동) | google-genai (자동) | LangChain | LangGraph |
|------|-------------------|-------------------|-----------|----------|
| 도구 정의 | FunctionDeclaration | 파이썬 함수 | @tool | @tool |
| 루프 관리 | 직접 구현 | SDK 자동 | 직접 구현 | 그래프 자동 |
| 상태 관리 | contents 리스트 | SDK 내부 | messages 리스트 | MessagesState |
| 병렬 호출 | 직접 파싱 | 자동 | 직접 파싱 | ToolNode 자동 |
| 적합한 경우 | 원리 학습, 세밀한 제어 | 빠른 프로토타입 | 커스텀 로직 | 프로덕션 에이전트 |

---

## 정리

- Tool Calling은 LLM이 함수를 직접 실행하는 것이 아니라, 호출 **의도(intent)**를 JSON으로 출력하고 클라이언트가 실행하여 결과를 되돌리는 4단계 루프이다
- google-genai의 수동 방식은 각 단계를 직접 제어하므로 인자 검증과 로깅에 유리하고, 자동 방식은 함수를 직접 전달하여 빠르게 프로토타이핑할 수 있다
- LangChain의 `@tool`, `bind_tools()`, `ToolMessage` 패턴은 도구 정의와 실행을 표준화하며, `tool_call_id` 매칭이 핵심이다
- LangGraph의 `ToolNode`와 `tools_condition`은 도구 실행 루프를 그래프로 자동화하여 에이전트 패턴의 기초가 된다
- 도구의 `description` 품질이 모델의 도구 선택 정확도를 좌우하므로, 기능, 입력 조건, 반환값, 제한사항을 명확히 서술해야 한다
