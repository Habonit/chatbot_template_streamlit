# 노트북 1. Gemini 직접 호출 vs LangChain 호출 비교

> Phase 1 — 기초

같은 Gemini 모델인데, 부르는 방법이 두 가지입니다. 차이를 알아야 언제 뭘 쓸지 판단할 수 있습니다.

**학습 목표**
- google-genai SDK로 Gemini API를 직접 호출할 수 있다
- LangChain을 통해 동일한 Gemini 모델을 호출할 수 있다
- 두 방식의 반환 객체와 사용 패턴 차이를 설명할 수 있다
- LCEL(LangChain Expression Language) 체인의 기본 구조를 이해한다

## google-genai SDK: 직접 호출

Gemini API를 호출하는 가장 기본적인 방법은 Google이 공식 제공하는 **google-genai** SDK를 사용하는 것입니다. `Client` 객체를 생성한 뒤 `generate_content()`를 호출하는 구조입니다.

> 2025년 11월부터 기존 `google-generativeai` 패키지는 지원 종료(EOL)되었습니다. 현재 공식 SDK는 `google-genai`이며, 인터넷에서 볼 수 있는 구버전(`import google.generativeai as genai`) 코드와는 다릅니다.

google-genai SDK의 기본 호출 구조:

```python
from google import genai

client = genai.Client(api_key="...")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="대한민국의 수도는 어디인가요?",
)
print(response.text)  # "서울입니다." (예시)
```

반환 객체는 **`GenerateContentResponse`** 타입입니다. `.text`로 텍스트를 꺼내고, `.usage_metadata`로 토큰 사용량을 확인할 수 있습니다.

### config 파라미터

`generate_content()`에는 `config` 딕셔너리를 전달하여 모델의 동작을 제어할 수 있습니다. 예를 들어 **`temperature`**(응답의 무작위성)와 **`max_output_tokens`**(출력 길이 제한)를 설정합니다.

`config` 딕셔너리로 생성 파라미터를 제어하는 예시:

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="인공지능의 미래를 한 문장으로 예측해주세요.",
    config={"temperature": 0.0, "max_output_tokens": 100},
)
```

google-genai SDK는 Gemini의 모든 기능(Live API, 이미지 생성 등)에 가장 먼저 접근할 수 있다는 것이 핵심 장점입니다. Google이 제공하는 1차 인터페이스이기 때문입니다.

## LangChain 래핑: 간접 호출

**LangChain**은 LLM 애플리케이션 개발을 위한 프레임워크입니다. 다양한 LLM 제공자(Google, OpenAI, Anthropic 등)를 **통일된 인터페이스**로 다룰 수 있게 해줍니다.

Gemini를 LangChain으로 호출하려면 `langchain-google-genai` 패키지의 **`ChatGoogleGenerativeAI`** 클래스를 사용합니다. 내부적으로 google-genai SDK를 한 번 더 감싼 래퍼(wrapper)입니다.

LangChain으로 Gemini를 호출하는 기본 구조:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = model.invoke("대한민국의 수도는 어디인가요?")
print(response.content)  # "서울입니다." (예시)
```

LangChain은 세 가지 호출 메서드를 제공합니다.

| 메서드 | 동작 | 반환 타입 |
|--------|------|----------|
| `.invoke()` | 단일 입력, 완성된 응답 반환 | `AIMessage` |
| `.stream()` | 토큰이 생성되는 즉시 하나씩 반환 | `AIMessageChunk` (제너레이터) |
| `.batch()` | 여러 입력을 리스트로 전달, 병렬 처리 | `list[AIMessage]` |

> 핵심: 이 세 가지 인터페이스는 모든 LangChain 모델에서 동일합니다. 모델을 교체해도 호출 방식을 바꿀 필요가 없습니다.

### LangChain 메시지 타입

LangChain은 LLM과의 대화를 **메시지 객체**로 관리합니다. `.invoke()`에 문자열을 전달하면 내부적으로 `HumanMessage`로 변환되지만, 메시지 객체를 직접 구성할 수도 있습니다.

| 타입 | 역할 | 설명 |
|------|------|------|
| **`HumanMessage`** | 사용자 | 사용자가 보내는 메시지 |
| **`AIMessage`** | AI | 모델이 생성한 응답 |
| **`SystemMessage`** | 시스템 | 모델의 행동 규칙을 지정 |

메시지 객체를 직접 생성하여 전달하는 예시:

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="당신은 친절한 한국어 도우미입니다."),
    HumanMessage(content="파이썬이란 무엇인가요?"),
]
response = model.invoke(messages)
print(response.content)
```

## 반환 객체 비교: GenerateContentResponse vs AIMessage

두 방식의 가장 눈에 띄는 차이는 반환 객체의 타입입니다. 같은 모델, 같은 질문이므로 담고 있는 정보는 거의 동일하지만, 접근 방식이 다릅니다.

| 구분 | google-genai | LangChain |
|------|-------------|----------|
| 반환 타입 | `GenerateContentResponse` | `AIMessage` |
| 텍스트 접근 | `.text` | `.content` |
| 입력 토큰 수 | `.usage_metadata.prompt_token_count` | `.usage_metadata["input_tokens"]` |
| 출력 토큰 수 | `.usage_metadata.candidates_token_count` | `.usage_metadata["output_tokens"]` |
| 스트리밍 반환 | chunk별 `GenerateContentResponse` | `AIMessageChunk` |
| 모델 정보 | `.model_version` | `.response_metadata["model_name"]` |

`GenerateContentResponse`는 내부에 `candidates` 리스트를 포함하며, 각 candidate의 `content.parts`에서 텍스트를 꺼낼 수 있습니다. `AIMessage`는 `.content`로 바로 접근하고, `.response_metadata` 딕셔너리에 모델 정보와 안전 필터 결과가 포함됩니다.

> 핵심: 같은 모델, 같은 질문이라면 토큰 수는 동일합니다. 차이는 데이터를 꺼내는 속성 이름뿐입니다.

## 언제 무엇을 쓸까?

| 상황 | 권장 방식 | 이유 |
|------|----------|------|
| Gemini 고유 기능 (Live API, 이미지 생성 등) | google-genai | LangChain이 아직 래핑하지 않은 기능 |
| 빠른 프로토타이핑 | google-genai | 의존성 적고 직관적 |
| 체이닝, 메모리, 도구 등 조합 | LangChain | 오케스트레이션 기능 풍부 |
| 모델 교체 가능성 있음 | LangChain | 인터페이스 통일 |
| 프로덕션 에이전트 | LangChain + LangGraph | 상태 관리, 조건 분기 지원 |

> 실무에서는 LangChain/LangGraph를 메인으로 사용하되, 특수 기능이 필요할 때 google-genai를 직접 사용하는 패턴이 일반적입니다.

## LCEL 기초: prompt | model | parser

**LCEL**(LangChain Expression Language)은 LangChain의 핵심 패턴입니다. 여러 컴포넌트를 `|` 연산자(파이프)로 연결하여 하나의 **체인(chain)**을 구성합니다.

가장 기본적인 체인 구조는 세 단계로 이루어집니다.

| 단계 | 컴포넌트 | 역할 |
|------|----------|------|
| 입력 가공 | **`ChatPromptTemplate`** | 변수를 받아 프롬프트를 생성 |
| LLM 호출 | **`ChatModel`** | 프롬프트를 모델에 전달하고 응답 수신 |
| 출력 추출 | **`StrOutputParser`** | AIMessage에서 텍스트(str)만 추출 |

기본 LCEL 체인 구성과 실행:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "{topic}에 대해 한 문장으로 설명해주세요."
)
chain = prompt | model | StrOutputParser()

result = chain.invoke({"topic": "인공지능"})
print(result)       # "인공지능이란..." (str 타입)
```

각 단계가 이전 단계의 출력을 받아 다음 단계의 입력으로 전달합니다. `{"topic": "인공지능"}` 딕셔너리가 prompt를 거쳐 완성된 메시지가 되고, model을 거쳐 AIMessage가 되고, parser를 거쳐 순수 문자열이 됩니다.

### 체인의 유연성

한 번 정의한 체인은 다양한 입력으로 재사용할 수 있으며, 변수를 여러 개 사용하거나 `from_messages()`로 system/human 역할을 구분하는 것도 가능합니다. 체인 역시 `.stream()`과 `.batch()`를 지원합니다.

`from_messages()`로 역할별 메시지를 포함하는 체인 구성:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role}입니다. 핵심만 간결하게 답변하세요."),
    ("human", "{question}"),
])
chain = prompt | model | StrOutputParser()
result = chain.invoke({"role": "요리 전문가", "question": "된장찌개 끓이는 순서"})
```

### StrOutputParser의 역할

`StrOutputParser`는 `AIMessage`에서 `.content`를 추출하여 순수 문자열을 반환합니다. 후속 코드에서 문자열로 바로 사용해야 할 때 편리합니다. 반대로 토큰 수 같은 메타데이터가 필요하면 파서를 빼고 `AIMessage`를 직접 다루면 됩니다.

> 핵심: LCEL 체인(`prompt | model | parser`)은 이후 노트북에서 반복적으로 등장하는 기본 패턴입니다. 여기서 구조를 확실히 이해해두면 이후 학습이 수월합니다.

---

## 정리

- **google-genai**는 Gemini 전용 1차 SDK이고, **LangChain**은 여러 모델을 통일된 인터페이스로 감싸는 래퍼입니다
- 반환 객체(`GenerateContentResponse` vs `AIMessage`)가 다르지만 담고 있는 정보(텍스트, 토큰 수)는 동일합니다
- LangChain의 `.invoke()`, `.stream()`, `.batch()` 인터페이스는 모델을 교체해도 변하지 않습니다
- LCEL 체인(`prompt | model | parser`)은 입력 가공, LLM 호출, 출력 추출을 파이프로 연결하는 핵심 패턴입니다
- 실무에서는 LangChain을 메인으로 쓰되, Gemini 고유 기능이 필요할 때 google-genai를 직접 사용합니다
