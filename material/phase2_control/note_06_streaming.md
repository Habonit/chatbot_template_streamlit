# Note 06. Streaming

> 대응 노트북: `note_06_streaming.ipynb`
> Phase 2 — 제어: 모델 동작을 다루기

---

## 학습 목표

- Streaming(스트리밍)과 Non-streaming(비스트리밍)의 동작 차이를 설명할 수 있다
- google-genai SDK와 LangChain에서 스트리밍을 구현할 수 있다
- Chunk(청크)의 구조를 분석하고 텍스트를 누적할 수 있다
- TTFT(Time To First Token)의 개념을 이해하고 직접 측정할 수 있다
- LCEL 체인에서 스트리밍을 적용하고, 비동기 스트리밍과의 차이를 이해할 수 있다

---

## 핵심 개념

### 6.1 스트리밍과 비스트리밍의 차이

**한 줄 요약**: 비스트리밍은 전체 응답 생성 후 한 번에 반환하고, 스트리밍은 토큰이 생성되는 즉시 청크 단위로 반환한다.

LLM API 호출에는 두 가지 방식이 있다.

**비스트리밍(Non-streaming)** 방식은 모델이 전체 응답을 생성한 후 한 번에 클라이언트로 반환한다. 응답이 길수록 사용자는 빈 화면을 오래 보게 된다.

**스트리밍(Streaming)** 방식은 토큰이 생성되는 즉시 청크(chunk) 단위로 반환한다. 첫 토큰이 나오는 순간부터 화면에 텍스트가 표시되기 시작한다.

```
비스트리밍:
  요청 ─────────────[전체 생성 대기]─────────────> 응답 한 번에 수신

스트리밍:
  요청 ──[TTFT]──> 청크1 > 청크2 > 청크3 > ... > 완료
                   여기서부터 화면에 표시 시작
```

총 생성 시간은 비슷하지만, 사용자가 체감하는 대기 시간은 스트리밍이 훨씬 짧다.

| 지표 | 비스트리밍 | 스트리밍 |
|------|----------|--------|
| 사용자가 첫 글자를 보는 시간 | 전체 생성 완료 후 | TTFT (보통 0.5~2초) |
| 전체 응답 완료 시간 | 거의 동일 | 거의 동일 |
| 네트워크 요청 수 | 1회 | 1회 (SSE 연결 유지) |
| 구현 복잡도 | 낮음 | 약간 높음 (청크 처리 필요) |
| UX 체감 | 느림 | 빠름 |

스트리밍은 HTTP의 **SSE(Server-Sent Events)** 프로토콜을 사용한다. SSE는 서버가 클라이언트와의 연결을 유지한 채로 데이터를 조각(이벤트) 단위로 보내주는 단방향 통신 방식이다. 일반 HTTP 요청-응답과 달리, 하나의 연결에서 서버가 여러 번 데이터를 전송할 수 있다.

Gemini API에서 스트리밍은 REST 엔드포인트 기준으로 `streamGenerateContent`에 해당하며, `?alt=sse` 쿼리 파라미터를 통해 SSE 형식으로 응답을 수신한다.

### 6.2 google-genai 스트리밍: generate_content_stream()

**한 줄 요약**: google-genai SDK에서는 `generate_content_stream()` 메서드로 스트리밍을 수행하며, 반환값은 `GenerateContentResponse`의 이터레이터이다.

`generate_content_stream()`은 비스트리밍의 `generate_content()`와 동일한 파라미터를 받는다. 차이는 반환값이 단일 응답 대신 이터레이터(iterator)라는 점이다.

```python
# 비스트리밍 — 전체 응답 한 번에 반환
response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

# 스트리밍 — 청크 단위로 반환
for chunk in client.models.generate_content_stream(model="gemini-2.5-flash", contents=prompt):
    print(chunk.text, end="")
```

각 청크에는 도착 시점의 타임스탬프를 기록할 수 있다. 이를 통해 청크 간 도착 간격을 확인할 수 있으며, 첫 번째 청크의 도착 시간이 곧 TTFT가 된다.

### 6.3 청크(Chunk) 구조 분석

**한 줄 요약**: 스트리밍의 각 청크는 `GenerateContentResponse`와 동일한 구조를 가지지만, 전체 응답의 일부분만 담겨 있다.

청크의 주요 필드는 다음과 같다.

| 필드 | 설명 | 비고 |
|------|------|------|
| `chunk.text` | 해당 청크의 텍스트 내용 | 편의 접근자 |
| `chunk.candidates[0].content.parts` | Part 리스트 | 내부 구조 접근 |
| `chunk.candidates[0].finish_reason` | 생성 종료 사유 | 마지막 청크에서만 유의미 |
| `chunk.usage_metadata` | 토큰 사용량 | 마지막 청크에서만 포함 |

첫 번째 청크에서는 `finish_reason`이 비어 있고 `usage_metadata`가 없다. 마지막 청크에서 `finish_reason`(예: `STOP`, `MAX_TOKENS`)과 전체 토큰 사용량이 포함된다.

**텍스트 누적 패턴**: 스트리밍에서 최종 텍스트를 얻으려면 각 청크의 텍스트를 이어붙여야 한다.

```python
full_text = ""
for chunk in client.models.generate_content_stream(model=MODEL, contents=prompt):
    full_text += chunk.text
# full_text에 전체 응답이 누적됨
```

**청크 크기의 불균일성**: 청크의 텍스트 길이는 균일하지 않다. 모델의 토큰 생성 속도와 네트워크 버퍼링에 따라 청크 크기가 달라진다. 어떤 청크는 몇 글자만 포함하고, 어떤 청크는 수십 글자를 포함할 수 있다.

### 6.4 스트리밍 토큰 사용량

**한 줄 요약**: 스트리밍에서 토큰 사용량은 마지막 청크의 `usage_metadata`에서만 확인할 수 있으며, 비스트리밍과 소비되는 토큰 수는 동일하다.

스트리밍에서도 비스트리밍과 동일하게 토큰 사용량을 확인할 수 있다. 다만 `usage_metadata`는 마지막 청크에서만 유의미한 값을 포함한다.

```python
last_chunk = None
for chunk in client.models.generate_content_stream(model=MODEL, contents=prompt):
    last_chunk = chunk

usage = last_chunk.usage_metadata
# usage.prompt_token_count — 입력 토큰
# usage.candidates_token_count — 출력 토큰
# usage.total_token_count — 총 토큰
```

스트리밍과 비스트리밍 사이의 토큰 사용 관계는 다음과 같다.

- 입력 토큰: 동일 (같은 프롬프트)
- 출력 토큰: 거의 동일 (생성 결과에 따라 미세한 차이 가능)
- 비용: 동일 (스트리밍은 UX 개선 도구이지, 비용 절감 도구가 아님)
- 토큰 사용량은 반드시 마지막 청크에서 확인해야 한다

### 6.5 LangChain 스트리밍: .stream()

**한 줄 요약**: LangChain의 `ChatGoogleGenerativeAI`는 `.stream()` 메서드로 스트리밍을 지원하며, 각 청크는 `AIMessageChunk` 타입이다.

LangChain에서 스트리밍은 `.stream()` 메서드를 호출하면 된다. 반환되는 각 청크는 `AIMessageChunk`로, 비스트리밍의 `AIMessage`와 유사한 구조이지만 전체 응답의 일부분만 담고 있다.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 비스트리밍
response = llm.invoke("질문")       # AIMessage 반환

# 스트리밍
for chunk in llm.stream("질문"):    # AIMessageChunk 반환
    print(chunk.content, end="")
```

### 6.6 AIMessageChunk 구조 분석

**한 줄 요약**: `AIMessageChunk`는 LangChain의 스트리밍 전용 메시지 타입으로, `content`, `id`, `usage_metadata`, `response_metadata` 필드를 가진다.

`AIMessageChunk`의 주요 필드는 다음과 같다.

| 필드 | 설명 | 비고 |
|------|------|------|
| `chunk.content` | 해당 청크의 텍스트 | `str` |
| `chunk.id` | 메시지 고유 ID | 모든 청크가 동일한 값 |
| `chunk.response_metadata` | 응답 메타데이터 | 마지막 청크에서 채워짐 |
| `chunk.usage_metadata` | 토큰 사용량 | 마지막 청크에서 채워짐 |

### 6.7 청크 병합 (+ 연산자)

**한 줄 요약**: `AIMessageChunk`는 `+` 연산자를 지원하여, 청크를 순차적으로 더하면 전체 응답을 하나의 객체로 합칠 수 있다.

스트리밍 중 실시간으로 화면에 출력하면서, 동시에 전체 결과를 하나의 객체로 보존해야 할 때 이 패턴이 유용하다. 병합된 결과에는 전체 텍스트(`content`)와 토큰 사용량(`usage_metadata`)이 모두 포함된다.

```python
full = None
for chunk in llm.stream("질문"):
    full = chunk if full is None else full + chunk
    print(chunk.content, end="")  # 실시간 출력
# full.content — 전체 텍스트
# full.usage_metadata — 토큰 사용량
```

다음은 google-genai와 LangChain 스트리밍의 주요 차이점이다.

| 항목 | google-genai | LangChain |
|------|-------------|----------|
| 메서드 | `generate_content_stream()` | `.stream()` |
| 청크 타입 | `GenerateContentResponse` | `AIMessageChunk` |
| 텍스트 접근 | `chunk.text` | `chunk.content` |
| 토큰 사용량 | `chunk.usage_metadata` (마지막) | `chunk.usage_metadata` (마지막) |
| 청크 병합 | 수동 문자열 누적 (`+=`) | `+` 연산자 지원 |
| 비동기 | `client.aio.models.generate_content_stream()` | `.astream()` |

### 6.8 TTFT (Time To First Token)

**한 줄 요약**: TTFT는 요청 시점부터 첫 번째 청크가 도착하는 시점까지의 시간으로, 스트리밍의 가장 중요한 성능 지표이다.

TTFT(Time To First Token, 첫 토큰 도달 시간)는 사용자가 "응답이 시작됐다"고 느끼는 시점을 결정하는 지표이다. TTFT에 영향을 미치는 요인은 다음과 같다.

- 모델 크기: 대형 모델일수록 TTFT가 증가한다
- 프롬프트 길이: 입력이 길수록 TTFT가 증가한다 (입력 처리에 더 많은 시간이 소요)
- 서버 부하: 동시 요청이 많을수록 큐 대기 시간이 증가한다
- 일반적인 범위: 0.5~3초

TTFT 측정은 다음과 같이 수행한다.

```python
import time

start = time.time()
for i, chunk in enumerate(client.models.generate_content_stream(
    model=MODEL, contents=prompt
)):
    if i == 0:
        ttft = time.time() - start  # 첫 번째 청크 도착 시간
total = time.time() - start         # 전체 완료 시간
```

UX 관점에서 TTFT의 중요성은 다음과 같다.

- 사용자는 1초 이내 응답이 시작되면 "빠르다"고 느낀다
- 3초 이상 아무 반응이 없으면 불안감을 느낀다
- 스트리밍은 총 응답 시간을 줄이지 않지만, TTFT를 낮춰 체감 속도를 크게 개선한다
- ChatGPT, Gemini 등 주요 챗봇 서비스가 스트리밍을 기본으로 사용하는 이유이다

### 6.9 TTFT vs Total Time

**한 줄 요약**: TTFT는 첫 글자가 보이는 시간(UX 지표)이고, Total Time은 전체 응답 완료 시간(처리량 지표)으로, 서로 다른 목적의 지표이다.

두 지표의 차이를 정리하면 다음과 같다.

| 지표 | 의미 | 용도 |
|------|------|------|
| TTFT | 첫 번째 청크 도착까지의 시간 | UX 체감 속도 |
| Total Time | 전체 응답 생성 완료까지의 시간 | 처리량, 시스템 성능 |

비스트리밍에서는 사용자가 첫 글자를 보는 시간이 Total Time과 동일하다. 스트리밍에서는 사용자가 첫 글자를 보는 시간이 TTFT이며, 이 값이 Total Time보다 훨씬 짧다. 이 차이가 스트리밍의 UX 이점이다.

프롬프트 길이도 TTFT에 영향을 준다. 모델이 입력을 처리하는 데(Prefill 단계) 더 많은 시간이 필요하기 때문에, 긴 프롬프트일수록 TTFT가 증가한다.

### 6.10 LCEL 체인 스트리밍

**한 줄 요약**: LangChain의 LCEL 체인(`prompt | model | parser`)에서도 `.stream()` 메서드를 사용할 수 있으며, 체인의 마지막 단계가 스트리밍을 지원하면 전체 체인이 스트리밍된다.

LCEL(LangChain Expression Language)로 구성된 체인은 자동으로 `.stream()` 및 `.astream()` 메서드를 지원한다. 체인 내 각 단계가 이전 단계의 출력 청크를 받아 처리하며, 파이프라인 전체가 스트리밍된다.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "한국어로 간결하게 답변하세요."),
    ("human", "{question}"),
])

chain = prompt_template | llm | StrOutputParser()

for chunk in chain.stream({"question": "Docker와 Kubernetes의 차이는?"}):
    print(chunk, end="")  # chunk는 str 타입
```

**StrOutputParser의 스트리밍 동작**: `StrOutputParser`는 `AIMessageChunk`에서 `.content`를 추출하여 문자열 청크로 변환한다.

| 구성 | 청크 타입 | 텍스트 접근 |
|------|----------|------------|
| `prompt \| model` | `AIMessageChunk` | `chunk.content` |
| `prompt \| model \| StrOutputParser()` | `str` | `chunk` (바로 문자열) |

### 6.11 스트리밍과 생성 파라미터

**한 줄 요약**: `temperature`, `max_output_tokens`, `stop_sequences` 등 생성 파라미터는 스트리밍에서도 동일하게 작동한다.

Note 05에서 다룬 생성 파라미터는 스트리밍 호출에서도 비스트리밍과 동일하게 적용된다.

**max_output_tokens**: 지정한 토큰 수에 도달하면 스트리밍이 중단되며, 마지막 청크의 `finish_reason`이 `MAX_TOKENS`가 된다.

**stop_sequences**: 지정된 문자열이 생성되는 순간 스트리밍이 즉시 중단된다.

```python
# max_output_tokens — 토큰 수 제한
for chunk in client.models.generate_content_stream(
    model=MODEL,
    contents=prompt,
    config={"max_output_tokens": 30},
):
    print(chunk.text, end="")

# stop_sequences — 특정 문자열에서 중단
for chunk in client.models.generate_content_stream(
    model=MODEL,
    contents=prompt,
    config={"stop_sequences": ["4.", "4)"]},
):
    print(chunk.text, end="")
```

### 6.12 비동기 스트리밍 (astream)

**한 줄 요약**: 비동기 스트리밍은 응답 대기 중 다른 작업을 동시에 처리할 수 있어, 웹 서버 환경에서 유용하다.

동기 스트리밍(`stream`, `for ... in`)은 응답을 기다리는 동안 스레드를 점유한다. 비동기 스트리밍(`astream`, `async for ... in`)은 대기 중 다른 요청을 처리할 수 있으므로, FastAPI나 Streamlit 같은 웹 서버 환경에서 유리하다.

```python
# LangChain 비동기 스트리밍
async for chunk in llm.astream(prompt):
    print(chunk.content, end="")

# google-genai 비동기 스트리밍
async for chunk in await client.aio.models.generate_content_stream(
    model="gemini-2.5-flash", contents=prompt
):
    print(chunk.text, end="")
```

| 항목 | 동기 (stream) | 비동기 (astream) |
|------|-------------|----------------|
| 사용 환경 | 스크립트, 노트북 | 웹 서버, 비동기 앱 |
| 스레드 점유 | 응답 중 스레드 블로킹 | 대기 중 다른 작업 가능 |
| 코드 형태 | `for chunk in ...` | `async for chunk in ...` |
| LangChain 메서드 | `.stream()` | `.astream()` |
| google-genai 메서드 | `generate_content_stream()` | `aio.models.generate_content_stream()` |
| 성능 차이 | 단일 요청에서는 동일 | 동시 요청이 많을 때 이점 |

---

## 장단점

| 장점 | 단점 |
|------|------|
| TTFT 단축으로 사용자 체감 속도 향상 | 청크 처리 로직이 필요하여 구현 복잡도 증가 |
| 사용자가 응답 생성 과정을 실시간으로 확인 가능 | 토큰 사용량은 마지막 청크에서만 확인 가능 |
| SSE 기반으로 단일 HTTP 연결 유지 (효율적) | 네트워크 중단 시 부분 응답만 수신될 수 있음 |
| 비동기 환경에서 동시 처리 가능 (astream) | 스트리밍 자체는 총 응답 시간을 줄이지 않음 |
| 생성 파라미터(temperature, stop 등)가 동일하게 작동 | 청크 크기가 불균일하여 UI 업데이트 주기가 일정하지 않음 |

---

## 핵심 정리

| 개념 | 핵심 포인트 |
|------|------------|
| 스트리밍 vs 비스트리밍 | 총 생성 시간은 동일하지만, 스트리밍은 TTFT를 낮춰 체감 속도를 개선한다 |
| SSE (Server-Sent Events) | 서버가 하나의 HTTP 연결을 유지한 채 데이터를 조각 단위로 전송하는 프로토콜 |
| generate_content_stream() | google-genai SDK의 스트리밍 메서드. `generate_content()`와 동일한 파라미터, 반환값만 이터레이터 |
| 청크 구조 | `GenerateContentResponse`와 동일 구조. `finish_reason`과 `usage_metadata`는 마지막 청크에서만 유의미 |
| 텍스트 누적 | `full_text += chunk.text`로 청크를 이어붙여 전체 응답 구성 |
| 토큰 사용량 | 스트리밍과 비스트리밍의 토큰 소비량과 비용은 동일하다 |
| .stream() | LangChain의 스트리밍 메서드. `AIMessageChunk`를 반환한다 |
| AIMessageChunk | LangChain 스트리밍 전용 메시지 타입. `+` 연산자로 청크를 병합할 수 있다 |
| TTFT | 요청부터 첫 번째 토큰 도착까지의 시간. 프롬프트 길이, 모델 크기, 서버 부하에 영향을 받는다 |
| LCEL 스트리밍 | `prompt \| model \| parser` 체인에서 `.stream()` 호출 시 전체 체인이 스트리밍된다 |
| StrOutputParser | `AIMessageChunk`에서 `.content`를 추출하여 `str` 청크로 변환한다 |
| 비동기 스트리밍 | `.astream()` 또는 `aio.models.generate_content_stream()`으로 비동기 스트리밍을 수행한다 |

---

## 참고 자료

- [Text generation - Gemini API (Google AI for Developers)](https://ai.google.dev/gemini-api/docs/text-generation) — Gemini API 공식 텍스트 생성 가이드. 스트리밍 엔드포인트(`streamGenerateContent`)와 SSE 형식 설명 포함
- [Generating content - Gemini API Reference](https://ai.google.dev/api/generate-content) — `generateContent`와 `streamGenerateContent` REST API 레퍼런스
- [google-genai Python SDK (GitHub)](https://github.com/googleapis/python-genai) — Google Gen AI Python SDK 소스 코드 및 문서. `generate_content_stream()`, `aio.models.generate_content_stream()` 구현 참조
- [Streaming Quickstart - Gemini API Cookbook (GitHub)](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Streaming.ipynb) — Gemini API 공식 쿡북의 스트리밍 퀵스타트 노트북
- [ChatGoogleGenerativeAI - LangChain API Reference](https://python.langchain.com/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html) — LangChain의 `ChatGoogleGenerativeAI` 클래스 레퍼런스. `.stream()`, `.astream()` 메서드 설명 포함
- [How to stream runnables - LangChain](https://python.langchain.com/docs/how_to/streaming/) — LangChain LCEL 체인의 스트리밍 가이드. `stream()`, `astream()`, `astream_events()` 사용법 설명
- [AIMessageChunk - LangChain API Reference](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html) — LangChain의 `AIMessageChunk` 클래스 레퍼런스. 청크 병합(`+` 연산자) 설명 포함
- [Time to First Token (TTFT) in LLM Inference](https://www.emergentmind.com/topics/time-to-first-token-ttft) — TTFT 개념, 측정 방법, 최적화 기법에 대한 종합 설명
