# 노트북 6. Streaming 응답

> Phase 2 — 제어

사용자는 빈 화면에서 3초 기다리는 것보다, 글자가 하나씩 나오는 것을 훨씬 빠르다고 느낍니다.
이 문서에서는 스트리밍의 동작 원리, 구현 방식, 그리고 핵심 성능 지표인 TTFT를 다룹니다.

**학습 목표**
- 스트리밍과 비스트리밍의 동작 차이를 설명할 수 있다
- google-genai와 LangChain에서 스트리밍을 구현할 수 있다
- 청크(Chunk)의 구조를 분석하고 텍스트를 누적할 수 있다
- TTFT(Time To First Token)의 개념을 이해하고 직접 측정할 수 있다
- LCEL 체인에서 스트리밍을 적용할 수 있다

## 스트리밍이란 무엇인가

LLM API 호출에는 두 가지 방식이 있습니다.

**비스트리밍(Non-streaming)** 방식은 모델이 전체 응답을 생성한 후 한 번에 반환합니다. 응답이 길수록 사용자가 빈 화면을 오래 바라봐야 합니다.

**스트리밍(Streaming)** 방식은 토큰이 생성되는 즉시 **청크(Chunk)** 단위로 반환합니다. 첫 토큰이 도착하는 순간부터 화면에 텍스트가 나타나기 시작합니다.

```
비스트리밍:  요청 ───────[전체 생성 대기]───────> 응답 한 번에 수신
스트리밍:    요청 ──[TTFT]──> 청크1 > 청크2 > ... > 완료
```

총 생성 시간은 두 방식이 거의 동일하지만, 사용자가 체감하는 대기 시간은 스트리밍이 훨씬 짧습니다.

| 지표 | 비스트리밍 | 스트리밍 |
|------|----------|--------|
| 사용자가 첫 글자를 보는 시간 | 전체 생성 완료 후 | TTFT (보통 0.5~2초) |
| 전체 응답 완료 시간 | 거의 동일 | 거의 동일 |
| 네트워크 요청 수 | 1회 | 1회 (SSE 연결 유지) |
| 구현 복잡도 | 낮음 | 약간 높음 (청크 처리 필요) |
| UX 체감 | 느림 | 빠름 |

스트리밍은 HTTP의 **SSE(Server-Sent Events)** 프로토콜을 사용합니다. 서버가 연결을 유지한 채로 데이터를 조각조각 보내주는 방식입니다.

## google-genai 스트리밍

google-genai SDK에서는 `generate_content_stream()` 메서드로 스트리밍을 구현합니다. 일반 호출의 `generate_content()`와 파라미터가 동일하며, 반환값만 이터레이터로 바뀝니다.

```python
# 비스트리밍
response = client.models.generate_content(model=MODEL, contents=prompt)

# 스트리밍 — 메서드 이름만 다름
for chunk in client.models.generate_content_stream(model=MODEL, contents=prompt):
    print(chunk.text, end="")
```

### 청크 구조

스트리밍의 각 청크는 `GenerateContentResponse` 객체와 동일한 구조를 가집니다. 다만 전체 응답의 일부분만 담겨 있습니다.

- `chunk.text`: 이 청크의 텍스트 내용
- `chunk.candidates[0].finish_reason`: 마지막 청크에서만 의미 있는 값 (예: `STOP`, `MAX_TOKENS`)
- `chunk.usage_metadata`: 마지막 청크에서 토큰 사용량 포함

### 텍스트 누적 패턴

스트리밍에서는 각 청크가 전체 응답의 일부이므로, 최종 텍스트를 얻으려면 청크를 이어붙여야 합니다.

```python
full_text = ""
for chunk in client.models.generate_content_stream(model=MODEL, contents=prompt):
    full_text += chunk.text
    print(chunk.text, end="", flush=True)
# full_text에 전체 응답이 누적됨
```

청크의 크기는 균일하지 않습니다. 첫 번째 청크는 작고, 중간 청크는 크며, 마지막 청크는 다시 작아지는 경향이 있습니다. 이는 모델의 토큰 생성 속도와 네트워크 버퍼링에 따라 달라집니다.

> 스트리밍이든 비스트리밍이든 소비되는 토큰 수와 비용은 동일합니다. 스트리밍은 UX 개선 도구이지, 비용 절감 도구가 아닙니다. 토큰 사용량은 반드시 마지막 청크에서 확인해야 합니다.

## LangChain 스트리밍

LangChain의 `ChatGoogleGenerativeAI`는 `.stream()` 메서드로 스트리밍을 지원합니다. 반환되는 각 청크는 **AIMessageChunk** 타입입니다.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 비스트리밍
response = llm.invoke("질문")       # → AIMessage

# 스트리밍
for chunk in llm.stream("질문"):     # → AIMessageChunk
    print(chunk.content, end="")
```

### AIMessageChunk의 주요 필드

- `chunk.content`: 이 청크의 텍스트
- `chunk.id`: 메시지 고유 ID (모든 청크가 동일한 ID 공유)
- `chunk.usage_metadata`: 토큰 사용량 (마지막 청크에서 채워짐)
- `chunk.response_metadata`: 응답 메타데이터 (마지막 청크에서 채워짐)

### 청크 병합 (+ 연산자)

`AIMessageChunk`는 `+` 연산자를 지원합니다. 청크를 하나씩 더하면 최종적으로 완전한 메시지가 됩니다. 이 패턴은 스트리밍 중 실시간 출력을 하면서 동시에 전체 결과도 보존해야 할 때 유용합니다.

```python
full = None
for chunk in llm.stream("질문"):
    full = chunk if full is None else full + chunk
    print(chunk.content, end="")
# full.content → 전체 텍스트, full.usage_metadata → 토큰 사용량
```

### google-genai vs LangChain 스트리밍 비교

| 항목 | google-genai | LangChain |
|------|-------------|----------|
| 메서드 | `generate_content_stream()` | `.stream()` |
| 청크 타입 | `GenerateContentResponse` | `AIMessageChunk` |
| 텍스트 접근 | `chunk.text` | `chunk.content` |
| 토큰 사용량 | `chunk.usage_metadata` (마지막) | `chunk.usage_metadata` (마지막) |
| 청크 병합 | 수동 문자열 누적 (`+=`) | `+` 연산자 지원 |
| 비동기 | `client.aio.models.generate_content_stream()` | `.astream()` |

## TTFT (Time To First Token)

**TTFT(Time To First Token)**는 요청을 보낸 시점부터 첫 번째 토큰(청크)이 도착하는 시점까지의 시간입니다. 스트리밍의 가장 중요한 성능 지표입니다.

TTFT를 측정하는 기본 패턴은 다음과 같습니다.

```python
import time

start = time.time()
for i, chunk in enumerate(client.models.generate_content_stream(
    model=MODEL, contents=prompt
)):
    if i == 0:
        ttft = time.time() - start
total = time.time() - start
print(f"TTFT: {ttft:.3f}초, Total: {total:.3f}초")
```

> 연구에 따르면 사용자는 1초 이내에 응답이 시작되면 "빠르다"고 느끼고, 3초 이상 아무 반응이 없으면 불안해합니다. 스트리밍은 총 응답 시간을 줄이지 않지만, TTFT를 낮춰 체감 속도를 크게 개선합니다. ChatGPT, Gemini 등 모든 주요 챗봇 서비스가 스트리밍을 기본으로 사용하는 이유입니다.

### TTFT에 영향을 주는 요인

- **모델 크기**: 큰 모델일수록 첫 토큰 생성까지 시간이 더 걸립니다
- **프롬프트 길이**: 입력이 길수록 모델이 처리하는 데 시간이 더 걸려 TTFT가 증가합니다
- **서버 부하**: 동시 요청이 많을 때 대기 시간이 길어질 수 있습니다

### TTFT vs Total Time

이 두 지표는 서로 다른 관점을 나타냅니다.

- **TTFT**: 사용자가 첫 글자를 보는 시간 — UX 지표
- **Total Time**: 전체 응답이 완료되는 시간 — 처리량 지표

비스트리밍에서는 사용자가 첫 글자를 보는 시간이 곧 Total Time입니다. 스트리밍에서는 TTFT만 기다리면 화면에 텍스트가 나타나기 시작하므로, 체감 대기 시간이 크게 줄어듭니다.

## LCEL 체인 스트리밍

LangChain의 LCEL 체인(`prompt | model | parser`)에서도 `.stream()` 메서드를 사용할 수 있습니다. 체인의 마지막 단계가 스트리밍을 지원하면 전체 체인이 스트리밍됩니다.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = ChatPromptTemplate.from_messages([
    ("system", "한국어로 간결하게 답변하세요."),
    ("human", "{question}"),
]) | llm | StrOutputParser()

for chunk in chain.stream({"question": "Docker와 Kubernetes의 차이는?"}):
    print(chunk, end="")  # chunk는 str 타입
```

**StrOutputParser**는 `AIMessageChunk`에서 `.content`를 추출하여 문자열 청크로 변환합니다. parser를 사용하면 청크 타입이 `AIMessageChunk`에서 `str`로 바뀌어, 별도의 `.content` 접근 없이 바로 문자열을 사용할 수 있습니다.

## 스트리밍과 생성 파라미터

노트북 5에서 배운 생성 파라미터(`max_output_tokens`, `stop_sequences` 등)는 스트리밍에서도 동일하게 작동합니다. `max_output_tokens`에 도달하면 스트리밍이 중단되고 마지막 청크의 `finish_reason`이 `MAX_TOKENS`가 됩니다. `stop_sequences`에 지정된 문자열이 생성되는 순간에도 스트리밍이 즉시 중단됩니다.

## 비동기 스트리밍

웹 서버(FastAPI, Streamlit 등) 환경에서는 비동기(async) 스트리밍이 유용합니다. 동기 스트리밍은 응답을 기다리는 동안 스레드를 점유하지만, 비동기 스트리밍은 대기 중에도 다른 요청을 동시에 처리할 수 있습니다.

```python
# LangChain 비동기 스트리밍
async for chunk in llm.astream("질문"):
    print(chunk.content, end="")

# google-genai 비동기 스트리밍
async for chunk in client.aio.models.generate_content_stream(
    model=MODEL, contents="질문"
):
    print(chunk.text, end="")
```

| 항목 | 동기 (stream) | 비동기 (astream) |
|------|-------------|----------------|
| 사용 환경 | 스크립트, 노트북 | 웹 서버, 비동기 앱 |
| 스레드 점유 | 응답 중 블로킹 | 대기 중 다른 작업 가능 |
| 코드 패턴 | `for chunk in ...` | `async for chunk in ...` |
| 성능 차이 | 단일 요청에서는 동일 | 동시 요청이 많을 때 이점 |

## 운영 고려사항

스트리밍 호출도 **LangSmith** 트레이스에 정상 기록됩니다. 트레이싱 환경변수만 설정되어 있으면 `.stream()`이든 `.invoke()`든 동일하게 추적할 수 있으므로, 스트리밍 전환 시 모니터링 코드를 별도로 수정할 필요가 없습니다.

프론트엔드와 백엔드를 연결할 때는 **SSE(Server-Sent Events)** 패턴이 일반적입니다. 서버가 HTTP 연결을 유지한 채로 청크를 순차적으로 전송하는 방식으로, 대부분의 챗봇 서비스가 이 패턴을 사용합니다.

---

## 정리

- 스트리밍은 총 응답 시간을 줄이지 않지만, 첫 토큰(TTFT)을 빠르게 전달하여 사용자 체감 속도를 크게 개선합니다
- google-genai는 `generate_content_stream()`, LangChain은 `.stream()`/`.astream()`으로 스트리밍을 지원하며, 청크 타입과 병합 방식에 차이가 있습니다
- 토큰 사용량과 비용은 스트리밍과 비스트리밍이 동일하며, 토큰 정보는 마지막 청크에서 확인합니다
- LCEL 체인에서도 `.stream()`을 호출하면 체인 전체가 스트리밍되며, `StrOutputParser`를 사용하면 문자열 청크를 바로 받을 수 있습니다
- 운영 환경에서는 LangSmith 트레이스 호환성과 SSE 패턴을 고려하여 설계합니다
