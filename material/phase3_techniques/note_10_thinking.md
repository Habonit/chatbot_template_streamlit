# Note 10. Thinking / 추론 모델

> 대응 노트북: `note_10_thinking.ipynb`
> Phase 3 — 실전: 챗봇을 똑똑하게

## 학습 목표

- 추론 모델(Thinking Model)의 작동 원리와 일반 LLM과의 차이를 이해한다
- `thinking_budget`으로 추론 깊이를 제어할 수 있다
- thinking_tokens의 과금 구조를 이해하고 비용을 계산할 수 있다
- `include_thoughts`로 모델의 내부 추론 과정을 확인할 수 있다
- 질문 유형에 따라 적절한 thinking_budget을 선택할 수 있다

---

## 핵심 개념

### 10.1 추론 모델(Thinking Model)의 원리

**한 줄 요약**: 추론 모델은 답변 생성 전에 내부적으로 Chain-of-Thought(CoT) 과정을 수행하여 복잡한 문제의 정확도를 높인다.

일반 LLM은 프롬프트를 받으면 곧바로 답변을 생성한다. 추론 모델은 답변 전에 문제를 분해하고, 가설을 세우고, 검증하는 과정을 자동으로 수행한다. 이 내부 추론 과정을 Chain-of-Thought(CoT, 사고 사슬)라고 한다.

| 구분 | 일반 LLM | 추론 모델 |
|------|---------|----------|
| 동작 방식 | 프롬프트 -> 바로 답변 생성 | 프롬프트 -> 내부 추론(CoT) -> 답변 생성 |
| 토큰 흐름 | input -> output | input -> thinking -> output |
| 강점 | 빠른 응답, 낮은 비용 | 복잡한 문제 해결, 높은 정확도 |
| 약점 | 복잡한 논리에 약함 | 느림, 비용 높음 |

토큰 흐름의 핵심 차이는 thinking tokens의 존재다. 이 토큰은 사용자에게 보이지 않지만 비용은 발생한다.

```
일반 LLM:     [input tokens] → [output tokens]
추론 모델:    [input tokens] → [thinking tokens] → [output tokens]
```

추론 모델이 필요한 경우와 불필요한 경우는 다음과 같다.

| 추론 필요 | 추론 불필요 |
|-----------|------------|
| 수학 문제 풀이 | 인사, 일상 대화 |
| 코드 버그 분석 | 단순 번역 |
| 논리 퍼즐, 다단계 추론 | 텍스트 요약, 분류 |

Gemini 2.5 모델(Flash/Pro)은 동일 모델에서 `thinking_budget` 설정만으로 추론/비추론을 전환한다. 별도의 추론 전용 모델을 사용할 필요가 없다.

### 10.2 Gemini 세대별 추론 제어

**한 줄 요약**: Gemini 2.5는 `thinking_budget`(토큰 단위), Gemini 3는 `thinking_level`(단계별)로 추론을 제어한다.

| 세대 | 제어 방식 | 값 범위 | 특징 |
|------|----------|---------|------|
| Gemini 2.0 (Flash 등) | 추론 미지원 | - | thinking 파라미터 설정 시 에러 발생 |
| Gemini 2.5 (Flash/Pro) | `thinking_budget` | 0 ~ 24576 토큰 | 토큰 단위로 세밀 제어 |
| Gemini 3 계열 | `thinking_level` | low / medium / high | 단계별 간편 제어 |

지원 모델 확인이 필수다. `gemini-2.0-flash`에 `thinking_budget`을 설정하면 `400 INVALID_ARGUMENT` 에러가 발생한다.

최신 API 문서 기준 모델별 thinking_budget 범위는 다음과 같다.

| 모델 | 기본 동작 | 범위 | 비활성화 |
|------|----------|------|---------|
| Gemini 2.5 Pro | 동적 추론 | 128 ~ 32768 | 비활성화 불가 |
| Gemini 2.5 Flash | 동적 추론 | 0 ~ 24576 | `thinking_budget=0` |
| Gemini 3 계열 | 동적 추론 | `thinking_level` 사용 | `thinking_level="low"` |

`thinking_budget=-1`을 설정하면 동적 추론(Dynamic Thinking)이 활성화되어, 모델이 요청의 복잡도에 따라 자동으로 토큰 사용량을 조절한다.

### 10.3 thinking_budget 심화

**한 줄 요약**: `thinking_budget`은 모델이 추론에 사용할 수 있는 최대 토큰 수이며, 실제 사용량은 이보다 적을 수 있다.

`thinking_budget`은 상한값이지 목표값이 아니다. 모델이 "충분히 추론했다"고 판단하면 budget에 도달하기 전에 추론을 종료한다.

| 범위 | 용도 | 예시 |
|------|------|------|
| `0` | 추론 비활성화 | 인사, 번역, 단순 요약 |
| `128 ~ 1024` | 가벼운 추론 | 감성 분류, 짧은 논리 판단 |
| `1024 ~ 4096` | 중간 수준 | 일반 분석, 코드 리뷰 |
| `4096 ~ 8192` | 깊은 추론 | 수학 증명, 복잡한 코드 디버깅 |
| `8192 ~ 24576` | 최대 추론 | 연구 수준의 복잡한 문제 |

과잉 추론(Overthinking)에 주의해야 한다. budget을 높인다고 항상 정확도가 올라가지 않는다. 단순한 문제에 높은 budget을 설정하면 불필요한 추론으로 응답이 느려지거나, 오히려 잘못된 결론에 도달할 수 있다.

```python
# thinking_budget 값별 비교
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="3의 15승을 단계별로 계산해줘",
    config={"thinking_config": {"thinking_budget": 2048}},
)

usage = response.usage_metadata
# budget=2048이지만 실제 사용량은 다를 수 있음
thinking_tokens = getattr(usage, 'thoughts_token_count', 0) or 0
```

### 10.4 Thinking Token과 과금 구조

**한 줄 요약**: thinking_tokens는 output_tokens와 동일한 단가로 과금되며, 추론이 깊을수록 비용이 증가한다.

추론 모델의 토큰은 3종류로 분리된다.

| 토큰 종류 | 설명 | 과금 |
|-----------|------|------|
| `input_tokens` | 프롬프트 (사용자 메시지 + system prompt + 대화 이력) | input 단가 |
| `output_tokens` | 최종 답변 텍스트 | output 단가 |
| `thinking_tokens` | 내부 추론 과정 (사용자에게 보이지 않음) | **output 단가** |

비용 계산 공식은 다음과 같다.

```
총 비용 = (input_tokens x input_단가) + (output_tokens x output_단가) + (thinking_tokens x output_단가)
```

노트북 실험 결과에서 `budget=0`과 `budget=4096`의 비용 차이가 약 6배에 달했다. thinking_tokens이 전체 비용의 40~50%를 차지하는 경우도 있다.

```python
# 토큰 사용량 확인
usage = response.usage_metadata
input_tokens = usage.prompt_token_count
output_tokens = usage.candidates_token_count
thinking_tokens = getattr(usage, 'thoughts_token_count', 0) or 0

# 비용 계산 (gemini-2.5-flash 기준)
INPUT_PRICE = 0.15 / 1_000_000   # $0.15 / 1M tokens
OUTPUT_PRICE = 0.60 / 1_000_000  # $0.60 / 1M tokens

cost = (input_tokens * INPUT_PRICE +
        (output_tokens + thinking_tokens) * OUTPUT_PRICE)
```

### 10.5 추론 과정 확인: include_thoughts

**한 줄 요약**: `include_thoughts=True`를 설정하면 모델의 내부 추론 과정을 응답에 포함하여 확인할 수 있다.

`include_thoughts`는 응답 포맷만 변경한다. 추론 자체를 활성화하는 것이 아니다. 추론 과정을 확인하려면 `thinking_budget > 0`과 `include_thoughts=True`를 함께 설정해야 한다.

응답의 content가 parts 리스트 형태로 반환되며, 각 part의 `thought` 속성으로 추론 과정과 최종 답변을 구분한다.

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="논리 퍼즐 질문",
    config={"thinking_config": {
        "thinking_budget": 4096,
        "include_thoughts": True,
    }},
)

# thought와 response 분리
parts = response.candidates[0].content.parts
for part in parts:
    if getattr(part, 'thought', False):
        print("[THOUGHT]", part.text)
    else:
        print("[RESPONSE]", part.text)
```

`thinking_budget=0`에서 `include_thoughts=True`를 설정하면 thought 블록은 비어 있다(0개).

### 10.6 Thinking과 다른 파라미터의 관계

**한 줄 요약**: Thinking 모드가 활성화되면 temperature는 1.0으로 고정되며, 일부 생성 파라미터에 제약이 발생한다.

| 파라미터 | thinking_budget=0 | thinking_budget > 0 |
|---------|-------------------|--------------------|
| temperature | 자유 설정 (0.0~2.0) | 고정: 1.0 (변경 불가) |
| top_p | 자유 설정 | 일부 제약 가능 |
| top_k | 자유 설정 | 일부 제약 가능 |
| response_mime_type | 사용 가능 | 사용 가능 |

temperature 제약이 가장 중요하다. thinking 모드에서 temperature를 명시하지 않으면 자동으로 1.0이 적용된다. `response_mime_type`(Structured Output)은 thinking 모드와 함께 사용할 수 있다.

### 10.7 LangChain에서 Thinking 사용

**한 줄 요약**: `ChatGoogleGenerativeAI`의 `thinking_budget` 파라미터로 LangChain에서도 추론 깊이를 제어할 수 있다.

LangChain의 `ChatGoogleGenerativeAI`는 `thinking_budget` 파라미터를 지원한다. 값의 의미는 google-genai SDK와 동일하다.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# 추론 비활성화
llm_no_think = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    thinking_budget=0,
)

# 추론 활성화
llm_think = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    thinking_budget=4096,
)
```

LangChain에서 thinking_tokens 사용량은 `result.usage_metadata`의 `output_token_details.reasoning` 필드에 기록된다. `content`는 기본적으로 문자열(str) 타입으로 반환되며, 추론 과정은 포함되지 않는다.

### 10.8 질문 유형별 최적 budget 전략

**한 줄 요약**: 질문의 복잡도에 따라 thinking_budget을 조절하여 비용-정확도 최적점을 찾는 것이 핵심이다.

| 질문 유형 | 권장 budget | 이유 |
|-----------|------------|------|
| 인사, 일상 대화 | 0 | 추론 불필요 |
| 번역, 단순 요약 | 0 ~ 512 | 패턴 매칭으로 충분 |
| 분류, 감성 분석 | 512 ~ 1024 | 가벼운 판단 |
| 코드 리뷰, 분석 | 1024 ~ 4096 | 논리적 분석 필요 |
| 수학, 논리 퍼즐 | 4096 ~ 8192 | 단계적 추론 필요 |
| 복잡한 디버깅 | 8192+ | 깊은 분석 |

budget 선택 경험 법칙:
1. 정답이 확실한 단순 질문 -> `0`
2. "설명해줘" 류의 중간 복잡도 -> `1024 ~ 2048`
3. "왜?" "증명해줘" 류의 깊은 분석 -> `4096 ~ 8192`
4. 확실하지 않으면 `1024`에서 시작하여 품질을 확인하고 조절

라우터를 사용하여 질문 유형을 먼저 분류하고, 유형에 따라 서로 다른 thinking_budget으로 모델을 호출하는 방식이 효과적이다.

```python
def select_budget(question: str) -> int:
    """질문의 복잡도에 따라 thinking_budget을 선택한다."""
    simple_patterns = ["안녕", "번역", "요약해줘"]
    medium_patterns = ["설명해줘", "비교해줘", "분석해줘"]
    hard_patterns = ["증명", "디버깅", "풀어줘", "계산"]

    q_lower = question.lower()
    if any(p in q_lower for p in hard_patterns):
        return 4096
    elif any(p in q_lower for p in medium_patterns):
        return 1024
    elif any(p in q_lower for p in simple_patterns):
        return 0
    return 512  # 기본값
```

### 10.9 다른 모델의 추론 접근법 비교

**한 줄 요약**: Gemini는 동일 모델에서 토큰 단위로 추론을 세밀하게 제어할 수 있다는 점에서 차별화된다.

| 모델 | 추론 제어 | 특징 |
|------|----------|------|
| Gemini 2.5 | `thinking_budget` (토큰 수) | 세밀한 토큰 단위 제어, 동일 모델에서 전환 |
| Gemini 3 | `thinking_level` (low/mid/high) | 간편한 단계 제어 |
| OpenAI o1/o3 | `reasoning_effort` (low/medium/high) | 별도 모델 계열 (o1, o3-mini 등) |
| Claude 3.5+ | `extended_thinking` + `budget_tokens` | 토큰 단위 제어, 별도 활성화 플래그 필요 |

Gemini의 차별점은 동일 모델에서 `thinking_budget=0`(비추론)과 `thinking_budget=8192`(깊은 추론)을 자유롭게 전환할 수 있다는 것이다. 별도 모델을 배포하거나 API 엔드포인트를 분리할 필요가 없다.

### 10.10 Thinking과 Structured Output

**한 줄 요약**: Thinking 모드와 Structured Output을 함께 사용하면 정확한 추론과 구조화된 결과를 동시에 얻을 수 있다.

모델이 내부적으로 추론한 후 `response_mime_type`과 `response_schema`에 맞는 JSON을 반환한다.

```python
from pydantic import BaseModel, Field

class MathSolution(BaseModel):
    question: str = Field(description="원래 질문")
    answer: int = Field(description="최종 답")
    steps: list[str] = Field(description="풀이 단계")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="연속하는 세 자연수의 합이 75일 때, 세 수를 구해줘",
    config={
        "thinking_config": {"thinking_budget": 2048},
        "response_mime_type": "application/json",
        "response_schema": MathSolution,
    },
)

result = MathSolution.model_validate_json(response.text)
```

활용 시나리오:

| 시나리오 | 조합 효과 |
|---------|----------|
| 코드 리뷰 자동화 | 추론으로 버그 분석 -> JSON 보고서 |
| 수학 교육 앱 | 추론으로 풀이 -> 단계별 JSON |
| 고객 문의 분류 | 추론으로 복합 이슈 분석 -> 카테고리 분류 |

---

## 장단점

| 장점 | 단점 |
|------|------|
| 복잡한 문제(수학, 논리, 코드)에서 정확도 향상 | thinking_tokens에 의한 추가 비용 발생 |
| 동일 모델에서 budget만 변경하여 추론/비추론 전환 | 응답 지연 증가 (thinking 과정 소요 시간) |
| 토큰 단위 세밀 제어로 비용 최적화 가능 | thinking 모드에서 temperature 등 파라미터 제약 |
| include_thoughts로 추론 과정 확인 및 디버깅 가능 | 단순한 질문에 과잉 추론(overthinking) 위험 |
| Structured Output과 함께 사용 가능 | budget을 높여도 반드시 정확도가 향상되지는 않음 |

---

## 핵심 정리

| 개념 | 핵심 포인트 |
|------|------------|
| 추론 모델 | 답변 전 CoT 수행으로 복잡한 문제의 정확도를 높이는 모델 |
| thinking_budget | 추론에 사용할 최대 토큰 수 (상한값). 0이면 비활성화, -1이면 동적 추론 |
| thinking_tokens | 내부 추론에 사용된 토큰. output 단가로 과금됨 |
| include_thoughts | 응답에 추론 과정을 포함하는 옵션. `thinking_budget > 0`과 함께 사용해야 함 |
| temperature 제약 | thinking 활성화 시 temperature는 1.0으로 고정 |
| Gemini 2.5 vs 3 | 2.5는 `thinking_budget`(토큰 단위), 3은 `thinking_level`(단계별) |
| 과잉 추론 | 단순 질문에 높은 budget 설정 시 비용 낭비 및 정확도 저하 가능 |
| budget 라우터 | 질문 유형을 분류하여 적절한 budget을 자동 선택하는 전략 |
| Thinking + Structured Output | 추론으로 정확한 분석, Structured Output으로 구조화된 결과를 동시에 획득 |
| LangChain 지원 | `ChatGoogleGenerativeAI`의 `thinking_budget` 파라미터로 동일하게 제어 |

---

## 참고 자료

- [Gemini thinking - Google AI for Developers](https://ai.google.dev/gemini-api/docs/thinking) — Gemini API 공식 Thinking 모드 가이드. thinking_budget, include_thoughts, 모델별 범위 설명
- [Thought Signatures - Google AI for Developers](https://ai.google.dev/gemini-api/docs/thought-signatures) — 멀티턴 대화에서 추론 컨텍스트를 유지하기 위한 Thought Signature 개념 설명
- [Google Gen AI Python SDK](https://github.com/googleapis/python-genai) — google-genai Python SDK 공식 저장소. ThinkingConfig, GenerateContentConfig 사용법
- [Google Gen AI SDK Documentation](https://googleapis.github.io/python-genai/) — google-genai SDK 공식 API 문서. ThinkingConfig 파라미터 상세 레퍼런스
- [추론 모델에서 추론이 진행되는 방법](https://modulabs.co.kr/blog/reasoning-model-ai) — 추론 모델 (Reasoning Model): AI가 단계별로 생각하는 혁신적 접근법
- [생각하는 AI의 시대, 패러다임의 변화: 추론 모델](https://clova.ai/tech-blog/%EC%83%9D%EA%B0%81%ED%95%98%EB%8A%94-ai%EC%9D%98-%EC%8B%9C%EB%8C%80-%ED%8C%A8%EB%9F%AC%EB%8B%A4%EC%9E%84%EC%9D%98-%EB%B3%80%ED%99%94-%EC%B6%94%EB%A1%A0-%EB%AA%A8%EB%8D%B8) — 생각하는 AI의 시대, 패러다임의 변화: 추론 모델
