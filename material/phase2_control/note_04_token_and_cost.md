# Note 04. 토큰과 컨텍스트 윈도우

> 대응 노트북: `note_04_token_and_cost.ipynb`
> Phase 2 --- 제어: 모델 동작을 다루기

## 학습 목표

- 토큰(token)의 실체가 서브워드(subword) 단위임을 이해하고, `count_tokens()` API로 직접 측정할 수 있다
- 한국어와 영어의 토큰 효율 차이를 수치로 비교하고 그 원인을 설명할 수 있다
- 컨텍스트 윈도우(context window)의 개념과 한계를 이해한다
- `usage_metadata`의 각 필드를 해석하고 비용 구조를 계산할 수 있다
- Thinking 토큰이 전체 비용에 미치는 영향을 파악할 수 있다

---

## 핵심 개념

### 4.1 토큰의 실체

**한 줄 요약**: 토큰(token)은 LLM이 텍스트를 처리하는 최소 단위이며, 단어가 아닌 서브워드(subword) 단위로 분할된다.

LLM은 텍스트를 단어 단위가 아니라 토큰 단위로 처리한다. 토큰은 서브워드(subword) 분할 알고리즘(BPE, SentencePiece 등)에 의해 결정되며, 하나의 단어가 여러 토큰으로 쪼개질 수도 있고, 짧은 단어는 하나의 토큰이 될 수도 있다.

- `"hello"` --- 1토큰
- `"unhappiness"` --- `["un", "happiness"]` --- 2토큰
- `"supercalifragilisticexpialidocious"` --- 11토큰 (긴 단어는 여러 서브워드로 분할)

토큰 수는 단어 수와 일치하지 않는다. 과거 노트북 실행 결과에서 `"Hello"`는 2토큰, 9단어 문장 `"The quick brown fox jumps over the lazy dog."`는 11토큰으로 측정되었다. 토큰 수가 비용, 속도, 컨텍스트 한계를 모두 결정하므로, 토큰의 실체를 이해하는 것이 LLM 활용의 출발점이다.

### 4.2 count_tokens() API

**한 줄 요약**: `client.models.count_tokens()`는 실제 API 호출 없이 텍스트의 토큰 수를 측정하는 무료 API이다.

Gemini API는 `count_tokens()` 엔드포인트를 제공한다. 이 API는 지정된 모델의 토크나이저를 사용하여 입력 텍스트의 토큰 수를 반환하며, 생성(generate) 호출이 아니므로 비용이 발생하지 않는다.

```python
# 토큰 수 측정 (비용 없음)
result = client.models.count_tokens(
    model="gemini-2.5-flash",
    contents="Hello, world!",
)
print(result.total_tokens)  # 5
```

`count_tokens()`는 문자열뿐 아니라 `Content` 객체, 리스트 형태의 멀티턴 대화 이력 등 `generate_content()`와 동일한 형태의 입력을 받을 수 있다. 전달 방식(문자열, Content 객체, 리스트)에 관계없이 동일한 텍스트는 동일한 토큰 수를 반환한다.

### 4.3 한국어와 영어의 토큰 효율

**한 줄 요약**: 한국어는 영어 대비 약 1.0~1.8배의 토큰을 소비하며, 이는 비용과 컨텍스트 사용량에 직접 영향을 준다.

한국어는 유니코드 멀티바이트 문자이며, 자모 조합 구조를 가진다. 이로 인해 영어(라틴 알파벳 기반)보다 같은 의미를 전달하는 데 더 많은 토큰이 필요하다. 노트북 측정 결과에서 동일 의미 문장의 한국어/영어 토큰 비율은 1.0x ~ 1.8x 범위를 보인다.

(예시)

| 영어 | EN 토큰 | 한국어 | KO 토큰 | 배율 | 
|------|---------|--------|---------|------|
| What is artificial intelligence? | 6 | 인공지능이란 무엇인가요? | 10 | 1.7x |
| Python is a popular programming language. | 8 | 파이썬은 인기 있는 프로그래밍 언어입니다. | 14 | 1.8x |
| Machine learning models learn patterns from data. | 9 | 머신러닝 모델은 데이터에서 패턴을 학습합니다. | 14 | 1.6x |

한국어 서비스를 운영할 때는 이 토큰 효율 차이를 고려하여 비용을 산정해야 한다. 같은 대화를 한국어로 수행하면 비용이 더 많이 들고, 컨텍스트 윈도우도 더 빨리 소진된다.

### 4.4 토큰 수 어림 규칙

**한 줄 요약**: API 호출 없이 토큰 수를 대략적으로 추정하는 경험 규칙이 있으나, 텍스트 유형에 따라 오차가 크다.

| 언어 | 어림 규칙 | 근거 |
|------|----------|------|
| 영어 | 1토큰 --- 약 4글자 (또는 0.75단어) | 라틴 알파벳 기반 서브워드 |
| 한국어 | 1토큰 --- 약 1 ~ 2글자 | 유니코드 멀티바이트, 조합형 |
| 코드 | 단어 수와 유사 | 식별자, 기호가 개별 토큰 |

이 규칙은 빠른 어림에 유용하지만 정확하지 않다. 이번 노트북 실험에서 영어는 0 ~ 44%, 한국어는 3 ~ 25%, 코드는 최대 53%의 오차를 보인다. 정확한 측정이 필요한 경우에는 `count_tokens()` API를 사용해야 한다.

텍스트 유형별 토큰 효율도 크게 다르다. 일반 텍스트의 글자/토큰 비율이 1.7 ~ 3.7인 반면, 숫자 나열은 1.0, URL은 2.2, JSON은 2.1로 유형마다 차이가 있다.

### 4.5 count_tokens() 심화 --- system_instruction과 멀티턴

**한 줄 요약**: `system_instruction`은 매 호출마다 토큰으로 소비되며, 멀티턴 대화에서는 전체 이력이 누적되어 토큰이 증가한다.

`system_instruction`(시스템 프롬프트)도 입력 토큰의 일부로 계산된다. 긴 system prompt를 사용하면 매 API 호출마다 그만큼의 토큰이 추가로 소비된다. 이번 노트북 실험에서 짧은 system prompt(12토큰)와 긴 system prompt(86토큰)의 차이는 74토큰이며, 이것이 매 호출마다 반복된다.

멀티턴 대화에서는 매 호출마다 전체 대화 이력이 전송된다. 대화가 진행될수록 입력 토큰이 누적적으로 증가한다.

```python
from google.genai.types import Content, Part

# 멀티턴 대화 이력의 토큰 수 측정
conversation = [
    Content(role="user", parts=[Part(text="안녕하세요")]),
    Content(role="model", parts=[Part(text="안녕하세요!")]),
    Content(role="user", parts=[Part(text="파이썬 공부를 시작하려고 합니다.")]),
]

tokens = client.models.count_tokens(
    model="gemini-2.5-flash",
    contents=conversation,
).total_tokens
```

5턴 대화에서 1턴째 9토큰이 5턴째에는 63토큰으로 증가한다. 이 누적 구조가 멀티턴 대화의 비용을 결정하는 핵심 요인이다.

### 4.6 컨텍스트 윈도우

**한 줄 요약**: 컨텍스트 윈도우(context window)는 모델이 한 번에 처리할 수 있는 최대 토큰 수이며, 입력과 출력의 합이 이 한계를 넘을 수 없다.

컨텍스트 윈도우는 다음 제약을 따른다:

```
입력 토큰 + 출력 토큰 <= 컨텍스트 윈도우
```

Gemini 모델별 컨텍스트 윈도우는 다음과 같다:

| 모델 | 컨텍스트 윈도우 | 최대 출력 토큰 |
|------|----------------|---------------|
| gemini-2.5-flash | 1,048,576 (약 100만) | 65,536 |
| gemini-2.5-pro | 1,048,576 (약 100만) | 65,536 |
| gemini-2.0-flash | 1,048,576 (약 100만) | 8,192 |

100만 토큰은 약 70만 단어, 책 약 10권 분량에 해당한다. 컨텍스트 윈도우가 크다고 항상 유리한 것은 아니다. 긴 컨텍스트에서는 모델의 주의력(attention)이 분산되는 **Lost in the Middle** 현상이 발생할 수 있다. 이는 입력의 중간에 위치한 정보가 앞이나 뒤에 있는 정보보다 덜 잘 인식되는 현상이다.

### 4.7 usage_metadata 상세 분석

**한 줄 요약**: API 응답의 `usage_metadata`는 입력/출력/Thinking 토큰 수를 포함하며, 비용 산정과 사용량 모니터링의 기반이 된다.

`generate_content()` 응답의 `usage_metadata`에는 다음 필드가 포함된다:

| 필드 | 설명 |
|------|------|
| `prompt_token_count` | 입력 토큰 수 (사용자 메시지 + system prompt) |
| `candidates_token_count` | 출력 토큰 수 (모델 응답 텍스트) |
| `total_token_count` | 전체 토큰 수 (입력 + 출력 + Thinking) |
| `thoughts_token_count` | Thinking(추론) 토큰 수 (Thinking 지원 모델에서만) |

`total_token_count`는 단순히 `prompt_token_count + candidates_token_count`가 아닐 수 있다. Thinking이 활성화된 경우 `thoughts_token_count`가 추가로 포함된다. 노트북 실험에서 `prompt(14) + candidates(326) = 340`이지만, `thoughts_token_count(1,023)`이 포함되어 `total_token_count = 1,363`이 되는 것을 확인할 수 있다.

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="파이썬의 장점을 3가지 알려주세요.",
)
usage = response.usage_metadata

# 각 필드 확인
print(usage.prompt_token_count)      # 입력 토큰
print(usage.candidates_token_count)  # 출력 토큰
print(usage.total_token_count)       # 전체 토큰
```

### 4.8 Thinking 토큰과 비용

**한 줄 요약**: Thinking(추론) 기능이 활성화되면 내부 추론 과정에서 추가 토큰이 소비되며, 이 토큰은 출력 단가와 동일하게 과금된다.

Gemini 2.5 모델은 Thinking 기능을 지원한다. Thinking이 활성화되면 모델이 답변 전에 내부적으로 추론하는 과정을 거치며, 이 과정에서 소비되는 토큰이 `thoughts_token_count`로 기록된다.

| 토큰 종류 | 설명 | 과금 기준 |
|----------|------|----------|
| Input tokens | 사용자 입력 + system prompt | 입력 단가 |
| Output tokens | 실제 응답 텍스트 | 출력 단가 |
| Thinking tokens | 내부 추론 과정 | 출력 단가와 동일 |

노트북 실험에서 단순한 질문("7 + 13은 얼마인가요?")에 대해 Thinking OFF일 때 전체 23토큰, Thinking ON(budget=1024)일 때 전체 87토큰으로, Thinking 토큰 64개가 추가되었다. `thinking_budget`를 높게 설정하면 추론 토큰이 크게 증가할 수 있으므로, 단순한 질문에는 Thinking을 비활성화하는 것이 비용 효율적이다.

```python
# Thinking 비활성화
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="7 + 13은 얼마인가요?",
    config={"thinking_config": {"thinking_budget": 0}},
)

# Thinking 활성화
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="7 + 13은 얼마인가요?",
    config={"thinking_config": {"thinking_budget": 1024}},
)
```

### 4.9 비용 구조

**한 줄 요약**: Gemini API 비용은 입력 토큰과 출력 토큰에 대해 별도 단가가 적용되며, 모델과 사용량에 따라 차등 과금된다.

Gemini API의 유료 티어(Paid tier) 기준 가격은 다음과 같다 

- 이미지 참조

비용 산정 시 고려해야 할 사항은 다음과 같다:

1. **system_instruction**: 매 호출마다 입력 토큰에 포함되므로, 긴 system prompt는 누적 비용을 높인다.
2. **멀티턴 대화**: 이전 대화 이력이 매번 전송되므로, 대화가 길어질수록 입력 토큰이 증가한다.
3. **한국어 토큰 효율**: 동일 내용을 한국어로 처리하면 영어보다 1.5~2배 더 많은 토큰이 소비된다.
4. **Thinking 토큰**: 출력 단가로 과금되므로, 불필요한 Thinking은 비용을 증가시킨다.

무료 티어도 제공되며, 요금 한도 내에서 입출력 모두 무료로 사용할 수 있다. Context Caching(컨텍스트 캐싱)을 활용하면 반복되는 큰 입력의 비용을 기본 입력 단가의 10% 수준으로 줄일 수 있다.

---

## 장단점

| 장점 | 단점 |
|------|------|
| `count_tokens()`로 사전에 비용을 정확하게 예측할 수 있다 | 한국어는 영어 대비 토큰 효율이 낮아 비용이 높다 |
| Gemini 모델의 컨텍스트 윈도우가 100만 토큰으로 대부분의 사용 사례를 수용한다 | 긴 컨텍스트에서 Lost in the Middle 현상으로 정보 인식 정확도가 떨어질 수 있다 |
| `usage_metadata`로 호출별 토큰 사용량을 상세히 추적할 수 있다 | 멀티턴 대화에서 이력 누적으로 입력 토큰이 급격히 증가한다 |
| 무료 티어가 제공되어 학습 및 프로토타입에 활용할 수 있다 | Thinking 토큰이 예상 이상으로 비용을 증가시킬 수 있다 |
| Context Caching으로 반복 입력 비용을 절감할 수 있다 | 어림 규칙의 오차가 커서 정확한 비용 산정에는 API 호출이 필요하다 |

---

## 핵심 정리

| 개념 | 핵심 포인트 |
|------|------------|
| 토큰(Token) | LLM의 텍스트 처리 최소 단위. 서브워드 분할 알고리즘으로 결정되며, 단어 수와 일치하지 않는다 |
| count_tokens() | 비용 없이 토큰 수를 사전 측정하는 API. 문자열, Content 객체, 멀티턴 이력 모두 지원 |
| 한국어 토큰 효율 | 영어 대비 약 1.0~1.8배 토큰 소비. 유니코드 멀티바이트 문자 구조가 원인 |
| 어림 규칙 | 영어: 1토큰 --- 4글자, 한국어: 1토큰 --- 1~2글자. 텍스트 유형에 따라 오차가 크다 |
| 컨텍스트 윈도우 | 모델의 최대 처리 토큰 수. Gemini 2.5 모델은 약 100만 토큰 |
| Lost in the Middle | 긴 컨텍스트에서 중간 위치 정보의 인식 정확도가 떨어지는 현상 |
| usage_metadata | prompt_token_count, candidates_token_count, thoughts_token_count, total_token_count로 구성 |
| Thinking 토큰 | 내부 추론 과정에서 소비되는 토큰. 출력 단가로 과금되며, thinking_budget로 제어 |
| 비용 구조 | 입력/출력 별도 과금. system_instruction, 멀티턴 이력, 한국어 효율, Thinking이 비용에 영향 |

---

## 참고 자료

- [Understand and count tokens (Gemini API)](https://ai.google.dev/gemini-api/docs/tokens) --- count_tokens() API 사용법과 토큰 개념 공식 문서
- [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing) --- Gemini 모델별 토큰 단가, 무료 티어, 과금 구조
- [Long context (Gemini API)](https://ai.google.dev/gemini-api/docs/long-context) --- 컨텍스트 윈도우 활용 가이드와 Context Caching
- [Gemini models (Gemini API)](https://ai.google.dev/gemini-api/docs/models) --- 모델별 컨텍스트 윈도우, 최대 출력 토큰 등 사양 정보
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2024)](https://arxiv.org/abs/2307.03172) --- 긴 컨텍스트에서 중간 정보 인식이 저하되는 현상에 대한 논문
- [Counting tokens API reference](https://ai.google.dev/api/tokens) --- count_tokens 엔드포인트의 API 레퍼런스
