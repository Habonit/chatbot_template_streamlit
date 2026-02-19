# 노트북 10. 추론 모델 심화 — Thinking, Budget, 파라미터 전략

> Phase 3 — 실전 기법

모든 질문에 '깊은 생각'이 필요하지는 않습니다. 추론 모델의 작동 원리와 다양한 제어 인자를 이해해야 비용-정확도 최적점을 찾을 수 있습니다.

**학습 목표**
- **추론 모델(Thinking Model)**의 작동 원리와 일반 LLM과의 차이를 이해한다
- `thinking_budget`으로 추론 깊이를 제어하고 질문 유형에 맞는 값을 선택할 수 있다
- thinking token의 과금 구조를 이해하고 비용을 산출할 수 있다
- `include_thoughts`로 모델의 추론 과정을 확인하고 디버깅에 활용할 수 있다

---

## 추론 모델이란

**일반 LLM**은 프롬프트를 받으면 곧바로 답변을 생성합니다. 반면 **추론 모델(Thinking Model)**은 답변 전에 문제를 분해하고, 가설을 세우고, 검증하는 과정을 자동으로 수행합니다. 이 내부 과정을 **Chain-of-Thought(CoT)**라고 합니다.

| 구분 | 일반 LLM | 추론 모델 |
|------|---------|----------|
| 동작 방식 | 프롬프트 → 바로 답변 생성 | 프롬프트 → 내부 추론(CoT) → 답변 생성 |
| 토큰 흐름 | input → output | input → **thinking** → output |
| 강점 | 빠른 응답, 낮은 비용 | 복잡한 문제 해결, 높은 정확도 |
| 약점 | 복잡한 논리에 약함 | 느림, 비용 높음 |

핵심은 토큰 구조의 차이입니다. 추론 모델은 사용자에게 보이지 않는 **thinking tokens**을 추가로 소비하며, 이 토큰에 대해서도 비용이 발생합니다.

> 사람이 "잠깐 생각해봐야" 하는 질문 — 수학 문제, 코드 디버깅, 논리 퍼즐, 다단계 추론 — 에 추론 모델을 사용하고, 인사나 단순 번역 같은 질문에는 비활성화하는 것이 원칙입니다.

---

## Gemini 세대별 추론 제어

| 세대 | 제어 방식 | 값 범위 | 특징 |
|------|----------|---------|------|
| Gemini 2.0 (Flash 등) | 추론 미지원 | - | thinking 파라미터가 무시되거나 에러 발생 |
| Gemini 2.5 (Flash/Pro) | `thinking_budget` | 0 ~ 24576 토큰 | 토큰 단위로 세밀 제어 |
| Gemini 3 계열 | `thinking_level` | low / medium / high | 단계별 간편 제어 |

Gemini 2.5의 강점은 동일 모델에서 `thinking_budget` 값만 바꿔 추론/비추론을 자유롭게 전환할 수 있다는 것입니다. 별도 모델을 배포하거나 API 엔드포인트를 분리할 필요가 없습니다.

```python
# 동일 모델에서 추론 on/off 전환 — config의 thinking_budget 값만 변경
client.models.generate_content(
    model="gemini-2.5-flash", contents=question,
    config={"thinking_config": {"thinking_budget": 0}},     # 추론 비활성화
)
client.models.generate_content(
    model="gemini-2.5-flash", contents=question,
    config={"thinking_config": {"thinking_budget": 4096}},  # 추론 활성화
)
```

---

## thinking_budget 심화

`thinking_budget`은 모델이 추론에 사용할 수 있는 **최대 토큰 수**입니다. 실제 사용량은 이보다 적을 수 있습니다. 모델이 "충분히 추론했다"고 판단하면 일찍 종료하기 때문입니다. budget은 **상한값**일 뿐 **목표값**이 아닙니다.

| 범위 | 용도 | 예시 |
|------|------|------|
| `0` | 추론 비활성화 | 인사, 번역, 단순 요약 |
| `128 ~ 1024` | 가벼운 추론 | 감성 분류, 짧은 논리 판단 |
| `1024 ~ 4096` | 중간 수준 | 일반 분석, 코드 리뷰, 문서 요약 판단 |
| `4096 ~ 8192` | 깊은 추론 | 수학 증명, 복잡한 코드 디버깅 |
| `8192 ~ 24576` | 최대 추론 | 연구 수준의 복잡한 문제 |

### 과잉 추론(Overthinking) 주의

> budget을 높인다고 항상 정확도가 올라가는 것은 아닙니다. 단순한 문제에 높은 budget을 설정하면 모델이 불필요하게 깊이 생각하여 오히려 잘못된 결론에 도달하거나 응답이 지나치게 느려질 수 있습니다.

budget 선택의 경험 법칙은 다음과 같습니다.

1. 정답이 확실한 단순 질문 → `0`
2. "설명해줘" 류의 중간 복잡도 → `1024 ~ 2048`
3. "왜?" "증명해줘" 류의 깊은 분석 → `4096 ~ 8192`
4. 확실하지 않으면 `1024`에서 시작하여 품질을 확인하고 조절

---

## Thinking Token과 과금 구조

추론 모델의 토큰은 3종류로 분리됩니다.

| 토큰 종류 | 설명 | 과금 |
|-----------|------|------|
| **input_tokens** | 프롬프트 (사용자 메시지 + system prompt + 대화 이력) | input 단가 |
| **output_tokens** | 최종 답변 텍스트 | output 단가 |
| **thinking_tokens** | 내부 추론 과정 (사용자에게 보이지 않음) | **output 단가** |

비용 계산 공식은 다음과 같습니다.

```
총 비용 = (input_tokens x input_단가) + (output_tokens x output_단가) + (thinking_tokens x output_단가)
```

| 모델 | input 단가 (/1M tokens) | output 단가 (/1M tokens) |
|------|------------------------|------------------------|
| gemini-2.5-flash | $0.15 | $0.60 |
| gemini-2.5-pro | $1.25 | $10.00 |

> thinking_tokens도 output 단가로 과금되므로, 깊은 추론일수록 비용이 급격히 증가합니다. 특히 `gemini-2.5-pro`에서 높은 budget을 설정하면 thinking 비용이 전체 비용의 상당 부분을 차지할 수 있습니다.

SDK의 `usage_metadata`에서 `prompt_token_count`, `candidates_token_count`, `thoughts_token_count` 세 값을 읽어 위 공식에 대입하면 호출당 비용을 산출할 수 있습니다. 누적 비용 추적기를 구현하면 세션 전체의 thinking 비용 비중을 모니터링할 수 있습니다.

---

## 추론 과정 확인: include_thoughts

`include_thoughts=True`를 설정하면 모델의 내부 추론 과정이 응답에 포함됩니다. 응답의 content가 리스트 형태로 바뀌며, **thought 블록**(추론 과정)과 **response 블록**(최종 답변)으로 구분됩니다.

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="논리 퍼즐 질문...",
    config={"thinking_config": {
        "thinking_budget": 4096,
        "include_thoughts": True,
    }},
)

# content 구조: [thought 블록, response 블록]
for part in response.candidates[0].content.parts:
    is_thought = getattr(part, 'thought', False)
    # is_thought=True → 추론 과정, False → 최종 답변
```

주의할 점은 `include_thoughts`가 응답 포맷만 바꿀 뿐, 추론 자체를 활성화하지는 않는다는 것입니다. `thinking_budget=0`이면 thought 블록이 비어 있습니다. 추론 과정을 확인하려면 반드시 `thinking_budget > 0`과 `include_thoughts=True`를 함께 설정해야 합니다.

---

## Thinking과 다른 파라미터의 관계

Thinking 모드에서는 일부 생성 파라미터에 제약이 있습니다.

| 파라미터 | thinking_budget=0 | thinking_budget > 0 |
|---------|-------------------|--------------------|
| temperature | 자유 설정 (0.0~2.0) | **고정: 1.0** (변경 시 에러) |
| top_p / top_k | 자유 설정 | 일부 제약 가능 |
| response_mime_type | 사용 가능 | 사용 가능 |

**temperature 제약**이 가장 중요합니다. thinking 모드에서 temperature를 0으로 설정하면 에러가 발생합니다. thinking_budget을 명시적으로 설정할 때는 temperature를 생략하거나 1.0으로 지정해야 합니다.

Thinking 모드와 **Structured Output**은 함께 사용할 수 있습니다. 모델이 추론한 후 구조화된 JSON을 반환하므로, "정확한 추론 + 구조화된 결과"를 동시에 얻을 수 있습니다. 코드 분석 보고서, 수학 풀이, 법률 문서 요약 등에 효과적입니다.

---

## LangChain에서 Thinking 사용

LangChain의 `ChatGoogleGenerativeAI`에서는 생성자 파라미터로 `thinking_budget`을 전달합니다.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    thinking_budget=4096,       # 토큰 단위 budget 설정
)
result = llm.invoke("질문...")
# result.usage_metadata에서 토큰 사용량 확인 가능
```

---

## 다른 모델의 추론 접근법 비교

| 항목 | Gemini 2.5 | Gemini 3 | OpenAI o1/o3 | Claude 3.5+ |
|------|-----------|----------|-------------|-------------|
| 제어 방식 | `thinking_budget` (토큰 수) | `thinking_level` (low/mid/high) | `reasoning_effort` (low/medium/high) | `extended_thinking` + `budget_tokens` |
| 제어 정밀도 | 토큰 단위 세밀 제어 | 단계별 간편 제어 | 단계별 제어 | 토큰 단위 제어 |
| 모델 전환 | 동일 모델에서 budget 변경 | 동일 모델에서 level 변경 | 별도 모델 계열 (o1, o3-mini 등) | 별도 활성화 플래그 필요 |
| 비추론 전환 | `budget=0`으로 즉시 전환 | level 설정 변경 | 비추론 모델 별도 사용 | 플래그 비활성화 |

> Gemini의 강점은 동일 모델 내에서 budget 값 하나로 비추론부터 최대 추론까지 전환할 수 있다는 점입니다. 토큰 단위 제어는 OpenAI의 단계별 방식(low/medium/high)보다 비용 최적화에 유리합니다.

---

## 실전 전략: budget 라우터

질문마다 적절한 budget을 자동으로 선택하는 **라우터(Router)** 패턴을 적용하면 비용 효율을 크게 높일 수 있습니다. 단순한 규칙 기반부터 분류 모델 기반까지 다양한 구현이 가능합니다.

```python
def select_budget(question: str) -> int:
    """질문의 복잡도에 따라 thinking_budget을 선택합니다."""
    hard_patterns = ["증명", "디버깅", "풀어줘", "계산", "단계별"]
    medium_patterns = ["설명해줘", "비교해줘", "분석해줘"]
    simple_patterns = ["안녕", "번역", "요약해줘"]

    if any(p in question for p in hard_patterns):
        return 4096
    elif any(p in question for p in medium_patterns):
        return 1024
    elif any(p in question for p in simple_patterns):
        return 0
    return 512  # 기본값
```

프로덕션 환경에서는 분류 모델이나 LLM 기반 라우터로 질문 유형을 판별한 뒤, 유형별로 다른 budget을 적용하는 방식이 효과적입니다. 비용 대비 품질 향상이 미미해지는 지점이 해당 유형의 최적 budget입니다.

---

## 정리

- **추론 모델**은 답변 전에 Chain-of-Thought 과정을 수행하여 정확도를 높이지만, 추가 토큰(thinking tokens)을 소비하여 비용과 시간이 증가합니다
- `thinking_budget`은 상한값이며, 모델이 충분히 추론했다고 판단하면 일찍 종료합니다. 단순한 질문에 높은 budget을 설정하면 과잉 추론이 발생할 수 있으므로 질문 복잡도에 맞는 값을 선택해야 합니다
- thinking tokens은 output 단가로 과금되므로, 비용 구조를 항상 인식하고 모니터링해야 합니다
- `include_thoughts=True`는 추론 과정을 응답에 포함시키는 설정이며, `thinking_budget > 0`과 함께 사용해야 의미가 있습니다
- Gemini는 동일 모델에서 budget 하나로 비추론~최대 추론을 전환할 수 있어, 라우터 패턴과 결합하면 비용-정확도 최적화에 유리합니다
