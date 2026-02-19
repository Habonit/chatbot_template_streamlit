# 노트북 5. 생성 파라미터

> Phase 2 — 제어

같은 모델, 같은 프롬프트인데 결과가 매번 다릅니다. LLM이 다음 토큰을 확률적으로 선택하기 때문이며, 이 확률 분포를 조절하는 도구가 **생성 파라미터(Generation Parameters)**입니다. 용도에 맞게 제어할 수 있어야 합니다.

**학습 목표**
- Temperature가 토큰 선택 확률 분포에 미치는 영향을 이해한다
- Top-p, Top-k의 역할과 Temperature와의 상호작용을 설명할 수 있다
- max_output_tokens, stop_sequences, seed를 활용하여 출력을 제어한다
- 용도별 최적 파라미터 조합을 도출할 수 있다

## 토큰 선택의 원리

LLM은 다음 토큰을 확률 분포에서 **샘플링(sampling)**하여 선택합니다. 예를 들어, "오늘 날씨가" 다음에 올 수 있는 토큰은 여러 개이며, 모델은 각 토큰의 확률에 따라 하나를 뽑습니다. 이 확률 분포를 어떻게 조절하느냐에 따라 출력의 성격이 달라집니다.

| 파라미터 | 역할 | 값 범위 |
|---------|------|--------|
| temperature | 확률 분포의 뾰족함 조절 | 0.0 ~ 2.0 |
| top_p | 누적 확률 기반 후보 필터링 | 0.0 ~ 1.0 |
| top_k | 상위 k개 후보만 남김 | 1 ~ 100+ |
| max_output_tokens | 출력 길이 상한 | 1 ~ 65,536 |
| stop_sequences | 특정 문자열에서 생성 중단 | 문자열 리스트 |
| seed | 재현성을 위한 난수 시드 | 정수 |

## Temperature

**Temperature**(온도)는 토큰 선택 확률 분포의 뾰족함을 조절하는, 가장 자주 사용하는 파라미터입니다.

- `temperature = 0`: 가장 확률이 높은 토큰만 선택하여 거의 결정적인 출력을 생성합니다.
- `temperature = 1.0`: 학습된 원래 확률 분포 그대로 샘플링합니다.
- `temperature > 1.0`: 분포가 평탄해져서 낮은 확률의 토큰도 선택될 수 있습니다.

> Temperature는 "창의성 다이얼"이라고 생각하면 됩니다. 낮으면 정확하고 일관되지만 뻔한 답, 높으면 다양하지만 엉뚱한 답이 나옵니다.

### 용도별 Temperature 가이드

| temperature | 용도 | 특징 |
|------------|------|------|
| 0.0 | 분류, 추출, 코드 생성 | 가장 확률 높은 답만 선택 |
| 0.3 | 요약, Q&A | 약간의 변화 허용 |
| 0.7 | 일반 대화 | 자연스러운 다양성 |
| 1.0 | 창작, 브레인스토밍 | 학습된 분포 그대로 |
| 1.5+ | 실험적 | 매우 무작위적, 비문 발생 가능 |

```python
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="인공지능의 미래를 한 문장으로 예측해줘",
    config={"temperature": 0.0},
)
# temperature=0.0 → 매번 거의 동일한 결과
```

### temperature = 0의 결정성과 한계

temperature를 0으로 설정하면 이론적으로 매번 동일한 결과가 나와야 합니다. 그러나 실제로는 서버 내부의 병렬 처리 등으로 미세한 차이가 발생할 수 있습니다. 완벽한 결정성이 필요하다면 응답을 캐싱하는 것이 더 확실합니다.

## Top-p (Nucleus Sampling)

**Top-p**(Nucleus Sampling)는 누적 확률이 p에 도달할 때까지의 토큰만 후보로 남기는 필터입니다.

예를 들어, `top_p = 0.9`이면 확률 상위 토큰들을 누적하여 90%에 도달할 때까지만 후보에 포함하고 나머지는 제외합니다. 확률이 극단적으로 낮은 엉뚱한 토큰을 걸러내는 안전망 역할을 합니다.

- `top_p = 1.0`: 필터링 없음 (기본값)
- `top_p = 0.9`: 상위 90% 확률 토큰만 후보
- `top_p = 0.5`: 매우 제한적인 후보군

### Temperature와 Top-p의 상호작용

두 파라미터는 순차적으로 작용합니다. Temperature가 확률 분포를 변형한 뒤, Top-p가 후보를 필터링합니다.

```
원래 분포 → Temperature 적용 → Top-p 필터링 → 최종 토큰 선택
```

| 조합 | 특징 |
|------|------|
| temp=0, top_p=1.0 | 완전 결정적 (top_p 무의미) |
| temp=0.7, top_p=0.9 | 일반적 추천 조합 |
| temp=1.5, top_p=0.95 | 높은 다양성에 안전망 |
| temp=1.5, top_p=1.0 | 최대 무작위 (비문 위험) |

> 대부분의 경우 temperature만 조절하면 충분합니다. top_p는 0.9~0.95로 고정하고, temperature로 다양성을 조절하는 것이 가장 일반적인 패턴입니다.

매트릭스 실험에서 관찰되는 핵심 패턴은 다음과 같습니다.

- **temp=0.0 행**: top_p를 바꿔도 결과가 거의 동일합니다. temperature가 0이면 이미 하나의 토큰만 선택하므로 top_p 필터링이 무의미합니다.
- **temp=1.5 행**: top_p에 따라 결과가 크게 달라집니다. 높은 temperature로 분포가 평탄해진 상태에서 top_p가 실질적인 안전망 역할을 합니다.

## Top-k

**Top-k**는 확률 상위 k개 토큰만 후보로 남기는 필터입니다. Top-p가 확률 비율로 필터링하는 반면, Top-k는 고정 개수로 필터링합니다.

| 비교 항목 | Top-k | Top-p |
|----------|-------|-------|
| 필터링 기준 | 고정 개수 | 누적 확률 비율 |
| 후보 수 | 항상 k개 | 분포에 따라 동적 변화 |
| 분포 적응성 | 없음 | 있음 (뾰족하면 적게, 평탄하면 많이) |
| 실무 선호도 | 낮음 | 높음 |
| API 지원 | Gemini 지원, OpenAI 미지원 | 대부분의 API에서 지원 |

> Top-p가 분포 적응적이기 때문에 실무에서 더 많이 사용됩니다. Top-k는 Gemini 기본값 40을 유지하는 것이 일반적입니다.

## max_output_tokens

**max_output_tokens**는 모델이 생성하는 출력의 최대 토큰 수를 제한합니다. 짧은 답변이 필요한 작업에서 비용과 속도를 절약하는 데 유용합니다.

토큰 한도에 도달하면 문장 중간에서 잘릴 수 있으며, 이때 응답의 `finish_reason`이 `MAX_TOKENS`로 표시됩니다. 정상 완료 시에는 `STOP`입니다.

```python
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="소수를 나열해줘",
    config={"max_output_tokens": 20},
)
print(resp.candidates[0].finish_reason)  # MAX_TOKENS
```

> 분류/추출처럼 짧은 답이 충분한 작업에서는 max_output_tokens를 50~100으로 제한하면 출력 토큰 비용을 절약할 수 있습니다. 단, 잘린 응답이 사용자에게 노출되지 않도록 finish_reason 기반 후처리가 필요합니다.

## stop_sequences

**stop_sequences**는 모델이 특정 문자열을 생성하면 즉시 생성을 중단하게 하는 파라미터입니다. 최대 5개까지 지정할 수 있으며, 중단 문자열 자체는 출력에 포함되지 않습니다.

실용적 활용 예시는 다음과 같습니다.

- 리스트에서 상위 N개만 가져오기: `stop_sequences=["4.", "4)"]`로 3개까지만 출력
- 첫 단락만 추출: `stop_sequences=["\n\n"]`
- 코드 블록만 추출: `` stop_sequences=["```"] ``

```python
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="좋은 습관 5가지를 번호로 알려줘",
    config={"stop_sequences": ["4.", "4)"]},
)
# 3번 항목까지만 출력됨
```

## seed

**seed**는 난수 생성의 시작점을 고정하여 재현 가능한 결과를 만드는 파라미터입니다. 같은 seed, 같은 프롬프트, 같은 파라미터를 사용하면 동일한 결과가 나올 가능성이 높아집니다.

실험 재현, A/B 테스트, 디버깅에서 유용하지만, 모델 업데이트나 서버 인프라 변경에 의해 결과가 달라질 수 있으므로 **최선의 노력(best-effort)** 수준의 재현성입니다. 완벽한 재현이 필요하면 응답을 캐싱하는 것이 더 확실합니다.

## 파라미터 조합 전략

용도별 추천 프리셋은 다음과 같습니다.

| 용도 | temperature | top_p | top_k | max_output_tokens |
|------|-----------|-------|-------|-------------------|
| 코드 생성 | 0.0 | 1.0 | - | 용도에 따라 |
| 분류/추출 | 0.0~0.2 | 1.0 | - | 50~100 |
| 요약/Q&A | 0.3 | 0.95 | 40 | 200~500 |
| 일반 대화 | 0.7 | 0.9 | 40 | 기본값 |
| 창작/브레인스토밍 | 1.0~1.2 | 0.95 | 60 | 기본값 |

> 파라미터 조절 우선순위: (1) temperature를 용도에 맞게 결정 (2) top_p는 0.9~0.95 기본 유지 (3) max_output_tokens로 비용 제어 (4) stop_sequences는 포맷 제어 필요 시에만 (5) seed는 실험 재현 시에만 (6) top_k는 Gemini 기본값 유지

## LangChain에서 파라미터 적용

`ChatGoogleGenerativeAI`에서는 생성자 인자로 파라미터를 설정하며, LCEL 체인에서도 그대로 적용됩니다.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    top_p=0.95,
    top_k=40,
    max_output_tokens=200,
)
```

`bind()` 메서드를 사용하면 기존 모델 인스턴스에 파라미터를 동적으로 추가하거나 변경할 수 있습니다. 같은 모델 인스턴스를 다양한 상황에서 재사용할 때 유용합니다.

```python
strict_model = llm.bind(stop=["\n"])  # stop_sequences 추가
```

### google-genai와 LangChain의 파라미터 표기 차이

| 파라미터 | google-genai (config dict) | LangChain (생성자) |
|---------|--------------------------|-------------------|
| temperature | `"temperature": 0.3` | `temperature=0.3` |
| top_p | `"top_p": 0.95` | `top_p=0.95` |
| top_k | `"top_k": 40` | `top_k=40` |
| max_output_tokens | `"max_output_tokens": 200` | `max_output_tokens=200` |
| stop_sequences | `"stop_sequences": [...]` | `bind(stop=[...])` |
| seed | `"seed": 42` | 생성자에서 직접 지원하지 않음 |

LangChain은 여러 모델 제공자를 추상화하므로, 제공자별 차이가 있는 파라미터(stop_sequences, seed 등)는 `bind()` 또는 `model_kwargs`를 통해 전달합니다.

---

## 정리

- **Temperature**는 가장 기본적인 생성 파라미터로, 확률 분포의 뾰족함을 조절하여 정확성(낮은 값)과 다양성(높은 값)의 균형을 결정합니다.
- **Top-p**는 누적 확률 기반 필터링으로 엉뚱한 토큰을 걸러내는 안전망이며, **Top-k**는 고정 개수 필터링으로 Gemini에서만 지원됩니다. 실무에서는 분포 적응적인 Top-p가 선호됩니다.
- **max_output_tokens**와 **stop_sequences**는 출력 길이와 포맷을 제어하는 도구이며, **seed**는 best-effort 수준의 재현성을 제공합니다.
- 대부분의 경우 temperature 하나만 용도에 맞게 조절하면 충분하며, top_p는 0.9~0.95로 고정하는 것이 일반적인 패턴입니다.
- LangChain에서는 생성자 인자와 `bind()` 메서드를 통해 동일한 파라미터를 적용할 수 있으며, LCEL 체인에서도 모델의 파라미터가 그대로 전달됩니다.
