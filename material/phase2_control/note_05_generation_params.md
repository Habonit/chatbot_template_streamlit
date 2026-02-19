# Note 05. 생성 파라미터

> 대응 노트북: `note_05_generation_params.ipynb`
> Phase 2 — 제어: 모델 동작을 다루기

---

## 학습 목표

- Temperature(온도 파라미터)가 토큰 선택 확률 분포에 미치는 영향을 이해한다
- Top-p(Nucleus Sampling)와 Top-k의 역할, 그리고 Temperature와의 상호작용을 설명할 수 있다
- max_output_tokens, stop_sequences, seed를 활용하여 출력을 제어할 수 있다
- 용도별 최적 파라미터 조합을 도출할 수 있다
- LangChain에서 동일한 파라미터를 적용하는 방법을 안다

---

## 핵심 개념

### 5.1 확률적 토큰 선택과 생성 파라미터

**한 줄 요약**: LLM은 다음 토큰을 확률 분포에서 샘플링하며, 생성 파라미터는 이 분포의 형태와 후보 범위를 조절한다.

LLM은 입력 시퀀스가 주어지면 어휘 전체에 대한 확률 분포를 계산하고, 그 분포에서 다음 토큰을 선택한다. 같은 프롬프트에 다른 결과가 나오는 이유는 이 선택이 확률적이기 때문이다.

```
"오늘 날씨가" → 좋습니다(40%)  맑습니다(25%)  흐립니다(15%)  ...
```

생성 파라미터(Generation Parameters)는 이 확률 분포를 변형하거나 후보를 필터링하여 출력의 특성을 제어한다.

| 파라미터 | 역할 | 값 범위 |
|---------|------|--------|
| temperature | 확률 분포의 뾰족함 조절 | 0.0 ~ 2.0 |
| top_p | 누적 확률 기반 후보 필터링 | 0.0 ~ 1.0 |
| top_k | 상위 k개 후보만 유지 | 1 ~ 100+ |
| max_output_tokens | 출력 길이 상한 | 1 ~ 65,536 |
| stop_sequences | 특정 문자열 생성 시 중단 | 최대 5개 |
| seed | 난수 시작점 고정 | 정수 |

이 파라미터들은 다음 순서로 적용된다.

```mermaid
flowchart LR
    A[원래 확률 분포] --> B[Temperature 적용]
    B --> C[Top-k 필터링]
    C --> D[Top-p 필터링]
    D --> E[최종 토큰 선택]
```

---

### 5.2 Temperature

**한 줄 요약**: Temperature는 확률 분포의 뾰족함을 조절하여 출력의 결정성과 다양성을 제어한다.

Temperature는 소프트맥스 함수의 출력 분포를 스케일링한다. 수학적으로, 각 토큰의 로짓(logit) 값을 temperature 값으로 나눈 뒤 소프트맥스를 적용한다.

- **temperature = 0**: 가장 확률이 높은 토큰만 선택한다 (거의 결정적 출력)
- **temperature = 1.0**: 모델이 학습한 원래 확률 분포 그대로 사용한다
- **temperature > 1.0**: 분포가 평탄해져 낮은 확률의 토큰도 선택될 가능성이 높아진다

```python
# temperature에 따른 응답 차이 비교
response_low = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config={"temperature": 0.0},  # 결정적 출력
)

response_high = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config={"temperature": 1.5},  # 높은 무작위성
)
```

**temperature = 0의 결정성**: temperature를 0으로 설정하면 이론적으로 매번 동일한 결과가 나와야 하지만, 서버 내부의 병렬 처리(부동소수점 연산 순서 차이 등)로 미세한 차이가 발생할 수 있다.

**용도별 권장 범위**:

| temperature | 용도 | 특징 |
|------------|------|------|
| 0.0 | 분류, 추출, 코드 생성 | 가장 확률 높은 답만 선택 |
| 0.3 | 요약, Q&A | 약간의 변화 허용 |
| 0.7 | 일반 대화 | 자연스러운 다양성 |
| 1.0 | 창작, 브레인스토밍 | 학습된 분포 그대로 |
| 1.5+ | 실험적 | 매우 무작위적, 비문 가능 |

---

### 5.3 Top-p (Nucleus Sampling)

**한 줄 요약**: Top-p는 누적 확률이 p에 도달할 때까지의 토큰만 후보로 남기는 동적 필터이다.

Top-p는 Nucleus Sampling이라고도 하며, Holtzman et al.(2020)의 논문 "The Curious Case of Neural Text Degeneration"에서 제안되었다. 확률이 높은 토큰부터 누적하여 합이 p를 초과하는 지점까지의 토큰만 후보로 유지한다.

```
top_p = 0.9일 때:
좋습니다(40%) + 맑습니다(25%) + 흐립니다(15%) + 따뜻합니다(10%) = 90%
→ 이 4개만 후보로 남기고, 나머지 토큰은 제외
```

- **top_p = 1.0**: 필터링 없음 (기본값)
- **top_p = 0.9**: 상위 90% 누적 확률의 토큰만 후보
- **top_p = 0.5**: 상위 50% 누적 확률의 토큰만 후보 (매우 제한적)

Top-p의 핵심 특성은 후보 수가 확률 분포의 형태에 따라 동적으로 변한다는 점이다. 분포가 뾰족하면(한 토큰의 확률이 매우 높으면) 적은 수의 후보만 남고, 분포가 평탄하면 많은 후보가 남는다.

```python
# top_p에 따른 응답 비교 (temperature 고정)
resp = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config={
        "temperature": 1.0,  # temperature를 고정하고 top_p만 변경
        "top_p": 0.9,
    },
)
```

**Temperature + Top-p 상호작용**: Temperature가 확률 분포를 변형한 뒤, Top-p가 후보를 필터링한다. 둘 다 높이면 매우 무작위적이고, 둘 다 낮추면 매우 결정적이다.

| 조합 | 특징 |
|------|------|
| temp=0, top_p=1.0 | 완전 결정적 |
| temp=0.7, top_p=0.9 | 일반적 추천 조합 |
| temp=1.5, top_p=0.95 | 높은 다양성, 안전망 있음 |
| temp=1.5, top_p=1.0 | 최대 무작위 (비문 위험) |

대부분의 경우 temperature만 조절하면 충분하다. top_p는 0.9~0.95로 고정하고, temperature로 다양성을 조절하는 것이 가장 일반적인 패턴이다.

---

### 5.4 Top-k

**한 줄 요약**: Top-k는 확률 상위 k개 토큰만 후보로 남기는 고정 크기 필터이다.

Top-k는 확률이 높은 순서대로 k개의 토큰만 남기고 나머지를 제거한다. Top-p가 확률 비율로 필터링하는 반면, Top-k는 고정된 개수로 필터링한다.

- **top_k = 1**: 가장 확률 높은 토큰 1개만 선택 (greedy decoding, temperature=0과 유사)
- **top_k = 40**: 상위 40개 토큰 중에서 선택 (Gemini 기본값)
- **top_k = 100+**: 거의 필터링 없음

```python
resp = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config={
        "temperature": 1.0,
        "top_k": 40,
    },
)
```

**Top-k vs Top-p의 차이**:

| 구분 | Top-k | Top-p |
|------|-------|-------|
| 필터링 기준 | 고정 개수 | 누적 확률 |
| 후보 수 | 항상 k개 | 분포에 따라 동적 변화 |
| 분포 적응성 | 없음 | 있음 |
| API 지원 | Gemini 지원, OpenAI 미지원 | 대부분의 API에서 지원 |

Top-p는 확률 분포에 적응적이므로 실무에서 더 많이 사용된다. Top-k는 Gemini API에서 지원하지만 OpenAI API에는 없는 파라미터이다.

---

### 5.5 파라미터 조합 전략

**한 줄 요약**: Temperature, Top-p, Top-k를 용도에 맞게 조합하면 출력 품질을 최적화할 수 있다.

세 파라미터를 용도별로 조합하면 다음과 같은 프리셋을 구성할 수 있다.

| 용도 | temperature | top_p | top_k | 설명 |
|------|-----------|-------|-------|------|
| 코드 생성 | 0.0 | 1.0 | - | 정확성 최우선 |
| 분류/추출 | 0.0~0.2 | 1.0 | - | 일관된 결과 |
| 요약/Q&A | 0.3 | 0.95 | 40 | 약간의 변화 허용 |
| 일반 대화 | 0.7 | 0.9 | 40 | 자연스러운 다양성 |
| 창작/브레인스토밍 | 1.0~1.2 | 0.95 | 60 | 높은 다양성 |

Temperature x Top-p 매트릭스 실험에서 관찰되는 패턴:

- **temp=0.0 행**: top_p를 바꿔도 결과가 거의 동일하다. temperature가 0이면 이미 하나의 토큰만 선택하므로 top_p 필터링이 무의미하다.
- **temp=1.5 행**: top_p에 따라 결과가 크게 달라진다. 높은 temperature로 분포가 평탄해진 상태에서 top_p가 안전망 역할을 한다.

---

### 5.6 max_output_tokens

**한 줄 요약**: max_output_tokens는 모델이 생성하는 출력의 최대 토큰 수를 제한하며, 비용 제어와 응답 길이 관리에 사용한다.

- 설정하지 않으면 모델 기본 최대값이 적용된다 (gemini-2.5-flash: 65,536)
- 짧은 답변이 필요할 때 비용과 지연 시간을 절약할 수 있다
- 토큰 한도에 도달하면 문장 중간에서 잘릴 수 있다

```python
resp = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config={"max_output_tokens": 200},
)
```

**finish_reason 확인**: max_output_tokens에 도달하여 생성이 중단되면 응답의 `finish_reason`이 `MAX_TOKENS`로 표시된다. 정상적으로 완료된 경우에는 `STOP`이다.

```python
# finish_reason으로 응답 절단 여부 확인
finish_reason = resp.candidates[0].finish_reason
# STOP: 정상 완료 / MAX_TOKENS: 토큰 한도 도달로 절단
```

비용 제어 관점에서, 분류나 추출처럼 짧은 답변이 필요한 작업에서는 50~100으로 제한하면 출력 토큰 비용을 절약할 수 있다. 단, 잘린 응답(`finish_reason=MAX_TOKENS`)이 사용자에게 노출되지 않도록 후처리가 필요하다.

---

### 5.7 stop_sequences

**한 줄 요약**: stop_sequences는 모델이 특정 문자열을 생성하면 즉시 생성을 중단시키는 파라미터이다.

- 최대 5개의 중단 시퀀스를 지정할 수 있다
- 중단 문자열 자체는 출력에 포함되지 않는다
- 출력 포맷 제어나 불필요한 생성 방지에 유용하다

```python
# "7"이 생성되면 즉시 중단
resp = client.models.generate_content(
    model=MODEL,
    contents="1부터 10까지 세어줘: 1, 2, 3, ",
    config={"stop_sequences": ["7"]},
)
```

**실용적 활용 예시**:

- 구분자 기반 파싱: `"\n\n"`으로 첫 단락만 가져오기
- 리스트 제한: `["4.", "4)"]`로 상위 3개 항목만 가져오기
- 코드 블록 끝: `` "```" ``으로 코드 블록만 추출

---

### 5.8 seed

**한 줄 요약**: seed는 난수 생성의 시작점을 고정하여 동일한 조건에서 재현 가능한 결과를 만드는 파라미터이다.

같은 seed, 같은 프롬프트, 같은 파라미터를 사용하면 동일한 결과가 나올 가능성이 높아진다. 실험 재현, A/B 테스트, 디버깅에 유용하다.

```python
# seed를 고정하여 재현 가능한 결과 생성
resp = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config={"temperature": 1.0, "seed": 42},
)
```

**seed의 한계**: seed를 사용해도 100% 동일한 결과가 보장되지는 않는다. 이는 "최선의 노력(best-effort)" 재현성이다.

- 모델 업데이트 시 결과가 달라질 수 있다
- 서버 인프라 변경에 영향을 받을 수 있다
- 완벽한 재현이 필요하면 응답을 캐싱하는 것이 더 확실하다

---

### 5.9 google-genai config 통합 사용

**한 줄 요약**: 모든 생성 파라미터를 하나의 config 딕셔너리 또는 `GenerateContentConfig` 객체로 통합하여 전달할 수 있다.

google-genai SDK에서는 config를 딕셔너리 또는 `types.GenerateContentConfig` 타입으로 전달한다.

```python
from google.genai import types

response = client.models.generate_content(
    model=MODEL,
    contents="좋은 코딩 습관 3가지를 알려줘",
    config=types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.95,
        top_k=40,
        max_output_tokens=200,
        stop_sequences=["4.", "4)"],
        seed=42,
    ),
)
```

**파라미터 적용 우선순위 가이드**:

1. **temperature**: 가장 먼저 조절한다. 용도에 따라 0.0~1.2 범위에서 결정한다
2. **top_p**: 기본값(0.9~0.95)이 대부분 충분하다. 극단적 다양성이 필요할 때만 조절한다
3. **max_output_tokens**: 비용 제어와 응답 길이 제한에 사용한다
4. **stop_sequences**: 특정 포맷 제어가 필요할 때만 사용한다
5. **seed**: 실험 재현이 필요할 때만 설정한다
6. **top_k**: Gemini 전용이다. 일반적으로 기본값(40)을 유지한다

---

### 5.10 LangChain에서 파라미터 적용

**한 줄 요약**: LangChain의 `ChatGoogleGenerativeAI`에서 생성자 파라미터 또는 `bind()` 메서드로 동일한 생성 파라미터를 적용할 수 있다.

**생성자에서 직접 설정**:

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

**LCEL 체인에서 사용**: `prompt | model | parser` 구조에서도 모델의 생성 파라미터가 그대로 적용된다.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "한 문장으로 답변하세요."),
    ("human", "{question}"),
])

chain = prompt_template | llm | StrOutputParser()
result = chain.invoke({"question": "AI란?"})
```

**bind()로 파라미터 덮어쓰기**: 같은 모델 인스턴스를 다양한 상황에서 재사용할 때 `bind()` 메서드로 파라미터를 동적으로 변경할 수 있다.

```python
base_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# stop_sequences를 추가 바인딩
strict_model = base_model.bind(stop=["\n"])
```

**google-genai vs LangChain 파라미터 비교**:

| 파라미터 | google-genai (config) | LangChain (생성자) |
|---------|----------------------|-------------------|
| temperature | `"temperature": 0.3` | `temperature=0.3` |
| top_p | `"top_p": 0.95` | `top_p=0.95` |
| top_k | `"top_k": 40` | `top_k=40` |
| max_output_tokens | `"max_output_tokens": 200` | `max_output_tokens=200` |
| stop_sequences | `"stop_sequences": [...]` | `bind(stop=[...])` |

LangChain은 여러 모델 제공자를 추상화하므로, stop_sequences처럼 제공자별 차이가 있는 파라미터는 `bind()` 또는 `model_kwargs`를 통해 전달한다.

---

## 장단점

| 장점 | 단점 |
|------|------|
| 동일한 모델에서 용도별로 출력 특성을 제어할 수 있다 | 파라미터 간 상호작용이 있어 최적 조합을 찾기 어렵다 |
| temperature 하나만으로도 출력의 결정성~다양성을 넓게 조절할 수 있다 | temperature=0이어도 100% 동일한 결과가 보장되지 않는다 |
| max_output_tokens로 비용과 지연 시간을 직접 제어할 수 있다 | max_output_tokens가 너무 작으면 응답이 문장 중간에서 절단된다 |
| stop_sequences로 출력 포맷을 구조적으로 제한할 수 있다 | 모델 버전 업데이트 시 동일 파라미터에서 다른 결과가 나올 수 있다 |
| seed로 실험 재현성을 확보할 수 있다 | seed의 재현성은 best-effort이므로 완벽하지 않다 |
| google-genai와 LangChain 양쪽 모두에서 동일한 파라미터를 사용할 수 있다 | top_k 등 일부 파라미터는 제공자별로 지원 여부가 다르다 |

---

## 핵심 정리

| 개념 | 핵심 포인트 |
|------|------------|
| 확률적 토큰 선택 | LLM은 다음 토큰을 확률 분포에서 샘플링한다. 생성 파라미터는 이 분포를 조절하는 도구이다 |
| Temperature | 확률 분포의 뾰족함을 조절한다. 0이면 결정적, 1이면 원래 분포, 1 초과이면 평탄한 분포이다 |
| Top-p | 누적 확률 p까지의 토큰만 후보로 남긴다. 분포에 따라 후보 수가 동적으로 변한다 |
| Top-k | 상위 k개 토큰만 후보로 남긴다. 고정 크기이므로 분포 적응성이 없다. Gemini 전용이다 |
| 파라미터 적용 순서 | Temperature 적용 → Top-k 필터 → Top-p 필터 → 최종 샘플링 |
| max_output_tokens | 출력 토큰 수 상한이다. 초과 시 finish_reason이 MAX_TOKENS가 된다 |
| stop_sequences | 특정 문자열 생성 시 즉시 중단한다. 최대 5개, 해당 문자열은 출력에 포함되지 않는다 |
| seed | 난수 시작점을 고정한다. best-effort 재현성이며 완벽하지 않다 |
| 조합 전략 | temperature를 먼저 조절하고, top_p는 0.9~0.95로 고정하는 것이 일반적이다 |
| LangChain 적용 | 생성자 파라미터로 기본값을 설정하고, bind()로 동적 변경이 가능하다 |

---

## 참고 자료

- [Gemini API: Content generation parameters](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/content-generation-parameters) — temperature, top_p, top_k 등 생성 파라미터 공식 설명
- [Google Gen AI Python SDK Documentation](https://googleapis.github.io/python-genai/) — GenerateContentConfig 레퍼런스 및 사용 예시
- [Experiment with parameter values (Vertex AI)](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/adjust-parameter-values) — 파라미터 값 조정 실험 가이드
- [The Curious Case of Neural Text Degeneration (Holtzman et al., 2020)](https://arxiv.org/abs/1904.09751) — Nucleus Sampling(Top-p) 제안 논문
- [Huyenchip: Generation configs — temperature, top-k, top-p](https://huyenchip.com/2024/01/16/sampling.html) — 샘플링 파라미터 개념 정리
- [LLM에서의 Temperature, Top P, Top K란?](https://www.toolify.ai/ko/ai-news-kr/llm-temperature-top-p-top-k-968111) — 한국어 개념 설명
- [Use model configuration to control responses (Firebase)](https://firebase.google.com/docs/ai-logic/model-parameters) — Gemini 모델 파라미터 설정 가이드
