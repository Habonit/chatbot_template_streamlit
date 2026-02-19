# 노트북 8. Structured Output

> Phase 3 — 실전 기법

LLM 출력을 사람이 읽는 게 아니라 코드가 파싱해야 한다면, 구조화된 출력이 필수입니다.
"JSON으로 답해줘"라는 프롬프트만으로는 `json.loads()` 실패가 빈번하게 발생하며, 이는 서비스 장애로 직결됩니다.

**학습 목표**
- 프롬프트 기반 JSON 요청의 불안정성을 이해한다
- google-genai의 `response_mime_type`과 `response_schema`를 사용할 수 있다
- LangChain의 `with_structured_output()`과 Pydantic 모델을 활용할 수 있다
- 파싱 실패 시 디버깅과 재시도 패턴을 적용할 수 있다

## 왜 구조화된 출력이 필요한가

LLM의 출력을 코드에서 처리해야 하는 상황은 많습니다. API 서버에서 JSON 응답을 반환할 때, 분류 결과를 데이터베이스에 저장할 때, 추출된 정보를 다른 시스템에 전달할 때가 대표적입니다.

프롬프트에 "JSON으로 답해줘"라고 쓰면 모델이 JSON을 반환하기는 하지만, 다음과 같은 문제가 발생합니다.

| 문제 | 예시 |
|------|------|
| 마크다운 코드블록 감싸기 | ` ```json {...} ``` ` |
| 자연어 혼합 | `다음은 결과입니다: {"title": ...}` |
| 키 이름 불일치 | `movie_title` vs `title` |
| 타입 불일치 | `rating: "9"` (문자열) vs `rating: 9` (정수) |
| 필드 누락 | 요청한 필드가 빠짐 |

> 출력이 사람이 아닌 코드가 소비한다면 **Structured Output**(구조화된 출력)을 사용합니다. 자유로운 대화, 설명, 창작, 번역처럼 사람이 직접 읽는 경우에는 일반 텍스트 출력이 적절합니다.

## google-genai: response_mime_type으로 JSON 강제

가장 간단한 방법은 `response_mime_type="application/json"`을 설정하는 것입니다. 모델이 유효한 JSON만 출력하도록 강제합니다.

```python
response = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config={"response_mime_type": "application/json"},
)
data = json.loads(response.text)  # 항상 성공
```

그러나 이것만으로는 키 이름, 타입, 필수 필드를 보장할 수 없습니다. 동일한 프롬프트로 여러 번 호출하면 키 이름이 달라질 수 있습니다. 더 확실한 제어를 위해 `response_schema`를 함께 사용해야 합니다.

## google-genai: response_schema로 제어 생성

`response_schema`를 설정하면 모델이 정확히 해당 스키마에 맞는 JSON만 생성합니다. 이를 **제어 생성(Controlled Generation)**이라 합니다.

스키마는 **JSON Schema(dict)**와 **Pydantic 모델(클래스)** 두 가지 방식으로 정의할 수 있습니다. JSON Schema는 `{"type": "object", "properties": {...}, "required": [...]}`형태의 dict를 전달하며, 배열 타입(`"type": "array"`)도 지원합니다.

Pydantic 모델을 사용하면 더 간결하고 타입 안전합니다. `response_schema`에 클래스를 직접 전달할 수 있습니다.

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title: str = Field(description="영화의 한국어 제목")
    year: int = Field(description="개봉 연도 (4자리 숫자)")
    rating: float = Field(description="IMDb 기준 평점 (0.0~10.0)")

response = client.models.generate_content(
    model=MODEL,
    contents="기생충 영화 정보를 알려줘",
    config={"response_mime_type": "application/json", "response_schema": Movie},
)
movie = Movie.model_validate_json(response.text)  # Pydantic 인스턴스로 변환
```

> `Field(description=...)`은 모델이 각 필드에 어떤 값을 채워야 하는지 판단하는 핵심 단서입니다. 값의 범위(`"0.0~10.0"`), 단위(`"만 명 단위"`), 언어(`"한국어로 작성"`)를 명시하면 출력 정확도가 높아집니다. description이 없으면 모델은 필드 이름만으로 추측합니다.

## LangChain: with_structured_output()

LangChain은 `with_structured_output()` 메서드를 제공합니다. Pydantic 모델을 전달하면 반환값이 자동으로 Pydantic 인스턴스가 됩니다. 수동 파싱이 필요 없습니다.

```python
llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=GEMINI_API_KEY)
structured_llm = llm.with_structured_output(Movie)
result = structured_llm.invoke("매트릭스 영화 정보를 알려줘")
# result는 Movie 인스턴스 — result.title, result.year로 바로 접근
```

LCEL 체인에서도 자연스럽게 동작합니다. `prompt_template | llm.with_structured_output(Movie)` 형태로 파이프라인을 구성하면 됩니다.

### method 비교: json_schema vs function_calling

`with_structured_output`는 두 가지 내부 메커니즘을 지원합니다.

| 항목 | `"json_schema"` | `"function_calling"` |
|------|-----------------|---------------------|
| 내부 동작 | Gemini의 `response_schema` 활용 | Tool Calling 메커니즘 활용 |
| 안정성 | 더 안정적 (Gemini 전용) | 범용적 (모든 LLM 지원) |
| 기본값 | Gemini에서 기본 | OpenAI 등 다른 LLM에서 기본 |

> Gemini 전용 서비스라면 `"json_schema"`가 권장됩니다 (더 안정적). 여러 LLM 제공자를 지원해야 한다면 `"function_calling"`이 범용적입니다. 대부분의 경우 기본값을 그대로 사용하면 됩니다.

## Pydantic 모델 설계 심화

### Literal로 선택지 제한

`Literal`을 사용하면 필드 값을 특정 선택지로 제한할 수 있습니다. 분류 작업에서 `str` 대신 `Literal`을 쓰면 정확도가 크게 향상됩니다.

```python
from typing import Literal

class Sentiment(BaseModel):
    text: str = Field(description="분석 대상 텍스트")
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(description="확신도 (0.0~1.0)")
```

### Optional 필드

`Optional`을 사용하면 값이 없을 수 있는 필드를 정의합니다. `Optional[int] = Field(default=None, description="나이 (언급되지 않으면 null)")` 형태로 선언하면, 모델이 정보가 없을 때 억지로 값을 만들지 않고 null을 반환합니다.

`Optional`과 기본값(`default="없음"`)은 동작이 다릅니다. `Optional`은 모르면 null을 반환하고, 기본값이 있으면 모델이 항상 문자열을 강제하므로 억지 값을 생성할 수 있습니다.

### 중첩 모델과 리스트

Pydantic 모델은 다른 모델을 필드로 포함할 수 있습니다.

```python
class Actor(BaseModel):
    name: str = Field(description="배우 이름")
    role: str = Field(description="극중 역할 이름")

class MovieFull(BaseModel):
    title: str = Field(description="영화 제목")
    genre: list[str] = Field(description="장르 리스트 (최대 3개)")
    actors: list[Actor] = Field(description="주요 배우 3명")
```

### 자주 하는 실수

| 실수 | 증상 | 해결 |
|------|------|------|
| `Field(description=...)` 누락 | 필드 값이 부정확하거나 일관성 없음 | 모든 필드에 description 추가 |
| 중첩 3단계 이상 | 깊은 필드를 빈 값으로 채움 | 2단계까지만 중첩, 나머지는 평면화 |
| `list[str]` 길이 미지정 | 0개 또는 수십 개 반환 | description에 "최대 N개" 명시 |
| `float` 범위 미지정 | 0.8인지 80인지 스케일 혼동 | description에 범위 명시 |
| `Optional` 남용 | 가능한 필드도 null로 반환 | 필수 필드는 required로 유지 |

## 실패 핸들링: include_raw와 재시도

`include_raw=True`를 설정하면 원본 응답, 파싱 결과, 파싱 에러를 함께 받을 수 있습니다.

```python
result = llm.with_structured_output(Movie, include_raw=True).invoke(prompt)
result["raw"]            # 원본 AIMessage
result["parsed"]         # 파싱된 Pydantic 인스턴스 (실패 시 None)
result["parsing_error"]  # 파싱 에러 (성공 시 None)
```

파싱 실패 시 자동으로 재시도하는 패턴을 구성할 수 있습니다. `result["parsing_error"]`가 None인지 확인하고, 실패 시 최대 2~3회 반복 호출한 뒤, 그래도 실패하면 에러를 발생시킵니다.

> 프로덕션 환경에서는 `include_raw=True`로 원본 응답을 항상 로깅하고, 최대 2~3회 재시도합니다. 재시도 비용이 높으므로 스키마를 단순화하는 것이 근본적 해결책입니다.

## 스트리밍 Structured Output

구조화 출력도 스트리밍할 수 있습니다. LangChain의 `with_structured_output`에서 `.stream()`을 사용하면 각 청크가 점진적으로 채워지는 부분적 Pydantic 인스턴스(또는 dict)를 반환합니다.

```python
structured_stream = llm.with_structured_output(Movie)
final = None
for chunk in structured_stream.stream("반지의 제왕 영화 정보를 알려줘"):
    final = chunk  # 마지막 청크가 최종 결과
print(final.title)  # 완성된 값
```

UI에서 점진적으로 필드를 채워 보여줄 때 유용합니다. 새로 채워진 필드를 감지하여 실시간으로 표시할 수 있습니다.

> 중간 청크는 불완전한 상태일 수 있습니다 (필드가 None이거나 부분 문자열). 최종 결과는 반드시 마지막 청크에서 얻어야 하며, 실시간 UI에 부분 결과를 표시할 때는 None 체크가 필요합니다.

## google-genai vs LangChain 비교

| 항목 | google-genai | LangChain |
|------|-------------|----------|
| 스키마 정의 | `response_schema` (JSON Schema 또는 Pydantic) | `with_structured_output(Pydantic)` |
| 반환 타입 | JSON 문자열 (수동 파싱 필요) | 자동으로 Pydantic 인스턴스 |
| 실패 핸들링 | 직접 `try/except` | `include_raw=True` |
| 스트리밍 | JSON 청크 직접 처리 | 부분 인스턴스 자동 생성 |
| 모델 교체 | Gemini 전용 | 모든 LLM 호환 |

> 빠른 프로토타이핑에는 google-genai가 직접적이고, 프로덕션 서비스에는 LangChain이 편리합니다 (자동 파싱, include_raw, 재시도). 모델 교체 가능성이 있다면 LangChain이 권장됩니다.

---

## 정리

- 프롬프트만으로 JSON을 요청하면 마크다운 감싸기, 키 불일치, 타입 불일치 등으로 `json.loads()` 실패가 빈번합니다
- google-genai의 `response_mime_type`은 유효한 JSON을 강제하고, `response_schema`는 스키마(키 이름, 타입, 필수 필드)까지 보장합니다
- LangChain의 `with_structured_output()`은 Pydantic 모델을 전달하면 파싱까지 자동으로 처리하며, `json_schema`와 `function_calling` 두 가지 method를 지원합니다
- Pydantic 모델 설계 시 `Field(description=...)`, `Literal`, `Optional`, 중첩 모델을 적절히 활용하되, 중첩은 2단계까지가 적절합니다
- 파싱 실패에 대비하여 `include_raw=True`로 원본을 확보하고, 재시도 로직을 구현하되 스키마 단순화가 근본적 해결책입니다
