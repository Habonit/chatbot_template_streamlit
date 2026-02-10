# 노트북 2. System Prompt vs User Prompt + LangSmith

> Phase 1 — 기초

LLM은 '역할(role)'에 따라 메시지를 다르게 취급합니다. 이 구조를 이해해야 챗봇의 행동을 제어할 수 있고, 그 제어가 실제로 잘 되는지 추적하는 도구가 LangSmith입니다.

**학습 목표**
- 메시지 역할 체계(system / user / assistant)를 이해한다
- google-genai의 system_instruction과 LangChain의 SystemMessage를 사용할 수 있다
- 효과적인 System Prompt를 설계하는 원칙을 적용할 수 있다
- LangSmith를 연동하여 LLM 호출을 추적하고 분석할 수 있다

## 메시지 역할 체계

LLM API는 단순히 "질문 -> 답변"이 아니라, **역할(role)이 부여된 메시지 리스트**를 입력으로 받습니다. 각 메시지에는 "누구의 것인지"를 나타내는 역할이 붙어 있습니다.

| 역할 | 설명 | 비유 |
|------|------|------|
| **system** | 모델의 행동 규칙을 정의. 사용자에게는 보이지 않는 지시 | 연극의 연출 노트 |
| **user** (human) | 실제 사용자가 보내는 입력 | 관객의 질문 |
| **assistant** (model/AI) | 모델이 이전에 생성한 응답 | 배우의 대사 |

> 핵심: system 메시지는 모델의 "성격"과 "능력 범위"를 결정합니다. 같은 모델이라도 system 메시지를 바꾸면 완전히 다른 챗봇이 됩니다.

### google-genai와 LangChain의 역할 표현 비교

두 SDK에서 역할을 표현하는 방식이 다릅니다.

| 역할 | google-genai | LangChain |
|------|-------------|-----------|
| system | `system_instruction` 파라미터로 별도 전달 | `SystemMessage` 객체 |
| user | `contents`에서 `role="user"` | `HumanMessage` 객체 |
| assistant | `contents`에서 `role="model"` | `AIMessage` 객체 |

google-genai에서는 `system_instruction`을 별도 파라미터로, user/model은 `contents` 리스트에 배치합니다:

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    config={"system_instruction": "당신은 친절한 한국어 도우미입니다."},
    contents=[Content(role="user", parts=[Part(text="안녕하세요?")])],
)
```

LangChain에서는 역할별 전용 메시지 클래스를 사용합니다:

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="당신은 친절한 한국어 도우미입니다."),
    HumanMessage(content="안녕하세요?"),
]
response = model.invoke(messages)  # 반환 타입: AIMessage
```

이전 대화 맥락을 전달하려면 `AIMessage`를 리스트에 포함합니다. 모델은 역할을 보고 "누가 말한 것인지"를 구분하여, 대화의 흐름을 이해합니다.

---

## System Prompt 설정 방법

**System Prompt**(시스템 프롬프트)는 모델에게 "이렇게 행동해라"를 지시하는 메시지입니다. 사용자에게는 보이지 않지만 모델의 모든 응답에 영향을 미칩니다.

google-genai에서는 `config` 딕셔너리의 `system_instruction` 키로 전달합니다:

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    config={
        "system_instruction": "당신은 시인입니다. 모든 답변을 시 형태로 작성하세요.",
    },
    contents="봄에 대해 알려주세요.",
)
```

LangChain에서는 `SystemMessage`를 메시지 리스트의 첫 번째에 배치하거나, `ChatPromptTemplate`에서 튜플로 정의합니다:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 시인입니다. 모든 답변을 시 형태로 작성하세요."),
    ("human", "{question}"),
])
chain = prompt | model | StrOutputParser()
result = chain.invoke({"question": "봄에 대해 알려주세요."})
```

> 핵심: google-genai는 `system_instruction` 파라미터로 분리하고, LangChain은 `SystemMessage`를 메시지 리스트에 포함합니다. 동작은 동일합니다.

### system prompt에 따른 응답 변화

동일한 질문이라도 system prompt를 바꾸면 관점, 톤, 길이가 완전히 달라집니다. 예를 들어 "인공지능이 일자리에 미치는 영향은?"이라는 질문에 "기술 낙관론자", "기술 비관론자", "중립 분석가" 페르소나를 각각 적용하면, 세 가지 서로 다른 관점의 응답을 얻게 됩니다. 이것이 system prompt의 핵심 가치입니다.

---

## System Prompt 설계 원칙

좋은 System Prompt는 세 가지 요소로 구성됩니다.

### 페르소나 (Persona)

"당신은 ~입니다"로 시작하는 역할 정의입니다. 모델은 이 정의에 맞춰 어휘, 톤, 지식 범위를 조절합니다. "초등학교 3학년 담임 선생님"과 "서울대학교 물리학과 교수"에게 같은 과학 질문을 하면, 설명의 깊이와 용어가 완전히 달라집니다.

### 제약 조건 (Constraints)

"반드시 ~해야 한다", "절대 ~하지 마라" 형태의 규칙입니다. 모델의 행동 범위를 좁혀서 예측 가능한 응답을 만듭니다.

```python
system_prompt = """
당신은 고객 상담 챗봇입니다.

규칙:
- 반드시 존댓말을 사용하세요.
- 모든 답변은 3문장 이내로 작성하세요.
- 가격이나 할인에 대한 질문에는 "담당자에게 연결해드리겠습니다"로 답변하세요.
- 경쟁사 제품에 대한 비교 질문에는 답변을 거절하세요.
"""
```

이 규칙이 적용된 상태에서 가격 관련 질문을 하면, 모델은 직접 답변하지 않고 담당자 연결을 안내합니다.

### 출력 포맷 (Format)

"JSON으로 응답", "3문장 이내", "번호를 매겨서" 등 출력 형태를 명시합니다. 후속 코드에서 응답을 파싱해야 할 때 특히 중요합니다. 포맷을 지정하지 않은 응답은 길이와 구조가 매번 달라지지만, 명시적으로 지정하면 일관된 형태를 얻을 수 있습니다.

> 핵심: 페르소나("~입니다"), 제약 조건("반드시/절대"), 출력 포맷("~형태로") — 이 세 가지가 구체적일수록 모델의 응답이 예측 가능하고 일관됩니다.

### 좋은 System Prompt vs 나쁜 System Prompt

| 구분 | 나쁜 예 | 좋은 예 |
|------|---------|---------|
| 페르소나 | "좋은 상담사가 되어줘" | "10년 경력의 조직심리학 전문 상담사" |
| 제약 | (없음) | "감정 인정 문장으로 시작, 행동 단계 3가지, 200자 이내" |
| 포맷 | (없음) | "번호 매기기, 한 줄씩, 마지막에 격려 문장" |
| 결과 | 길이와 구조가 매번 달라짐 | 일관되고 구조화된 응답 |

### LangChain에서의 변수화 패턴

`ChatPromptTemplate`을 사용하면 system prompt에도 변수를 넣어, 하나의 템플릿으로 다양한 페르소나를 구현할 수 있습니다:

```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 {specialty} 전문가입니다. {style}으로 답변하세요."),
    ("human", "{question}"),
])
chain = prompt_template | model | StrOutputParser()
result = chain.invoke({"specialty": "한식 요리", "style": "레시피 단계별 형태", "question": "김치찌개 만드는 법"})
```

변수만 바꾸면 동일한 체인으로 법률 전문가, 요리 전문가, 역사학자 등 다양한 도메인 챗봇을 만들 수 있습니다.

---

## LangSmith 연동

**LangSmith**는 LangChain 팀이 만든 LLM 개발 플랫폼으로, LLM 호출을 자동으로 **트레이싱(tracing)**하여 실제 전송된 프롬프트와 응답을 대시보드에서 확인할 수 있게 합니다.

### 왜 LangSmith가 필요한가

system prompt를 바꿨을 때 "정말 의도대로 동작하는지"를 확인하는 방법이 필요합니다. "느낌"이 아니라 "데이터"로 프롬프트를 개선하려면, 실제 전송된 프롬프트와 응답을 기록하고 비교할 수 있어야 합니다. LangSmith가 바로 그 역할을 합니다.

### 설정 방법

환경변수 3개만 설정하면 모든 LangChain 호출이 자동으로 트레이싱됩니다:

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"           # 트레이싱 활성화
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"       # LangSmith API 키
os.environ["LANGCHAIN_PROJECT"] = "note-02-prompt"     # 프로젝트명
```

이후 `model.invoke()`나 LCEL 체인을 실행하면, 해당 호출이 자동으로 LangSmith 대시보드에 기록됩니다. 추가 코드 변경 없이 동작합니다.

### 대시보드에서 볼 수 있는 것

| 항목 | 설명 |
|------|------|
| Input | 실제 전송된 프롬프트 전문 (system + user 메시지) |
| Output | 모델의 응답 전문 |
| Tokens | 입력/출력 토큰 수 |
| Latency | 응답 시간 |
| Cost | 비용 추정 |
| Model | 사용된 모델명과 버전 |

LCEL 체인을 실행하면 체인 내부의 각 단계(prompt -> model -> parser)가 개별적으로 트레이싱됩니다. 체인의 어느 단계에서 시간이 걸리는지도 파악할 수 있습니다.

### 프로젝트 분리로 실험 관리

`LANGCHAIN_PROJECT` 환경변수로 프로젝트를 분리하면, system prompt A와 B의 결과를 각각 다른 프로젝트에 기록하여 비교할 수 있습니다. 예를 들어 "짧은 답변 스타일"과 "상세한 답변 스타일"을 별도 프로젝트로 기록한 뒤, 대시보드에서 토큰 사용량, 응답 품질, 비용을 나란히 비교하는 방식입니다.

> 핵심: LangSmith 연동은 환경변수 3개만 설정하면 됩니다. system prompt를 변경할 때마다 "의도대로 동작하는지"를 데이터로 확인할 수 있고, 프로젝트를 분리하여 A/B 비교도 가능합니다.

---

## 정리

- LLM API는 **역할(role)이 부여된 메시지 리스트**를 입력으로 받으며, system/user/assistant 세 역할이 각각 모델의 행동 규칙, 사용자 입력, 이전 응답을 담당한다
- google-genai는 `system_instruction`으로, LangChain은 `SystemMessage`로 system prompt를 전달하며, 동작은 동일하다
- 좋은 System Prompt는 **페르소나 + 제약 조건 + 출력 포맷**으로 구성되며, 구체적일수록 응답이 예측 가능하고 일관된다
- LangSmith는 환경변수 3개 설정만으로 모든 LLM 호출을 자동 트레이싱하여, 프롬프트 전문/토큰/지연시간/비용을 데이터로 확인할 수 있다
- 프로젝트 분리 기능을 활용하면 system prompt 변경 전후를 비교하여, "느낌"이 아닌 "근거"로 프롬프트를 개선할 수 있다
