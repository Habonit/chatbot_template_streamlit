알겠습니다. 각 노트북이 **"무엇을 가르치는가"**에 집중해서 컨텐츠 자체를 정리하겠습니다.

---

## 노트북 1. Gemini 직접 호출 vs LangChain 호출 비교

### 핵심 메시지
"같은 Gemini 모델인데, 부르는 방법이 두 가지다. 차이를 알아야 언제 뭘 쓸지 판단할 수 있다."

### 가르쳐야 할 개념

**google-genai SDK (직접 호출)**
- 2025년 11월 기준으로 구 `google-generativeai` 패키지는 지원 종료(EOL). 현재 공식 SDK는 `google-genai`이며 `Client` 패턴을 사용
- `client = genai.Client(api_key=...)` → `client.models.generate_content()` 구조
- 이것은 Google이 제공하는 1차 인터페이스로, 모든 Gemini 기능에 가장 먼저 접근 가능

**LangChain 래핑 (간접 호출)**
- `langchain-google-genai` 4.x부터 내부적으로 `google-genai` SDK를 사용 (구 SDK 의존성 제거)
- `ChatGoogleGenerativeAI`는 LangChain의 `BaseChatModel`을 구현 → `.invoke()`, `.stream()`, `.batch()`, `|` 체이닝 등 통일된 인터페이스 제공
- **핵심 가치**: 모델 교체 용이성. Google → OpenAI → Anthropic으로 바꿔도 나머지 코드가 동일

**비교 포인트**
- 반환 객체: google-genai의 `GenerateContentResponse` vs LangChain의 `AIMessage`
- google-genai는 Gemini 전용 기능(Live API, 이미지 생성 등)에 직접 접근 가능
- LangChain은 체이닝, 메모리, 도구 바인딩 등 오케스트레이션에 강점
- 실무에서는 대부분 LangChain/LangGraph를 통해 호출하되, 특수 기능이 필요할 때 google-genai 직접 사용

**LCEL 맛보기**
- `prompt | model | output_parser` 체인을 처음 구성해봄으로써 이후 노트북에서 계속 쓸 패턴을 미리 체험

---

## 노트북 2. System Prompt vs User Prompt + LangSmith

### 핵심 메시지
"LLM은 '역할(role)'에 따라 메시지를 다르게 취급한다. 이 구조를 이해해야 챗봇의 행동을 제어할 수 있다. 그리고 그 제어가 실제로 잘 되는지 추적하는 도구가 LangSmith다."

### 가르쳐야 할 개념

**메시지 역할 체계**
- `system`: 모델의 행동 규칙을 정의. 사용자에게는 보이지 않는 "무대 뒤 지시"
- `user`(human): 실제 사용자 입력
- `assistant`(model/AI): 모델의 이전 응답
- google-genai에서는 `system_instruction`으로 분리, LangChain에서는 `SystemMessage`로 messages 리스트 첫 번째에 배치
- **실험**: 동일한 질문에 system prompt만 바꾸면 톤, 길이, 포맷이 완전히 달라진다는 것을 직접 확인

**System Prompt 설계 원칙**
- 페르소나 정의 ("너는 ~이다")
- 제약 조건 ("반드시 ~해야 한다", "절대 ~하지 마라")
- 출력 포맷 지정 ("JSON으로 응답", "3문장 이내")
- 이것들이 결국 챗봇의 "성격"과 "능력 범위"를 결정

**LangSmith 연동**
- 환경변수 3개 세팅만으로 모든 LLM 호출이 자동 트레이싱
- 대시보드에서 볼 수 있는 것들: 실제 전송된 프롬프트 전문, 모델 응답, 토큰 수, 지연시간, 비용 추정
- **왜 중요한가**: system prompt를 바꿨을 때 "정말 의도대로 동작하는지"를 데이터로 확인 가능. "느낌"이 아니라 "근거"로 프롬프트를 개선
- LangSmith Hub: 프롬프트를 버전 관리하고, 팀원과 공유하고, A/B 비교 가능

---

## 노트북 3. Single-turn vs Multi-turn + 대화 저장 전략

### 핵심 메시지
"LLM에는 '기억'이 없다. 매번 전체 대화 내역을 통째로 보내야 한다. 그 대화를 어디에 저장하느냐가 아키텍처 결정이다."

### 가르쳐야 할 개념

**Single-turn의 본질적 한계**
- LLM API 호출은 본질적으로 stateless. 이전 호출의 내용을 기억하지 못함
- "아까 말한 거"라고 하면 "무엇을 말씀하셨는지 모르겠습니다"가 나오는 이유

**Multi-turn의 작동 원리**
- 클라이언트 측에서 `messages` 리스트를 누적 관리하고, 매 호출 시 전체를 전송
- google-genai: `contents` 리스트에 `user`/`model` role 교대로 쌓기, 또는 `client.chats.create()` 세션 활용
- LangChain: `List[BaseMessage]` 수동 관리, 또는 `ChatMessageHistory` + `RunnableWithMessageHistory` 패턴

**대화 저장소 선택**
- 저장하지 않음 (InMemory): 프로세스 종료 시 소멸. 프로토타입용
- Redis: 빠르고, TTL로 세션 자동 만료 가능. 다중 서버 환경에 적합
- RDB (SQLite/PostgreSQL): 영구 보존, 검색·분석 가능. 운영 복잡도 상승
- **선택 기준**: 대화를 얼마나 오래 보관해야 하는가? 서버가 몇 대인가? 대화 이력을 나중에 분석해야 하는가?

**LangGraph에서의 접근**
- `MessagesState`와 `add_messages` reducer: 그래프 상태 자체에 메시지 이력이 내장
- 이후 노트북 15에서 이 패턴을 종합 조립에 직접 사용

---

## 노트북 4. 컨텍스트 윈도우와 토큰

### 핵심 메시지
"토큰은 LLM의 화폐다. 비용도 토큰, 성능 한계도 토큰, 응답 품질도 토큰에 달렸다."

### 가르쳐야 할 개념

**토큰의 실체**
- 단어 단위가 아니라 서브워드(subword) 단위. "unhappiness" → ["un", "happiness"] 같은 분할
- 한국어는 영어보다 토큰 효율이 낮음 (같은 의미를 전달하는 데 더 많은 토큰 소비)
- Gemini의 `count_tokens()` API로 직접 확인

**컨텍스트 윈도우**
- input tokens + output tokens ≤ context window
- Gemini 2.5 Flash/Pro: 1M tokens (약 70만 단어 분량). 하지만 긴 컨텍스트 ≠ 좋은 성능
- 윈도우를 초과하면 에러 발생. 이 에러를 직접 재현해보는 것이 중요

**비용 구조**
- input tokens과 output tokens의 과금 단가가 다름
- 멀티턴 대화에서 토큰이 누적되면 비용이 기하급수적으로 증가하는 구조
- 이것이 노트북 7(컨텍스트 매니지먼트)의 동기

**Long Context vs RAG**
- Gemini의 1M 윈도우에 문서를 통째로 넣는 것과, RAG로 관련 부분만 검색해서 넣는 것의 트레이드오프
- 비용, 정확도(needle-in-a-haystack 문제), 응답 속도 측면 비교
- 이것이 노트북 12(RAG)의 동기

---

## 노트북 5. 생성 파라미터

### 핵심 메시지
"같은 모델, 같은 프롬프트인데 결과가 매번 다른 이유를 이해하고, 용도에 맞게 제어할 수 있어야 한다."

### 가르쳐야 할 개념

**Temperature**
- 다음 토큰 선택 시 확률 분포의 "뾰족함"을 조절
- 0: 가장 확률 높은 토큰만 선택 (거의 결정적)
- 1.0+: 분포가 평탄해져 다양한 토큰이 선택될 수 있음
- 코드 생성·분류·추출에는 낮게, 창작·브레인스토밍에는 높게

**Top-p (Nucleus Sampling)**
- 누적 확률이 p에 도달할 때까지의 토큰만 후보로 남김
- temperature와 조합하여 사용. 둘 다 높이면 매우 무작위적

**Top-k**
- 확률 상위 k개 토큰만 후보로 남김
- Gemini에서는 지원하지만 OpenAI API에는 없는 파라미터. 모델별 차이 인식

**기타 파라미터**
- `max_output_tokens`: 출력 길이 상한 제어
- `stop_sequences`: 특정 문자열이 나오면 생성 중단
- `seed`: 재현성을 위한 시드 (Gemini 지원)

**실습 설계**
- temperature × top_p 조합 매트릭스 실험 → 결과를 표로 정리해서 "이런 용도에는 이런 세팅" 가이드를 스스로 도출

---

## 노트북 6. Streaming 응답

### 핵심 메시지
"사용자는 빈 화면에서 3초 기다리는 것보다, 글자가 하나씩 나오는 것을 훨씬 빠르다고 느낀다."

### 가르쳐야 할 개념

**스트리밍 vs 비스트리밍**
- 비스트리밍: 모델이 전체 응답을 생성한 후 한 번에 반환. 긴 응답일수록 사용자 대기 시간 증가
- 스트리밍: 토큰이 생성되는 즉시 청크 단위로 반환. 첫 토큰이 나오는 시간(TTFT)만 기다리면 됨

**구현 방식**
- google-genai: `client.models.generate_content_stream()` → 청크별 `.text` 접근
- LangChain: `.stream()` → `AIMessageChunk` 순회 / `.astream()` 비동기 버전
- LCEL 체인에서도 `chain.stream(input)`으로 체인 전체가 스트리밍 가능

**TTFT (Time To First Token)**
- 스트리밍의 핵심 지표. 모델 크기, 프롬프트 길이, 서버 부하에 영향받음
- 직접 측정하는 코드를 작성하여 모델별/파라미터별 비교

**운영 고려사항**
- 스트리밍 호출도 LangSmith 트레이스에 정상 기록됨
- 프론트엔드와 연결 시 SSE(Server-Sent Events) 패턴 사용이 일반적

---

## 노트북 7. 컨텍스트 매니지먼트 전략

### 핵심 메시지
"대화가 길어지면 토큰은 폭발하고 비용은 치솟고 품질은 떨어진다. 이걸 관리하는 전략이 챗봇의 실전 품질을 결정한다."

### 가르쳐야 할 개념

**문제 정의**
- 멀티턴 대화에서 매 호출마다 전체 이력을 보내므로, 20턴만 되어도 input 토큰이 수천~수만 개
- 비용 증가 + 긴 컨텍스트에서 모델 주의력 분산(Lost in the Middle 현상)

**전략 1: Sliding Window**
- 가장 최근 N턴만 유지, 오래된 메시지 삭제
- 간단하지만 중요한 초기 맥락(이름, 목적 등)이 사라질 수 있음
- LangChain의 `trim_messages()` 유틸리티

**전략 2: Token 기반 트리밍**
- 턴 수가 아니라 토큰 수 기준으로 자르기
- "최근 메시지부터 역순으로 채워서 N 토큰 이내로"

**전략 3: 요약 기반 압축**
- 오래된 대화를 LLM으로 요약 → 요약본을 system prompt에 포함 + 최근 대화만 유지
- 추가 LLM 호출 비용 발생, 요약 과정에서 정보 손실 가능

**전략 4: 하이브리드**
- 요약 + Sliding Window 조합. 실전에서 가장 많이 쓰이는 패턴

**LangGraph에서의 상태 관리**
- `MessagesState`와 `add_messages` reducer
- 그래프 노드 안에서 trim_messages를 적용하는 패턴

**비교 기준**
- 토큰 사용량, 응답 품질, 구현 복잡도, 정보 보존율

---

## 노트북 8. Structured Output

### 핵심 메시지
"LLM 출력을 사람이 읽는 게 아니라 코드가 파싱해야 한다면, 구조화된 출력이 필수다."

### 가르쳐야 할 개념

**왜 필요한가**
- "JSON으로 대답해줘"라고 프롬프트에 쓰면? → 가끔 마크다운 코드블록으로 감싸거나, 자연어를 섞거나, 키 이름이 달라지는 등 불안정
- 코드에서 `json.loads()`가 실패하는 순간 서비스 장애

**google-genai의 접근**
- `response_mime_type="application/json"`: 모델 출력을 JSON으로 강제
- `response_schema`: JSON Schema를 명시하면 모델이 해당 스키마에 맞게만 생성 (제어 생성, Controlled Generation)

**LangChain의 접근**
- `model.with_structured_output(PydanticModel)`: Pydantic 클래스를 바인딩하면 반환값이 자동으로 Pydantic 인스턴스
- 두 가지 method: `"json_schema"` (Gemini 제어 생성 활용, 더 안정적) vs `"function_calling"` (도구 호출 메커니즘 활용)

**Pydantic 모델 설계**
- 필드 타입, description(모델이 이걸 보고 뭘 채울지 판단), Optional, Literal(enum 대체)
- 중첩 객체, 리스트 필드, validator

**스트리밍 시 주의**
- structured output을 스트리밍하면 dict 청크가 옴 → `+=`가 아니라 `.update()`로 병합

**실패 핸들링**
- `include_raw=True`로 원본 응답까지 함께 받아서 디버깅
- 파싱 실패 시 재시도 패턴

---

## 노트북 9. Tool Calling + LangChain/LangGraph 연계

### 핵심 메시지
"LLM은 세상의 정보를 알지 못하고, 계산도 못 하고, API도 호출 못 한다. Tool Calling은 LLM에게 '손과 발'을 달아주는 메커니즘이다."

### 가르쳐야 할 개념

**Tool Calling의 본질**
- LLM이 직접 함수를 실행하는 게 아님
- 모델은 "이 함수를 이 인자로 호출해야 한다"는 **의도(intent)**를 JSON으로 출력
- 실제 실행은 클라이언트(개발자 코드)가 수행 → 결과를 다시 모델에 전달 → 모델이 최종 답변 생성
- 이 "의도 → 실행 → 결과 주입 → 답변" 루프가 핵심

**google-genai에서의 구현**
- `FunctionDeclaration`으로 도구 스키마 정의 → `tools=[...]` 전달
- 응답에서 `function_calls` 파싱 → 함수 실행 → `FunctionResponse`로 결과 주입
- 파이썬 함수를 직접 전달하면 자동 function calling도 가능 (최대 10회)

**LangChain에서의 구현**
- `@tool` 데코레이터: docstring이 도구 설명이 됨
- `model.bind_tools([tool1, tool2])`: 모델에 도구 바인딩
- `ToolMessage`: 실행 결과를 모델에 돌려보내는 메시지 타입

**LangGraph ToolNode**
- `ToolNode`가 도구 실행을 자동화
- 조건부 엣지로 "모델이 도구를 호출했는가?" → Yes면 ToolNode로, No면 END로 분기
- 이것이 에이전트 패턴의 기초

**다중 도구, 병렬 호출**
- 모델이 한 번에 여러 도구를 동시에 요청할 수 있음
- 어떤 도구를 언제 쓸지는 모델이 판단 → 도구 설명(description)의 품질이 중요

---

## 노트북 10. 추론 모델 vs 비추론 모델

### 핵심 메시지
"모든 질문에 '깊은 생각'이 필요하지는 않다. 언제 추론 모델을 쓰고, 언제 비추론으로 충분한지 판단하는 것이 비용 효율의 핵심이다."

### 가르쳐야 할 개념

**추론(Thinking) 모델이란**
- Gemini 2.5 계열은 모두 "thinking" 기능 내장. 답변 전에 내부적으로 추론 과정을 거침
- thinking_budget으로 추론에 쓸 토큰 수를 제어 (0이면 추론 비활성화)
- Gemini 3 계열은 `thinking_level`로 low/medium/high 제어

**thinking token의 의미**
- 모델이 "생각하는 과정"에 소비하는 토큰. 사용자에게 보이지 않지만 과금됨
- thinking 과정을 확인하고 싶으면 `include_thoughts=True`

**언제 추론 모델이 효과적인가**
- 수학·논리 문제, 코드 디버깅, 복잡한 분석 → thinking이 정확도를 크게 올림
- 단순 대화, 분류, 요약 → thinking이 오히려 느리고 비용만 증가

**비용-정확도 트레이드오프**
- thinking_budget=0 (비추론) vs 1024 vs 8192 에서 동일 문제를 풀어 비교
- "이 정도 정확도 향상에 이 정도 추가 비용을 지불할 가치가 있는가?" 판단 프레임워크

**주의사항**
- thinking 모델은 temperature 제약이 있을 수 있음
- system prompt와의 호환성 이슈

---

## 노트북 11. Embedding

### 핵심 메시지
"텍스트를 숫자 벡터로 바꾸면, '의미적으로 비슷한 것'을 수학적으로 찾을 수 있다. 이것이 RAG의 기초 기술이다."

### 가르쳐야 할 개념

**임베딩의 원리**
- 텍스트 → 고차원 벡터 (예: 768차원). 의미가 비슷한 문장은 벡터 공간에서 가까이 위치
- "서울의 날씨"와 "수도 기온"은 단어는 다르지만 벡터가 유사

**Gemini Embedding API**
- google-genai: `client.models.embed_content(model="text-embedding-004", contents="...")`
- LangChain: `GoogleGenerativeAIEmbeddings(model="text-embedding-004")`

**유사도 계산**
- Cosine Similarity를 numpy로 직접 구현
- 문장 간 유사도 매트릭스를 시각화하면 직관적으로 이해

**벡터 스토어**
- 임베딩 벡터를 저장하고 빠르게 검색하는 특수 DB
- Chroma, FAISS 등을 사용하여 `similarity_search()` 수행
- "질의와 가장 유사한 문서 N개를 찾아라" → 이것이 RAG의 Retrieve 단계

**한국어 임베딩**
- 다국어 모델의 한국어 성능 한계
- 한국어 특화 임베딩 모델 존재 (참고)

---

## 노트북 12. RAG + GraphRAG

### 핵심 메시지
"모델이 학습하지 않은 내부 문서를 기반으로 답변하게 하려면 RAG가 필요하다. 그리고 단순 RAG로 답할 수 없는 복잡한 질문에는 GraphRAG가 대안이다."

### 가르쳐야 할 개념

**RAG (Retrieval-Augmented Generation)**

*Retrieve*
- 사용자 질문을 임베딩 → 벡터 스토어에서 유사 문서 청크 검색
- 문서 로딩 (PDF, TXT, 웹) → 청크 분할 (RecursiveCharacterTextSplitter, chunk_size/overlap) → 임베딩 → Chroma 인덱싱

*Augment*
- 검색된 청크를 프롬프트의 context로 주입
- "아래 문서를 참고하여 질문에 답하세요" 패턴

*Generate*
- 모델이 주입된 context 기반으로 답변 생성
- LCEL 체인: `retriever | prompt | model | parser`

*검색 품질 개선*
- MMR (중복 제거), k값 튜닝, 리랭킹 개념

**GraphRAG**

*기존 RAG의 한계*
- 멀티홉 질문 ("A의 상사는 누구이고, 그 사람의 프로젝트는?") → 단일 청크 검색으로 불가능
- 전체 문서 요약 질문 → 하나의 청크에 전체 맥락이 없음

*GraphRAG 접근*
- 문서에서 엔티티(사람, 조직, 개념)와 관계를 추출 → 지식 그래프 구축
- 그래프 클러스터링 → 커뮤니티별 요약 생성
- Local Search (특정 엔티티 중심 탐색) vs Global Search (커뮤니티 요약 기반)

*비교*
- RAG: 구현 간단, 대부분의 질문에 효과적, 비용 낮음
- GraphRAG: 관계 기반 복잡한 질문에 강점, 구축 비용 높음, 그래프 유지보수 필요

---

## 노트북 13. 챗봇 평가 지표 및 방법

### 핵심 메시지
"'잘 되는 것 같다'는 평가가 아니다. 체계적이고 반복 가능한 평가 체계가 있어야 개선할 수 있다."

### 가르쳐야 할 개념

**왜 평가가 어려운가**
- 생성형 AI의 "정답"은 하나가 아님. 같은 질문에 여러 좋은 답변이 가능
- 인간 평가는 비용 높고, 평가자 간 일관성도 떨어짐

**평가 기준 설계**
- 정확성(Correctness): 사실적으로 맞는가?
- 관련성(Relevance): 질문에 대한 답인가?
- 충실도(Faithfulness): 주어진 context에 근거한 답인가? (RAG 특화)
- 유해성(Harmfulness): 위험하거나 부적절한 내용이 있는가?
- 톤(Tone): 챗봇의 페르소나에 맞는가?

**LLM-as-Judge 패턴**
- LLM에게 평가 기준과 채점 루브릭을 주고, 응답을 채점하게 하는 방법
- 장점: 대규모 자동 평가 가능, 비용 효율적
- 단점: Judge 모델 자체의 편향, 자기 응답 선호 등
- Judge 프롬프트 설계: 점수 스케일(1~5), 평가 근거(reasoning)를 함께 출력하도록

**Pairwise 비교**
- 두 응답을 나란히 놓고 "어느 것이 더 나은가" 판정
- A/B 테스트나 모델 교체 시 유용

**RAG 특화 평가 (RAG Triad)**
- Context Relevance: 검색된 문서가 질문과 관련 있는가?
- Answer Faithfulness: 답변이 검색된 문서에 근거하는가?
- Answer Relevance: 답변이 원래 질문에 대한 답인가?

**LangSmith Evaluation**
- Dataset에 테스트 케이스 등록 → Evaluator 정의 → 자동 평가 실행 → 결과 대시보드

---
