# 노트북 12. Embedding

> Phase 4 — 지식 확장

텍스트를 숫자 벡터로 바꾸면, "의미적으로 비슷한 것"을 수학적으로 찾을 수 있습니다. 이것이 다음 노트북에서 배울 RAG(검색 증강 생성)의 기초 기술입니다.

**학습 목표**
- 임베딩의 원리와 벡터 공간의 의미를 이해한다
- Gemini Embedding API를 google-genai와 LangChain 두 방식으로 사용할 수 있다
- Cosine Similarity를 직접 구현하고 유사도 매트릭스를 시각화할 수 있다
- Chroma와 FAISS 벡터 스토어를 사용하고 차이를 비교할 수 있다

## 임베딩이란

**Embedding**(임베딩)은 텍스트를 고차원 숫자 벡터로 변환하는 기술입니다. 핵심 아이디어는 단순합니다. 의미가 비슷한 텍스트는 벡터 공간에서 가까이 위치합니다.

```
"서울의 날씨"  →  [0.12, -0.34, 0.56, ...] (768차원)
"수도 기온"    →  [0.11, -0.32, 0.55, ...] (768차원)  ← 벡터가 유사
"파이썬 문법"  →  [0.78, 0.21, -0.43, ...] (768차원)  ← 벡터가 다름
```

단어가 다르더라도 의미가 비슷하면 벡터가 유사합니다. 이 성질을 이용하면 "질문과 가장 관련 있는 문서 찾기"(RAG의 Retrieve), "유사한 상품 추천", "중복 문서 탐지" 같은 작업이 가능합니다.

> 임베딩은 단어가 아닌 **의미**를 포착합니다. "서울의 날씨"와 "수도 기온"처럼 표면적으로 다른 텍스트도 의미가 같으면 높은 유사도를 보이며, 이 성질이 RAG의 검색(Retrieve)을 가능하게 합니다.

## Gemini Embedding API

Gemini는 `text-embedding-004` 모델을 제공합니다. 768차원 벡터를 반환하며, 한국어를 포함한 다국어를 지원합니다.

### google-genai SDK

`embed_content()`에 문자열 하나 또는 리스트를 전달합니다. 리스트를 전달하면 한 번의 API 호출로 여러 텍스트를 임베딩합니다(배치 처리).

```python
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

result = client.models.embed_content(
    model="text-embedding-004",
    contents=["서울의 날씨가 좋습니다", "수도 기온이 따뜻합니다"],
)
embedding = result.embeddings[0].values  # 768차원 float 리스트
```

### task_type: 용도별 임베딩

`config`에 `task_type`을 지정하면 용도에 맞는 임베딩을 생성할 수 있습니다.

| task_type | 용도 |
|-----------|------|
| `RETRIEVAL_QUERY` | 검색 질의 임베딩 |
| `RETRIEVAL_DOCUMENT` | 저장할 문서 임베딩 |
| `SEMANTIC_SIMILARITY` | 문장 유사도 비교 |
| `CLASSIFICATION` | 텍스트 분류 |
| `CLUSTERING` | 클러스터링 |

> 같은 텍스트라도 task_type에 따라 미세하게 다른 벡터가 생성됩니다. RAG에서는 문서 저장 시 `RETRIEVAL_DOCUMENT`, 검색 시 `RETRIEVAL_QUERY`를 사용하면 검색 정확도가 향상될 수 있습니다.

### 출력 차원 조절

`output_dimensionality` 파라미터로 임베딩 벡터의 차원을 줄일 수 있습니다. 256이나 512 차원으로 줄이면 저장 공간과 검색 속도가 개선되지만, 정확도가 약간 감소할 수 있으므로 실제 데이터로 테스트가 필요합니다.

### LangChain 임베딩

**GoogleGenerativeAIEmbeddings**는 LangChain의 통일된 인터페이스를 제공합니다. 벡터 스토어와 연결할 때 이 클래스를 사용합니다.

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY,
)

query_emb = embeddings_model.embed_query("서울의 날씨")        # 문자열 1개
doc_embs = embeddings_model.embed_documents(["문서1", "문서2"])  # 리스트
```

| 메서드 | 용도 | 입력 |
|--------|------|------|
| `embed_query()` | 검색 질의 임베딩 | 문자열 1개 |
| `embed_documents()` | 문서 임베딩 (배치) | 문자열 리스트 |

> `embed_query`와 `embed_documents`를 구분하는 것은 LangChain의 관례입니다. Gemini text-embedding-004는 둘 다 동일한 결과를 반환하지만, 벡터 스토어 호환을 위해 구분하여 사용합니다.

## Cosine Similarity

두 벡터가 얼마나 비슷한지를 측정하는 가장 대표적인 방법이 **Cosine Similarity**(코사인 유사도)입니다. 벡터의 크기(길이)가 아닌 **방향**만 비교하므로, 긴 문장이든 짧은 문장이든 의미가 같으면 유사도가 높습니다.

```
cosine_similarity(A, B) = (A . B) / (||A|| x ||B||)
```

| 값 | 의미 |
|----|------|
| 1.0 | 완전히 같은 방향 (매우 유사) |
| 0.0 | 직교 (관련 없음) |
| -1.0 | 반대 방향 (매우 다름) |

numpy로 직접 구현하면 다음과 같습니다.

```python
import numpy as np

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### 유사도 매트릭스와 시각화

여러 문장 간의 유사도를 한눈에 파악하려면 NxN 매트릭스를 히트맵으로 시각화하는 것이 효과적입니다. 벡터를 정규화한 후 행렬곱을 수행하면 반복문 없이 전체 유사도 매트릭스를 한 번에 계산할 수 있습니다.

```python
def cosine_similarity_matrix(vectors):
    vecs = np.array(vectors)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normalized = vecs / norms
    return normalized @ normalized.T  # NxN 코사인 유사도 매트릭스
```

시각화 결과에서 같은 주제의 문장 쌍은 높은 유사도를, 다른 주제 간에는 낮은 유사도를 보입니다. 이 패턴이 벡터 검색의 근거가 됩니다.

## 벡터 스토어

**Vector Store**(벡터 스토어)는 임베딩 벡터를 저장하고, 질의 벡터와 가장 유사한 벡터를 빠르게 검색하는 특수 데이터베이스입니다. RAG에서의 역할은 다음과 같습니다.

1. 문서들을 임베딩하여 벡터 스토어에 저장
2. 사용자 질문을 임베딩
3. 질문 벡터와 가장 유사한 문서 벡터 N개를 검색
4. 검색된 문서를 LLM에 전달하여 답변 생성

### Chroma vs FAISS 비교

| 항목 | Chroma | FAISS |
|------|--------|-------|
| 개발사 | Chroma Inc. | Meta |
| 저장 방식 | SQLite 백엔드 (영속) | 인메모리 (파일 저장 별도) |
| 영속성 | `persist_directory` 지정으로 자동 저장 | `save_local()` / `load_local()` 수동 호출 |
| 메타데이터 필터링 | 지원 (`filter={"city": "서울"}`) | 미지원 (별도 구현 필요) |
| 검색 속도 | 보통 | 빠름 (대규모 데이터에서 우수) |
| 설치 | `pip install chromadb` | `pip install faiss-cpu` |
| 적합한 규모 | 프로토타입 ~ 중규모 | 대규모 프로덕션 |

두 벡터 스토어 모두 LangChain의 `VectorStore` 인터페이스를 따르므로, `similarity_search()`, `similarity_search_with_score()` 등 동일한 메서드를 사용합니다. 벡터 스토어를 교체해도 코드 변경이 최소화됩니다.

### 기본 사용법: 저장과 검색

```python
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document

documents = [
    Document(page_content="서울은 대한민국의 수도입니다.", metadata={"city": "서울"}),
    Document(page_content="부산은 해운대 해수욕장이 유명합니다.", metadata={"city": "부산"}),
]

# Chroma에 저장 + 검색
chroma_db = Chroma.from_documents(documents=documents, embedding=embedding_func)
results = chroma_db.similarity_search("바다가 있는 관광지", k=2)

# FAISS에 저장 + 검색 (동일 인터페이스)
faiss_db = FAISS.from_documents(documents=documents, embedding=embedding_func)
results = faiss_db.similarity_search("바다가 있는 관광지", k=2)
```

> Chroma의 `similarity_search_with_score`는 **거리**(distance)를 반환합니다. 값이 작을수록 더 유사합니다. 기본 거리 함수는 L2(유클리드 거리)입니다.

### 메타데이터 필터링 (Chroma)

Chroma의 핵심 강점은 벡터 유사도 검색과 메타데이터 필터를 결합할 수 있다는 점입니다.

```python
results = chroma_db.similarity_search(
    "유명한 관광지", k=2,
    filter={"city": {"$in": ["부산", "제주"]}},
)
```

### 문서 추가와 영속화

`from_documents()`는 새 스토어를 생성하고, `add_documents()`는 기존 스토어에 문서를 추가합니다. Chroma는 `persist_directory`를 지정하면 SQLite에 자동 영속화되고, FAISS는 `save_local()` / `load_local()`을 직접 호출해야 합니다.

## Retriever 변환과 검색 전략

벡터 스토어를 RAG 체인에 연결하려면 `as_retriever()`로 **Retriever** 객체로 변환합니다. 이것이 노트북 13에서 배울 RAG 파이프라인의 핵심 연결점입니다.

```python
retriever = chroma_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
docs = retriever.invoke("바다 여행")  # Document 리스트 반환
```

### MMR (Maximal Marginal Relevance)

일반 유사도 검색은 가장 유사한 문서 k개를 반환하지만, 상위 결과가 서로 비슷한 내용일 수 있습니다. **MMR**은 유사도와 다양성의 균형을 맞춰, 서로 다른 관점의 문서를 반환합니다.

| 검색 전략 | 적합한 경우 |
|-----------|------------|
| `similarity` | 가장 관련 있는 문서가 필요할 때 (사실 확인, 정의 질문) |
| `mmr` | 다양한 관점의 문서가 필요할 때 (비교 분석, 종합 보고) |
| `similarity_score_threshold` | 일정 유사도 이상만 필요할 때 (관련 없는 결과 제거) |

> MMR의 `fetch_k`는 내부적으로 가져오는 후보 수이며, k보다 커야 합니다. 후보 중에서 유사도와 다양성을 함께 고려하여 최종 k개를 선택합니다.

## 한국어 임베딩 특성

Gemini의 `text-embedding-004`는 다국어를 지원하지만, 한국어에는 몇 가지 알아두어야 할 특성이 있습니다.

| 특성 | 설명 |
|------|------|
| 토큰 효율 | 한국어는 영어보다 같은 의미를 전달하는 데 더 많은 토큰을 소비 |
| 동의어/동형어 | "사과"(과일) vs "사과"(apologize) — 문맥 없이 구분이 어려움 |
| 조사 영향 | "서울은", "서울이", "서울에서" — 같은 엔티티이지만 다른 표현 |
| 한/영 혼용 | "Python 코딩"과 "파이썬 코딩"의 유사도가 상당히 높음 |

한/영 혼용 문장도 높은 유사도를 보여, 다국어 임베딩 모델이 언어를 넘어 의미를 포착하고 있음을 확인할 수 있습니다.

> 프로덕션에서는 한국어 특화 임베딩 모델(BGE-M3, multilingual-e5 등)을 실제 데이터로 벤치마크한 후 선택하는 것이 좋습니다. 모델 선택 기준은 정확도, 비용, 차원, 생태계 호환성 순입니다.

## 임베딩 비용 고려

| 항목 | 설명 |
|------|------|
| 임베딩 생성 | 문서 저장 시 1회 (저장 후에는 재생성 불필요) |
| 검색 질의 | 매 검색마다 질의 임베딩 1회 (짧은 텍스트이므로 비용 매우 적음) |
| 배치 처리 | 한 번에 여러 텍스트를 임베딩하면 API 호출 횟수 절감 |
| 차원 축소 | `output_dimensionality`로 저장 공간 절감 |

> 비용 최적화의 핵심은 불필요한 재임베딩을 피하는 것입니다. 문서 임베딩은 한 번만 하면 되므로 초기 비용만 발생합니다.

---

## 정리

- **Embedding**(임베딩)은 텍스트를 고차원 벡터로 변환하여, 의미적 유사성을 수학적 거리로 측정할 수 있게 합니다
- Gemini `text-embedding-004`는 768차원 다국어 벡터를 반환하며, google-genai의 `embed_content()`와 LangChain의 `GoogleGenerativeAIEmbeddings` 두 방식으로 사용할 수 있습니다
- **Cosine Similarity**(코사인 유사도)는 벡터의 방향만 비교하여 문장 길이에 무관하게 의미적 유사성을 측정하며, 유사도 매트릭스 시각화로 직관적 확인이 가능합니다
- **Chroma**는 메타데이터 필터링과 자동 영속성으로 프로토타입에 적합하고, **FAISS**는 대규모 데이터에서 검색 속도가 우수하며, 둘 다 LangChain의 동일한 VectorStore 인터페이스를 따릅니다
- 벡터 스토어를 `as_retriever()`로 Retriever로 변환하면 RAG 파이프라인에 바로 연결할 수 있으며, similarity/MMR 등 검색 전략을 상황에 맞게 선택합니다
