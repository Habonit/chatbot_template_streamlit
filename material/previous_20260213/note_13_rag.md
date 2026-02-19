# 노트북 13. RAG + GraphRAG

> Phase 4 — 지식 확장

모델이 학습하지 않은 내부 문서를 기반으로 답변하게 하려면 RAG가 필요합니다. 그리고 단순 RAG로 답할 수 없는 복잡한 질문에는 GraphRAG가 대안이 됩니다.

**학습 목표**
- RAG(Retrieve → Augment → Generate) 파이프라인을 이해하고 구현할 수 있다
- 문서 로딩, 청크 분할, 벡터 저장, 검색의 전체 흐름을 구성할 수 있다
- LCEL 체인으로 RAG를 구현하고, 검색 품질을 개선할 수 있다
- GraphRAG의 개념을 이해하고, NetworkX로 간단한 지식 그래프를 구축할 수 있다

## RAG란?

**RAG**(Retrieval-Augmented Generation, 검색 증강 생성)는 LLM이 학습하지 않은 외부 문서를 검색하여 답변에 활용하는 기법입니다. 사용자의 질문을 임베딩으로 변환하고, 벡터 스토어에서 유사한 문서를 검색한 뒤, 검색된 문서를 프롬프트에 주입하여 LLM이 문서 기반으로 답변을 생성합니다.

### 외부 지식 활용 방법 비교

| 방법 | 장점 | 단점 |
|------|------|------|
| Fine-tuning | 모델 자체에 지식 내장 | 비용 높음, 데이터 업데이트 어려움 |
| Long Context | 전체 문서를 프롬프트에 삽입 | 토큰 비용 폭발, 길어지면 정확도 하락 |
| **RAG** | 필요한 부분만 검색하여 주입 | 검색 품질에 의존 |

> RAG는 비용 효율적이면서도 최신 정보를 반영할 수 있는 가장 실용적인 방법입니다. 문서가 업데이트되면 벡터 스토어만 갱신하면 됩니다.

## RAG 파이프라인 전체 흐름

RAG 파이프라인은 **인덱싱**(준비, 1회 수행)과 **질의**(실행, 매 질문마다)로 나뉩니다.

```
[인덱싱] 문서 로딩 → 청크 분할 → 임베딩 → 벡터 스토어 저장
[질의]   질문 임베딩 → 유사 청크 검색 → 프롬프트 조합 → LLM 답변
```

### 문서 로딩

LangChain은 다양한 형식의 문서 로더를 제공합니다.

| 로더 | 형식 | 사용법 |
|------|------|--------|
| `TextLoader` | .txt | `TextLoader("file.txt")` |
| `PyPDFLoader` | .pdf | `PyPDFLoader("file.pdf")` |
| `CSVLoader` | .csv | `CSVLoader("file.csv")` |
| `WebBaseLoader` | 웹페이지 | `WebBaseLoader("https://...")` |

### 청크 분할

문서가 길면 전체를 하나의 임베딩으로 만들기 어렵습니다. **청크 분할**(Chunking)은 문서를 적절한 크기로 나누는 과정입니다. **RecursiveCharacterTextSplitter**는 `separators` 순서(`\n\n` → `\n` → `. ` → ` `)대로 분할을 시도하여 단락, 문장 단위의 자연스러운 분할을 수행합니다.

| 파라미터 | 역할 | 권장값 |
|----------|------|--------|
| `chunk_size` | 청크 최대 문자 수 | 200~1000 (도메인에 따라 실험) |
| `chunk_overlap` | 인접 청크 간 겹치는 문자 수 | chunk_size의 10~20% |

> `chunk_overlap`은 청크 경계에서 문맥이 끊기는 것을 방지합니다. 겹치는 부분이 있으면 한 청크에서 놓친 정보를 다음 청크에서 포착할 수 있습니다.

청크 분할 후 임베딩하여 Chroma 벡터 스토어에 저장하면 인덱싱이 완료됩니다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " "],
)
chunks = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_func)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
```

## LCEL RAG 체인

LCEL(LangChain Expression Language)로 RAG 파이프라인을 하나의 체인으로 구성합니다.

| 단계 | 역할 | 입력 → 출력 |
|------|------|------------|
| retriever | 유사 문서 검색 | 질문 → Document 리스트 |
| prompt | 프롬프트 조합 | context + question → 프롬프트 |
| model | LLM 호출 | 프롬프트 → AIMessage |
| parser | 텍스트 추출 | AIMessage → 문자열 |

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

### RAG 프롬프트 설계

RAG 프롬프트에서 가장 중요한 지시는 **"문서에 없는 내용은 모른다고 답하라"**입니다. 이 지시 없이는 LLM이 학습 데이터를 기반으로 답변을 지어내는 **할루시네이션**(Hallucination)이 발생할 수 있습니다.

| 팁 | 예시 |
|-----|------|
| 범위 제한 | "아래 문서만 참고하여 답하세요" |
| 모르면 인정 | "문서에 없으면 '알 수 없습니다'라고 답하세요" |
| 출처 요구 | "어떤 문서를 참고했는지 함께 답하세요" |

### 출처 포함 RAG

`RunnableParallel`을 사용하면 답변과 참조 문서를 동시에 반환할 수 있습니다. 이 패턴은 프로덕션에서 "출처 표시" 기능을 구현할 때 유용합니다.

```python
rag_chain_with_source = RunnableParallel(
    answer=rag_chain,
    source_docs=retriever,
)
# result["answer"], result["source_docs"]로 각각 접근
```

## 검색 품질 개선

RAG의 성능은 검색 품질에 크게 의존합니다. 답변이 잘못되었을 때 검색된 문서를 확인하는 것이 디버깅의 첫 단계입니다.

| 전략 | 설명 | 효과 |
|------|------|------|
| **k 값 조정** | 검색 문서 수 조절 | 너무 적으면 누락, 너무 많으면 노이즈 |
| **MMR** | 다양성 고려 검색 | 중복 문서 방지, 골고루 검색 |
| **청크 크기** | 문서 분할 단위 조절 | 너무 작으면 문맥 손실, 너무 크면 노이즈 |
| **메타데이터 필터** | 카테고리별 필터링 | 관련 없는 문서 제거 |
| **리랭킹** | 검색 결과를 LLM으로 재정렬 | 정확도 향상, 비용 증가 |
| **Multi-Query** | 질문을 여러 관점으로 재구성 | 검색 범위 확대 |

> 일반 유사도 검색(Similarity)은 관련성이 높지만 내용이 비슷한 문서가 중복될 수 있습니다. **MMR**(Maximal Marginal Relevance)은 관련성과 다양성을 동시에 고려하여 서로 다른 문서에서 골고루 검색합니다.

### RAG 평가의 3요소 (RAG Triad)

| 지표 | 질문 |
|------|------|
| **Context Relevance** | 검색된 문서가 질문과 관련 있는가? |
| **Answer Faithfulness** | 답변이 검색된 문서에 근거하는가? |
| **Answer Relevance** | 답변이 원래 질문에 대한 답인가? |

세 지표 모두 높아야 좋은 RAG입니다. 하나라도 낮으면 검색 품질(Retrieve) 또는 생성 품질(Generate)을 개선해야 합니다.

## FAISS로 벡터 스토어 전환

LangChain의 통일된 인터페이스 덕분에 Chroma에서 **FAISS**로 벡터 스토어를 쉽게 전환할 수 있습니다. Retriever만 교체하면 나머지 체인은 동일하게 동작합니다.

```python
from langchain_community.vectorstores import FAISS

faiss_store = FAISS.from_documents(documents=chunks, embedding=embedding_func)
faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 3})
# 이후 rag_chain에서 retriever만 교체
```

## GraphRAG 개요

일반 RAG는 단일 청크의 벡터 유사도 검색에 의존합니다. 다음과 같은 질문 유형에서는 한계가 있습니다.

| 질문 유형 | 일반 RAG의 한계 |
|-----------|----------------|
| **멀티홉**: "김 대리의 팀장이 맡은 프로젝트는?" | 두 청크를 연결해야 함 |
| **전체 요약**: "이 문서의 핵심 주제 3가지는?" | 하나의 청크에 전체 맥락 없음 |
| **관계 추론**: "마케팅팀과 개발팀이 협업한 프로젝트는?" | 관계 정보가 청크에 흩어져 있음 |

**GraphRAG**는 문서에서 **엔티티**(Entity, 사람/조직/개념)와 **관계**(Relation)를 추출하여 **지식 그래프**(Knowledge Graph)를 구축하고, 그래프 탐색으로 context를 구성합니다.

### 엔티티/관계 추출과 그래프 구축

LLM을 사용하여 텍스트에서 엔티티와 관계를 JSON 형태로 추출한 뒤, **NetworkX** 그래프에 노드와 엣지로 저장합니다. 추출 프롬프트에 예시(few-shot)를 포함하거나 Structured Output을 활용하면 정확도가 높아집니다.

```python
import networkx as nx

G = nx.DiGraph()
for entity in graph_data['entities']:
    G.add_node(entity['name'], type=entity['type'])
for rel in graph_data['relations']:
    G.add_edge(rel['source'], rel['target'], relation=rel['relation'])
```

### 그래프 탐색으로 질의 응답

질문과 관련된 엔티티를 찾고, BFS(너비 우선 탐색)로 주변 노드를 탐색하여 context를 구성합니다. 이 context를 LLM에 전달하면 멀티홉 질문에도 답할 수 있습니다. 예를 들어 "김철수 → 협업 → 정하늘 → 프로젝트 베타"로 2단계를 거쳐야 하는 질문도 그래프 탐색으로 해결됩니다.

### Local Search vs Global Search

| 전략 | 방식 | 적합한 질문 |
|------|------|------------|
| **Local Search** | 특정 엔티티 중심 탐색 | "김철수의 역할은?" |
| **Global Search** | 전체 그래프 커뮤니티 요약 | "이 조직의 프로젝트 현황은?" |

Local Search는 특정 엔티티에서 출발하여 관련 노드를 탐색합니다. Global Search는 그래프를 커뮤니티(클러스터)로 분할하고, 각 커뮤니티의 요약을 LLM으로 생성한 뒤 질문에 관련된 요약을 context로 활용합니다.

### 그래프 저장소: NetworkX vs Neo4j

| 항목 | NetworkX | Neo4j |
|------|----------|-------|
| 성격 | 순수 Python 라이브러리 | 그래프 데이터베이스 |
| 설치 | `pip install networkx` | 별도 서버 설치 필요 |
| 영속성 | 없음 (인메모리) | 완전 영속성 |
| 대규모 탐색 | 한계 있음 | Cypher 쿼리로 고성능 탐색 |
| 적합한 경우 | 교육, 프로토타입 | 프로덕션 |

> 프로덕션에서는 Neo4j가 사실상 표준입니다. Neo4j AuraDB(클라우드) + LangChain `Neo4jGraph` 통합으로 확장할 수 있습니다.

## RAG vs GraphRAG 비교

| 비교 항목 | RAG | GraphRAG |
|-----------|-----|----------|
| 검색 방식 | 벡터 유사도 | 그래프 탐색 |
| 강점 | 단순 질문, 사실 확인 | 관계 추론, 멀티홉 |
| 약점 | 관계 추론 어려움 | 구축 비용 높음 |
| 구현 난이도 | 낮음 | 높음 (엔티티/관계 추출 필요) |
| 비용 | 임베딩 비용만 | 추출용 LLM 호출 + 그래프 유지보수 |
| 업데이트 | 문서 추가/삭제 쉬움 | 그래프 재구축 필요 |
| 적합한 질문 | FAQ, 문서 내 사실 확인 | 조직도, 관계 분석, 멀티홉 |

> 대부분의 경우 RAG로 충분합니다. GraphRAG는 관계 기반 질문이 핵심인 도메인(조직 관리, 법률, 의학)에서 고려하세요.

### 실무 의사결정 가이드

```
외부 문서 기반 답변이 필요한가?
  ├─ No → LLM 직접 호출
  └─ Yes → 질문이 관계 추론을 포함하는가?
       ├─ No → RAG
       └─ Yes → GraphRAG 검토
              └─ 데이터 규모가 큰가?
                   ├─ No → NetworkX (프로토타입)
                   └─ Yes → Neo4j (프로덕션)
```

---

## 정리

- RAG는 Retrieve(검색) → Augment(주입) → Generate(생성) 3단계로 LLM에 외부 지식을 제공하는 가장 실용적인 방법이다
- 인덱싱 파이프라인(문서 로딩 → 청크 분할 → 임베딩 → 벡터 스토어)은 1회 구축하고, 질의 파이프라인은 LCEL 체인(`retriever | prompt | model | parser`)으로 간결하게 구성할 수 있다
- 검색 품질이 RAG 전체 성능을 좌우하므로, k값 조정, MMR, 청크 크기 실험, 리랭킹 등으로 지속적으로 개선해야 한다
- GraphRAG는 엔티티와 관계를 지식 그래프로 구축하여, 일반 RAG로는 답하기 어려운 멀티홉/관계 추론 질문을 해결한다
- 대부분의 경우 RAG로 충분하며, GraphRAG는 관계 기반 질문이 핵심인 도메인에서 선택적으로 도입한다
