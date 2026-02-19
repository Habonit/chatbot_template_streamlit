# Phase 2: 제어 --- 모델 동작을 다루기

> 토큰, 생성 파라미터, 스트리밍으로 모델 동작을 제어한다

## 목표

이 Phase를 마치면 다음을 할 수 있다:

- 토큰의 실체를 이해하고, 토큰 수를 측정하여 비용을 계산할 수 있다
- Temperature, Top-p, Top-k 등 생성 파라미터를 조합하여 출력 특성을 제어할 수 있다
- 스트리밍 방식으로 응답을 수신하고, TTFT를 측정하여 사용자 경험을 최적화할 수 있다

## 개념 관계도

```mermaid
flowchart TD
    A[토큰의 실체] --> B[서브워드 분할]
    B --> C[한국어 vs 영어 효율]
    A --> D["count_tokens() API"]
    D --> E[컨텍스트 윈도우]
    E --> F[입력/출력 토큰 제한]
    A --> G["usage_metadata"]
    G --> H[비용 계산 구조]
    H --> I[Thinking 토큰 비용]
    H --> J[생성 파라미터]
    J --> K[Temperature]
    J --> L["Top-p / Top-k"]
    J --> M[max_output_tokens]
    J --> N["stop_sequences / seed"]
    K --> O[확률 분포 조절]
    L --> O
    O --> P[용도별 조합 전략]
    H --> Q[스트리밍]
    Q --> R[SSE 프로토콜]
    Q --> S["generate_content_stream()"]
    S --> T[청크 구조 분석]
    T --> U[텍스트 누적]
    Q --> V["TTFT vs Total Time"]
    Q --> W["LangChain .stream()"]
    W --> X[AIMessageChunk 병합]
    Q --> Y["비동기 스트리밍 (astream)"]
```

## 포함된 노트

| # | 제목 | 핵심 개념 |
|---|------|-----------|
| 04 | 토큰과 컨텍스트 윈도우 | 서브워드 분할, count_tokens(), 한국어/영어 토큰 효율, 컨텍스트 윈도우, usage_metadata, Thinking 토큰, 비용 계산 |
| 05 | 생성 파라미터 | Temperature, Top-p(Nucleus Sampling), Top-k, 파라미터 조합 전략, max_output_tokens, stop_sequences, seed, LangChain 파라미터 적용 |
| 06 | Streaming | 스트리밍 vs 비스트리밍, generate_content_stream(), 청크 구조, TTFT 측정, AIMessageChunk, LCEL 체인 스트리밍, 비동기 스트리밍 |
