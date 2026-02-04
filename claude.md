# Development Guidelines

## 개발 원칙

1. **단계별 진행**: 한 번에 하나의 모듈만 구현. 완료 후 다음 단계로 이동.
2. **TDD 필수**: 테스트 코드 먼저 작성 → 구현 → 리팩토링 순서 준수.
3. **문서 참조**: `doc/phase_01/_01.md` 스펙을 기준으로 구현.

## 금지 사항

- **Git 작업 절대 금지**: commit, push, branch, merge 등 모든 git 명령어 사용 금지.

## 개발 순서

1. `domain/` - 데이터 모델 정의 + 테스트
2. `repository/` - 저장소 로직 + 테스트
3. `service/` - 비즈니스 로직 + 테스트
4. `component/` - UI 컴포넌트
5. `app.py` - 통합

## TDD 사이클

```
1. RED: 실패하는 테스트 작성
2. GREEN: 테스트 통과하는 최소 코드 작성
3. REFACTOR: 코드 정리
```

## 테스트 실행

```bash
uv run pytest tests/ -v
```
