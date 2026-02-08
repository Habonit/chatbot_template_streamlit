"""평가 실행 스크립트

사용법:
    python -m evaluation.run_evaluation
"""
import os

from dotenv import load_dotenv

load_dotenv()

from service.react_graph import ReactGraphBuilder
from evaluation.runner import EvaluationRunner
from evaluation.config import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_THINKING_BUDGET,
    DEFAULT_SHOW_THOUGHTS,
)


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY가 설정되지 않았습니다.")
        return

    # 챗봇 초기화 (Phase 03-5 파라미터 포함)
    graph_builder = ReactGraphBuilder(
        api_key=api_key,
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_budget=DEFAULT_THINKING_BUDGET,
        show_thoughts=DEFAULT_SHOW_THOUGHTS,
    )
    graph_builder.build()

    # 평가 실행기 생성
    runner = EvaluationRunner(graph_builder)

    # === 일반 QA 평가 ===
    print("=" * 50)
    print("일반 QA 평가 실행 중...")
    results = runner.run_general_evaluation(
        experiment_prefix="chatbot-v3-general"
    )

    print(f"\n총 테스트: {results['total_examples']}")
    for evaluator, scores in results["evaluator_scores"].items():
        print(f"  {evaluator}: {scores['mean']:.1%} ({scores['count']}개)")

    # === 토큰 효율성 평가 ===
    print("\n" + "=" * 50)
    print("토큰 효율성 평가 실행 중...")

    efficiency_results = runner.run_efficiency_evaluation(
        questions=[
            "안녕하세요",
            "오늘 날씨 어때?",
            "AI의 역사에 대해 자세히 설명해줘",
            "피보나치 수열을 파이썬으로 구현해줘",
        ],
        experiment_prefix="chatbot-v3-efficiency",
    )

    print(f"\n총 테스트: {efficiency_results['total_examples']}")
    for evaluator, scores in efficiency_results["evaluator_scores"].items():
        print(f"  {evaluator}: {scores['mean']:.1%}")


if __name__ == "__main__":
    main()
