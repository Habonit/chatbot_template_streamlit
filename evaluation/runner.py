"""평가 실행기"""
import uuid

from langsmith import Client
from langsmith.evaluation import evaluate

from evaluation.config import MAX_CONCURRENCY
from evaluation.datasets import create_general_qa_dataset
from evaluation.evaluators import (
    tool_usage_correct,
    answer_contains_keywords,
    response_not_empty,
    no_error,
    token_efficiency,
)


class EvaluationRunner:
    """LangSmith 평가 실행기"""

    def __init__(self, graph_builder):
        """
        Args:
            graph_builder: ReactGraphBuilder 인스턴스 (build() 완료 상태)
        """
        self.client = Client()
        self.graph_builder = graph_builder

    def _target_function(self, inputs: dict) -> dict:
        """평가 대상 함수 — 각 호출마다 독립 세션 사용

        세션 격리: uuid 기반 고유 session_id로 평가 간 독립성 보장.
        turn_count=1: 항상 첫 턴으로 실행 (요약 트리거 방지).
        """
        question = inputs.get("question", "")
        unique_session = f"eval-{uuid.uuid4().hex[:8]}"

        result = self.graph_builder.invoke(
            user_input=question,
            session_id=unique_session,
            turn_count=1,
        )
        return result

    def run_general_evaluation(
        self,
        experiment_prefix: str = "chatbot-eval",
        max_concurrency: int = None,
    ) -> dict:
        """일반 QA 평가 실행"""
        if max_concurrency is None:
            max_concurrency = MAX_CONCURRENCY

        dataset_name = create_general_qa_dataset(self.client)

        results = evaluate(
            self._target_function,
            data=dataset_name,
            evaluators=[
                tool_usage_correct,
                answer_contains_keywords,
                response_not_empty,
                no_error,
            ],
            experiment_prefix=experiment_prefix,
            max_concurrency=max_concurrency,
        )

        return self._summarize_results(results)

    def run_efficiency_evaluation(
        self,
        questions: list[str],
        experiment_prefix: str = "chatbot-efficiency",
    ) -> dict:
        """토큰 효율성 평가"""
        dataset_name = f"{experiment_prefix}-dataset"

        try:
            self.client.delete_dataset(dataset_name=dataset_name)
        except Exception:
            pass

        dataset = self.client.create_dataset(dataset_name=dataset_name)

        examples = [
            {"inputs": {"question": q}, "outputs": {}}
            for q in questions
        ]
        self.client.create_examples(dataset_id=dataset.id, examples=examples)

        results = evaluate(
            self._target_function,
            data=dataset_name,
            evaluators=[
                token_efficiency,
                no_error,
            ],
            experiment_prefix=experiment_prefix,
        )

        return self._summarize_results(results)

    def _summarize_results(self, results) -> dict:
        """평가 결과 요약

        ExperimentResults를 이터레이션하여 결과 집계.
        ExperimentResultRow는 TypedDict: {run, example, evaluation_results}
        evaluation_results는 TypedDict: {results: list[EvaluationResult]}
        EvaluationResult는 Pydantic model: key, score, comment 등
        """
        results_list = list(results)

        summary = {
            "total_examples": len(results_list),
            "evaluator_scores": {},
        }

        for r in results_list:
            # ExperimentResultRow는 TypedDict → dict 접근
            if isinstance(r, dict):
                eval_results = r.get("evaluation_results", {})
                result_items = eval_results.get("results", [])
            else:
                eval_results = getattr(r, "evaluation_results", None)
                if hasattr(eval_results, "results"):
                    result_items = eval_results.results
                else:
                    result_items = []

            for er in result_items:
                key = getattr(er, "key", None) or (
                    er.get("key") if isinstance(er, dict) else "unknown"
                )
                score = getattr(er, "score", None)
                if score is None and isinstance(er, dict):
                    score = er.get("score", 0)

                if key not in summary["evaluator_scores"]:
                    summary["evaluator_scores"][key] = []
                if score is not None:
                    summary["evaluator_scores"][key].append(score)

        # 평균 계산
        for key, scores in summary["evaluator_scores"].items():
            summary["evaluator_scores"][key] = {
                "mean": round(sum(scores) / len(scores), 3) if scores else 0,
                "count": len(scores),
            }

        return summary
