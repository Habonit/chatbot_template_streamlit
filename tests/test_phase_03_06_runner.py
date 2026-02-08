"""Phase 03-6: datasets, runner, run_evaluation 단위 테스트"""
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# === TestLoadTestData ===


class TestLoadTestData:
    """_load_test_data() 테스트"""

    def test_load_general_qa_json(self):
        """general_qa.json 로드 확인"""
        from evaluation.datasets import _load_test_data

        data = _load_test_data("general_qa.json")
        assert isinstance(data, list)
        assert len(data) == 7
        assert data[0]["question"] == "지금 몇 시야?"
        assert data[0]["expected_tool"] == "get_current_time"

    def test_load_nonexistent_file(self):
        """없는 파일 → 빈 리스트"""
        from evaluation.datasets import _load_test_data

        data = _load_test_data("nonexistent.json")
        assert data == []

    def test_all_entries_have_required_fields(self):
        """모든 항목에 필수 필드가 있는지 확인"""
        from evaluation.datasets import _load_test_data

        data = _load_test_data("general_qa.json")
        for item in data:
            assert "question" in item
            assert "expected_tool" in item
            assert "answer_contains" in item


# === TestCreateGeneralQaDataset ===


class TestCreateGeneralQaDataset:
    """create_general_qa_dataset() 테스트"""

    def test_create_dataset_calls_client(self):
        """Client.create_dataset 호출 확인"""
        from evaluation.datasets import create_general_qa_dataset

        mock_client = MagicMock()
        mock_client.read_dataset.side_effect = Exception("not found")
        mock_dataset = MagicMock()
        mock_dataset.id = "test-dataset-id"
        mock_client.create_dataset.return_value = mock_dataset

        result = create_general_qa_dataset(mock_client)

        assert result == "chatbot-general-qa"
        mock_client.create_dataset.assert_called_once_with(
            dataset_name="chatbot-general-qa"
        )
        mock_client.create_examples.assert_called_once()

        # examples 구조 확인
        call_kwargs = mock_client.create_examples.call_args
        examples = call_kwargs[1].get("examples") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs[1]["examples"]
        assert len(examples) == 7
        assert examples[0]["inputs"]["question"] == "지금 몇 시야?"
        assert examples[0]["outputs"]["expected_tool"] == "get_current_time"

    def test_existing_dataset_skipped(self):
        """이미 존재하면 create_dataset 호출 안 함"""
        from evaluation.datasets import create_general_qa_dataset

        mock_client = MagicMock()
        mock_client.read_dataset.return_value = MagicMock()

        result = create_general_qa_dataset(mock_client)

        assert result == "chatbot-general-qa"
        mock_client.create_dataset.assert_not_called()
        mock_client.create_examples.assert_not_called()


# === TestCreateRagDataset ===


class TestCreateRagDataset:
    """create_rag_dataset() 테스트"""

    def test_create_rag_dataset_with_placeholder(self):
        """RAG 데이터셋 placeholder 생성 확인"""
        from evaluation.datasets import create_rag_dataset

        mock_client = MagicMock()
        mock_client.read_dataset.side_effect = Exception("not found")
        mock_dataset = MagicMock()
        mock_dataset.id = "rag-dataset-id"
        mock_client.create_dataset.return_value = mock_dataset

        result = create_rag_dataset(mock_client)

        assert result == "chatbot-rag-qa"
        mock_client.create_dataset.assert_called_once()
        mock_client.create_examples.assert_called_once()

    def test_existing_rag_dataset_skipped(self):
        """이미 존재하면 skip"""
        from evaluation.datasets import create_rag_dataset

        mock_client = MagicMock()
        mock_client.read_dataset.return_value = MagicMock()

        result = create_rag_dataset(mock_client)

        assert result == "chatbot-rag-qa"
        mock_client.create_dataset.assert_not_called()


# === TestEvaluationRunner ===


class TestEvaluationRunner:
    """EvaluationRunner 테스트"""

    def _make_runner(self):
        """테스트용 EvaluationRunner 생성"""
        from evaluation.runner import EvaluationRunner

        mock_graph_builder = MagicMock()
        mock_graph_builder.invoke.return_value = {
            "text": "테스트 응답",
            "tool_history": [],
            "tool_results": {},
            "error": None,
            "total_tokens": 100,
        }

        with patch("evaluation.runner.Client"):
            runner = EvaluationRunner(mock_graph_builder)

        return runner, mock_graph_builder

    def test_target_function_uses_unique_session(self):
        """각 호출마다 고유 session_id 사용"""
        runner, mock_graph = self._make_runner()

        runner._target_function({"question": "질문1"})
        runner._target_function({"question": "질문2"})

        calls = mock_graph.invoke.call_args_list
        assert len(calls) == 2

        session1 = calls[0][1]["session_id"]
        session2 = calls[1][1]["session_id"]

        assert session1.startswith("eval-")
        assert session2.startswith("eval-")
        assert session1 != session2

    def test_target_function_calls_invoke(self):
        """graph_builder.invoke가 올바른 인자로 호출되는지 확인"""
        runner, mock_graph = self._make_runner()

        result = runner._target_function({"question": "테스트 질문"})

        mock_graph.invoke.assert_called_once()
        call_kwargs = mock_graph.invoke.call_args[1]
        assert call_kwargs["user_input"] == "테스트 질문"
        assert call_kwargs["turn_count"] == 1
        assert call_kwargs["session_id"].startswith("eval-")

        assert result["text"] == "테스트 응답"

    def test_target_function_returns_invoke_result(self):
        """_target_function이 invoke 결과를 그대로 반환"""
        runner, mock_graph = self._make_runner()

        result = runner._target_function({"question": "테스트"})

        assert result == mock_graph.invoke.return_value

    def test_summarize_results_structure(self):
        """_summarize_results가 올바른 구조의 dict 반환"""
        runner, _ = self._make_runner()

        # ExperimentResultRow를 시뮬레이션
        mock_row = {
            "run": MagicMock(),
            "example": MagicMock(),
            "evaluation_results": {
                "results": [
                    MagicMock(key="tool_usage_correct", score=1.0),
                    MagicMock(key="response_not_empty", score=1.0),
                ]
            },
        }

        summary = runner._summarize_results([mock_row])

        assert "total_examples" in summary
        assert "evaluator_scores" in summary
        assert summary["total_examples"] == 1
        assert "tool_usage_correct" in summary["evaluator_scores"]
        assert "response_not_empty" in summary["evaluator_scores"]
        assert summary["evaluator_scores"]["tool_usage_correct"]["mean"] == 1.0
        assert summary["evaluator_scores"]["tool_usage_correct"]["count"] == 1

    def test_summarize_results_multiple_rows(self):
        """여러 row에서 평균 계산"""
        runner, _ = self._make_runner()

        rows = [
            {
                "run": MagicMock(),
                "example": MagicMock(),
                "evaluation_results": {
                    "results": [
                        MagicMock(key="no_error", score=1.0),
                    ]
                },
            },
            {
                "run": MagicMock(),
                "example": MagicMock(),
                "evaluation_results": {
                    "results": [
                        MagicMock(key="no_error", score=0.0),
                    ]
                },
            },
        ]

        summary = runner._summarize_results(rows)

        assert summary["total_examples"] == 2
        assert summary["evaluator_scores"]["no_error"]["mean"] == 0.5
        assert summary["evaluator_scores"]["no_error"]["count"] == 2

    def test_summarize_results_empty(self):
        """빈 결과 처리"""
        runner, _ = self._make_runner()

        summary = runner._summarize_results([])

        assert summary["total_examples"] == 0
        assert summary["evaluator_scores"] == {}

    @patch("evaluation.runner.evaluate")
    @patch("evaluation.runner.create_general_qa_dataset")
    def test_run_general_evaluation(self, mock_create_dataset, mock_evaluate):
        """run_general_evaluation이 evaluate()를 올바르게 호출"""
        runner, _ = self._make_runner()
        mock_create_dataset.return_value = "chatbot-general-qa"
        mock_evaluate.return_value = []

        result = runner.run_general_evaluation(experiment_prefix="test-eval")

        mock_create_dataset.assert_called_once_with(runner.client)
        mock_evaluate.assert_called_once()
        call_kwargs = mock_evaluate.call_args[1]
        assert call_kwargs["data"] == "chatbot-general-qa"
        assert call_kwargs["experiment_prefix"] == "test-eval"
        assert len(call_kwargs["evaluators"]) == 4

    @patch("evaluation.runner.evaluate")
    def test_run_efficiency_evaluation(self, mock_evaluate):
        """run_efficiency_evaluation이 evaluate()를 올바르게 호출"""
        runner, _ = self._make_runner()
        mock_evaluate.return_value = []

        questions = ["질문1", "질문2"]
        result = runner.run_efficiency_evaluation(
            questions=questions,
            experiment_prefix="test-efficiency",
        )

        runner.client.create_dataset.assert_called_once()
        mock_evaluate.assert_called_once()
        call_kwargs = mock_evaluate.call_args[1]
        assert call_kwargs["experiment_prefix"] == "test-efficiency"
        assert len(call_kwargs["evaluators"]) == 2


# === TestRunEvaluationScript ===


class TestRunEvaluationScript:
    """run_evaluation 모듈 테스트"""

    def test_run_evaluation_module_importable(self):
        """evaluation.run_evaluation 모듈 import 가능"""
        import evaluation.run_evaluation

        assert hasattr(evaluation.run_evaluation, "main")

    def test_main_without_api_key(self, capsys):
        """GEMINI_API_KEY 없으면 경고 메시지 출력"""
        from evaluation.run_evaluation import main

        with patch.dict("os.environ", {}, clear=False):
            with patch("os.getenv", return_value=None):
                main()

        captured = capsys.readouterr()
        assert "GEMINI_API_KEY" in captured.out
