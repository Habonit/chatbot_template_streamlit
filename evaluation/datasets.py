"""평가용 데이터셋 정의"""
import json
from pathlib import Path

from langsmith import Client


def _load_test_data(filename: str) -> list:
    """JSON 파일에서 테스트 데이터 로드"""
    data_path = Path(__file__).parent / "test_data" / filename
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def create_general_qa_dataset(client: Client) -> str:
    """일반 QA 데이터셋 생성"""
    dataset_name = "chatbot-general-qa"

    try:
        client.read_dataset(dataset_name=dataset_name)
        return dataset_name
    except Exception:
        pass

    dataset = client.create_dataset(dataset_name=dataset_name)

    test_data = _load_test_data("general_qa.json")
    examples = [
        {
            "inputs": {"question": item["question"]},
            "outputs": {
                "expected_tool": item.get("expected_tool"),
                "unexpected_tools": item.get("unexpected_tools", []),
                "answer_contains": item.get("answer_contains", []),
            },
        }
        for item in test_data
    ]

    client.create_examples(dataset_id=dataset.id, examples=examples)
    return dataset_name


def create_rag_dataset(client: Client) -> str:
    """RAG 평가용 데이터셋 생성

    Note: PDF 업로드 → 임베딩 생성 → 평가 실행의 전체 파이프라인이 필요.
    현재는 placeholder. 실제 PDF 내용에 맞게 test_data/rag_qa.json 작성 필요.
    """
    dataset_name = "chatbot-rag-qa"

    try:
        client.read_dataset(dataset_name=dataset_name)
        return dataset_name
    except Exception:
        pass

    dataset = client.create_dataset(dataset_name=dataset_name)

    test_data = _load_test_data("rag_qa.json")
    if not test_data:
        test_data = [
            {
                "question": "문서에서 주요 내용을 요약해줘",
                "expected_tool": "search_pdf_knowledge",
                "answer_contains": [],
            },
        ]

    examples = [
        {
            "inputs": {"question": item["question"]},
            "outputs": {
                "expected_tool": item.get("expected_tool"),
                "answer_contains": item.get("answer_contains", []),
            },
        }
        for item in test_data
    ]

    client.create_examples(dataset_id=dataset.id, examples=examples)
    return dataset_name
