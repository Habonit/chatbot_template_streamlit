"""평가 시스템 설정"""
import os

# LangSmith 설정 (.env에서 로드)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "gemini-hybrid-chatbot")

# 평가 대상 모델 설정
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_OUTPUT_TOKENS = 8192

# Phase 03-5: thinking 기본 설정 (평가 시)
DEFAULT_THINKING_BUDGET = 0
DEFAULT_SHOW_THOUGHTS = False

# 데이터셋 이름
GENERAL_QA_DATASET = "chatbot-general-qa"
RAG_QA_DATASET = "chatbot-rag-qa"

# 평가 기준
MAX_TOKENS_PER_RESPONSE = 5000
MAX_CONCURRENCY = 2
