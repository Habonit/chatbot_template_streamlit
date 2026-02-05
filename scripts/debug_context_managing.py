"""Phase 03-3: Context Managing 디버그 스크립트

LangSmith + 콘솔 출력으로 Context 구성 확인
"""
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("❌ GEMINI_API_KEY 환경 변수를 설정하세요")
    exit(1)

from service.react_graph import ReactGraphBuilder, extract_last_n_turns, extract_current_turn

def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def print_messages(messages: list, label: str):
    print(f"\n📋 {label} ({len(messages)}개):")
    for i, msg in enumerate(messages):
        role = msg.__class__.__name__
        content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"   [{i}] {role}: {content}")

def main():
    builder = ReactGraphBuilder(api_key=API_KEY, db_path=":memory:")
    builder.build()

    # 시뮬레이션할 대화 히스토리
    conversation_history = []
    summary_history = []

    # Turn 1-7 시뮬레이션
    turns = [
        "안녕하세요! 오늘 날씨 어때요?",
        "서울의 인구가 얼마나 되나요?",
        "파이썬으로 웹 크롤링하는 방법 알려주세요",
        "LangChain이 뭔가요?",
        "React와 Vue 중 뭐가 좋나요?",
        "Docker 컨테이너 만드는 법 알려줘",
        "마지막으로 오늘 할 일 정리해줘",
    ]

    for turn_num, user_input in enumerate(turns, 1):
        print_separator(f"Turn {turn_num}")
        print(f"👤 User: {user_input}")

        # Context 구성 시뮬레이션
        if turn_num > 1:
            # 이전 턴들 계산
            summarized_turns = len(summary_history) * 3
            unsummarized_start = summarized_turns + 1
            unsummarized_count = turn_num - 1 - summarized_turns

            print(f"\n📊 Context 구성:")
            print(f"   - 요약된 턴: 1~{summarized_turns} ({len(summary_history)}개 요약)")
            print(f"   - Raw 턴: {unsummarized_start}~{turn_num-1} ({unsummarized_count}개)")
            print(f"   - 현재 턴: {turn_num}")

            # extract 함수 테스트
            raw_turns = extract_last_n_turns(conversation_history, n=unsummarized_count)
            print_messages(raw_turns, "Raw Turns (unsummarized)")

        # 실제 invoke 실행
        result = builder.invoke(
            user_input=user_input,
            session_id=f"debug_session_{turn_num}",
            messages=conversation_history.copy(),
            turn_count=turn_num,
            compression_rate=0.3,
            summary_history=summary_history.copy(),
        )

        # 대화 히스토리 업데이트
        conversation_history.append(HumanMessage(content=user_input))
        conversation_history.append(AIMessage(content=result.get("text", ""), tool_calls=[]))

        # summary_history 업데이트
        if result.get("summary_history"):
            summary_history = result["summary_history"]

        # 결과 출력
        print(f"\n🤖 Assistant: {result.get('text', '')[:100]}...")

        if result.get("summary_history"):
            print(f"\n📝 Summary History ({len(summary_history)}개):")
            for i, sh in enumerate(summary_history):
                print(f"   [{i}] Turns {sh['turns']}: {sh['summary'][:50]}...")
                print(f"       original: {sh['original_chars']}자 → summary: {sh['summary_chars']}자 (rate: {sh['compression_rate']})")

        print(f"\n💰 Tokens: input={result.get('input_tokens', 0)}, output={result.get('output_tokens', 0)}")

        # 잠시 대기 (API rate limit 고려)
        if turn_num < len(turns):
            import time
            time.sleep(1)

    print_separator("최종 상태")
    print(f"📋 총 메시지: {len(conversation_history)}개")
    print(f"📝 총 요약: {len(summary_history)}개")
    print("\n✅ LangSmith 대시보드에서 상세 트레이스 확인 가능")
    print("   https://smith.langchain.com/")

if __name__ == "__main__":
    main()
