import streamlit as st
from domain.message import Message


def format_summary_card(summary_entry: dict) -> str:
    """요약 히스토리 엔트리를 마크다운 카드 형식으로 포맷팅"""
    covers_turns = summary_entry.get("covers_turns", "?")
    summary = summary_entry.get("summary", "")
    return f"**Turn {covers_turns}**\n\n{summary}"


def render_chat_tab(
    on_send: callable,
    messages: list[Message],
    summary_history: list[dict] = None,
) -> None:
    st.header("Chat")

    # 2-Column 레이아웃 (3:1 비율)
    if summary_history is None:
        summary_history = []

    chat_col, summary_col = st.columns([3, 1])

    # 왼쪽 컬럼: 채팅 메시지
    with chat_col:
        chat_container = st.container()

        with chat_container:
            for msg in messages:
                role = "user" if msg.role == "user" else "assistant"
                with st.chat_message(role):
                    st.markdown(msg.content)

                    if msg.role == "assistant":
                        # 툴 사용 정보 Expander (Phase 02)
                        if msg.function_calls or msg.tool_results:
                            with st.expander("🔧 툴 사용 정보", expanded=False):
                                if msg.function_calls:
                                    tool_names = [fc.get("name", "unknown") for fc in msg.function_calls]
                                    st.markdown(f"**선택된 툴:** {', '.join(tool_names)}")
                                    st.divider()

                                if msg.tool_results:
                                    for tool_name, result in msg.tool_results.items():
                                        st.markdown(f"📌 **[{tool_name}]**")
                                        if isinstance(result, dict):
                                            st.json(result)
                                        else:
                                            st.code(str(result), language=None)

                        # 모델 상세 정보 Expander
                        if msg.model_used:
                            with st.expander("Details", expanded=False):
                                st.caption(f"Model: {msg.model_used}")
                                st.caption(f"Tokens: {msg.input_tokens} in / {msg.output_tokens} out")

    # 오른쪽 컬럼: 요약 히스토리
    with summary_col:
        st.markdown("#### 📋 Summary")
        if summary_history:
            for entry in summary_history:
                with st.container(border=True):
                    st.markdown(format_summary_card(entry))
        else:
            st.caption("대화 요약이 3턴마다 생성됩니다.")

    # 채팅 입력창 (컬럼 외부에 배치)
    user_input = st.chat_input("메시지를 입력하세요...")

    if user_input:
        with chat_col:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = on_send(user_input)

                if response:
                    st.markdown(response.get("text", ""))

                    if response.get("tool_calls"):
                        with st.expander("Tool Calls", expanded=False):
                            for tool in response["tool_calls"]:
                                st.json(tool)

                    if response.get("search_results"):
                        with st.expander("Search Results", expanded=False):
                            st.markdown(response["search_results"])