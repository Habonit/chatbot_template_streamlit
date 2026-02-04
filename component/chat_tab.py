import streamlit as st
from domain.message import Message


def render_chat_tab(
    on_send: callable,
    messages: list[Message],
) -> None:
    st.header("Chat")

    chat_container = st.container()

    with chat_container:
        for msg in messages:
            role = "user" if msg.role == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

                if msg.role == "assistant" and msg.model_used:
                    with st.expander("Details", expanded=False):
                        st.caption(f"Model: {msg.model_used}")
                        st.caption(f"Tokens: {msg.input_tokens} in / {msg.output_tokens} out")
                        if msg.function_calls:
                            st.markdown("**Tool Calls:**")
                            for fc in msg.function_calls:
                                st.code(f"{fc['name']}({fc.get('args', {})})")

    user_input = st.chat_input("메시지를 입력하세요...")

    if user_input:
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


def render_tool_expander(tool_name: str, args: dict, result: str) -> None:
    with st.expander(f"Tool: {tool_name}", expanded=False):
        st.markdown("**Arguments:**")
        st.json(args)
        st.markdown("**Result:**")
        st.markdown(result)


def render_thinking_expander(thinking: str, model: str) -> None:
    with st.expander(f"Reasoning ({model})", expanded=False):
        st.markdown(thinking)
