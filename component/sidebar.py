import streamlit as st
from domain.session import Session


def render_sidebar() -> dict:
    st.sidebar.title("Settings")

    with st.sidebar.expander("API Keys", expanded=True):
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            key="gemini_api_key",
            help="Google AI Studio API Key",
        )
        tavily_key = st.text_input(
            "Tavily API Key",
            type="password",
            key="tavily_api_key",
            help="Tavily Search API Key",
        )

    st.sidebar.divider()

    with st.sidebar.expander("Model Settings", expanded=True):
        model = st.selectbox(
            "Chat Model",
            options=[
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash",
            ],
            index=0,
            help="gemini-2.0-flash: 2026년 3월 종료 예정",
        )

        embedding_model = st.selectbox(
            "Embedding Model",
            options=["gemini-embedding-001"],
            index=0,
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
        )

        top_p = st.slider(
            "Top-p",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
        )

    st.sidebar.divider()

    with st.sidebar.expander("External Search", expanded=False):
        search_enabled = st.toggle("Enable Tavily Search", value=True)
        search_depth = st.selectbox(
            "Search Depth",
            options=["basic", "advanced"],
            index=0,
        )
        max_results = st.slider(
            "Max Results",
            min_value=1,
            max_value=10,
            value=5,
        )

    st.sidebar.divider()

    with st.sidebar.expander("Session", expanded=False):
        if "sessions" not in st.session_state:
            st.session_state.sessions = []

        session_options = ["New Session"] + st.session_state.sessions
        selected_session = st.selectbox(
            "Select Session",
            options=session_options,
            index=0,
        )

        if selected_session == "New Session":
            if st.button("Create New Session"):
                new_id = Session.generate_id()
                st.session_state.sessions.append(new_id)
                st.session_state.current_session = new_id
                st.rerun()

    st.sidebar.divider()

    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {"input": 0, "output": 0, "total": 0}

    usage = st.session_state.token_usage
    st.sidebar.markdown("### Token Usage")
    st.sidebar.markdown(f"""
    입력: **{usage['input']:,}** tokens
    출력: **{usage['output']:,}** tokens
    총계: **{usage['total']:,}** tokens
    """)

    return {
        "gemini_api_key": gemini_key,
        "tavily_api_key": tavily_key,
        "model": model,
        "embedding_model": embedding_model,
        "temperature": temperature,
        "top_p": top_p,
        "search_enabled": search_enabled,
        "search_depth": search_depth,
        "max_results": max_results,
        "session_id": st.session_state.get("current_session", ""),
    }
