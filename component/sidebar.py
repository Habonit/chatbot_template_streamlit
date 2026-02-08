import streamlit as st
import csv
import io
import os
from domain.session import Session
from component.education_tips import get_parameter_help


def _generate_csv_data(messages: list) -> bytes:
    """대화 내역을 CSV 형식으로 변환 (UTF-8 BOM 포함)"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["turn_id", "role", "content", "model_used", "input_tokens", "output_tokens"])

    for msg in messages:
        writer.writerow([
            msg.turn_id,
            msg.role,
            msg.content,
            getattr(msg, "model_used", ""),
            getattr(msg, "input_tokens", ""),
            getattr(msg, "output_tokens", ""),
        ])

    return output.getvalue().encode("utf-8-sig")


def _render_token_usage():
    """토큰 사용량을 progress bar로 시각화"""
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {"input": 0, "output": 0, "total": 0}

    usage = st.session_state.token_usage
    token_limit_k = int(os.getenv("TOKEN_LIMIT_K", "256"))
    token_limit = token_limit_k * 1000

    st.sidebar.markdown("### Token Usage")

    total = usage["total"]
    progress = min(total / token_limit, 1.0) if token_limit > 0 else 0.0
    percent = int(progress * 100)

    st.sidebar.progress(progress, text=f"{total:,} / {token_limit:,} ({percent}%)")

    col1, col2 = st.sidebar.columns(2)
    col1.caption(f"입력: {usage['input']:,}")
    col2.caption(f"출력: {usage['output']:,}")


def render_sidebar() -> dict:
    st.sidebar.title("Settings")

    # === API Keys ===
    with st.sidebar.expander("API Keys", expanded=True):
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            key="gemini_api_key",
            help="Google AI Studio API Key",
        )
        # Gemini API Key 피드백
        if gemini_key:
            if gemini_key.startswith("AIza") and len(gemini_key) >= 39:
                st.caption("✓ Gemini API Key 형식 확인됨")
            else:
                st.caption("⚠ API Key 형식이 올바르지 않을 수 있습니다")

        tavily_key = st.text_input(
            "Tavily API Key",
            type="password",
            key="tavily_api_key",
            help="Tavily Search API Key",
        )
        # Tavily API Key 피드백
        if tavily_key:
            if tavily_key.startswith("tvly-"):
                st.caption("✓ Tavily API Key 형식 확인됨")
            else:
                st.caption("⚠ API Key 형식이 올바르지 않을 수 있습니다")

    st.sidebar.divider()

    # === Model & Reasoning ===
    with st.sidebar.expander("Model & Reasoning", expanded=True):
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
            help=get_parameter_help("temperature"),
        )

        top_p = st.slider(
            "Top-p",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help=get_parameter_help("top_p"),
        )

        max_output_tokens = st.slider(
            "Max Output Tokens",
            min_value=256,
            max_value=65536,
            value=8192,
            step=256,
            help=get_parameter_help("max_output_tokens"),
        )

        # Phase 03-1: seed 파라미터 추가
        seed = st.number_input(
            "Seed (재현성)",
            min_value=-1,
            max_value=2147483647,
            value=-1,
            step=1,
            help=get_parameter_help("seed"),
        )

        st.divider()

        # Phase 02-7: 추론 모드 설정
        reasoning_mode = st.toggle(
            "추론 모드 (Reasoning Mode)",
            value=False,
            help="복잡한 추론이 필요한 질문에 thinking 활성화",
        )

        # Phase 03-5: thinking 설정
        if reasoning_mode:
            st.caption("Thinking 활성화됨 — 복잡한 추론 질문에 사고 과정을 사용합니다.")
            thinking_budget = st.slider(
                "Thinking Budget",
                min_value=0,
                max_value=8192,
                value=1024,
                step=128,
                help=get_parameter_help("thinking_budget"),
            )

            show_thoughts = st.toggle(
                "추론 과정 표시",
                value=False,
                help="모델의 사고 과정을 UI에 표시",
            )

            st.caption(f"📊 Thinking budget: {thinking_budget} tokens")
        else:
            thinking_budget = 0
            show_thoughts = False

    st.sidebar.divider()

    # === Search ===
    with st.sidebar.expander("Search", expanded=False):
        search_enabled = st.toggle("Enable Tavily Search", value=True)
        search_depth = st.selectbox(
            "Search Depth",
            options=["basic", "advanced"],
            index=0,
            help="basic: 빠른 검색, advanced: 심층 검색 (더 많은 크레딧 소모)",
        )
        max_results = st.slider(
            "Max Results",
            min_value=1,
            max_value=10,
            value=5,
            help="검색 결과 최대 개수",
        )

    st.sidebar.divider()

    # === Agent ===
    with st.sidebar.expander("Agent", expanded=False):
        max_iterations = st.slider(
            "Max Tool Iterations",
            min_value=1,
            max_value=10,
            value=5,
            help="ReAct 에이전트가 툴을 호출할 수 있는 최대 횟수",
        )

        # Phase 03-3: 요약 압축률 설정
        compression_rate = st.slider(
            "요약 압축률",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05,
            help=get_parameter_help("compression_rate"),
        )

    st.sidebar.divider()

    # === Session ===
    with st.sidebar.expander("Session", expanded=False):
        if "sessions" not in st.session_state:
            st.session_state.sessions = []

        session_options = st.session_state.sessions if st.session_state.sessions else []

        # 현재 세션이 목록에 없으면 추가
        current = st.session_state.get("current_session", "")
        if current and current not in session_options:
            session_options = [current] + session_options

        # 세션 선택
        if session_options:
            current_index = session_options.index(current) if current in session_options else 0
            selected_session = st.selectbox(
                "Select Session",
                options=session_options,
                index=current_index,
                key="session_selector",
            )

            # 세션이 변경되었는지 감지
            if selected_session != st.session_state.get("current_session"):
                st.session_state.session_changed = True
                st.session_state.current_session = selected_session
                st.rerun()
        else:
            st.caption("세션이 없습니다. 아래 버튼으로 새 세션을 생성하세요.")

        # 새 세션 생성 버튼
        if st.button("Create New Session"):
            new_id = Session.generate_id()
            if new_id not in st.session_state.sessions:
                st.session_state.sessions.append(new_id)
            st.session_state.session_changed = True
            st.session_state.new_session_created = True
            st.session_state.current_session = new_id
            st.rerun()

        # 대화 내역 CSV 다운로드
        if st.session_state.get("messages"):
            csv_data = _generate_csv_data(st.session_state.messages)
            session_id = st.session_state.get("current_session", "session")
            st.download_button(
                label="📥 대화 내역 다운로드 (CSV)",
                data=csv_data,
                file_name=f"conversation_{session_id}.csv",
                mime="text/csv",
            )

    st.sidebar.divider()

    # === Token Usage (progress bar) ===
    _render_token_usage()

    return {
        "gemini_api_key": gemini_key,
        "tavily_api_key": tavily_key,
        "model": model,
        "embedding_model": embedding_model,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "seed": seed if seed >= 0 else None,  # Phase 03-1: -1은 None (랜덤)
        "search_enabled": search_enabled,
        "search_depth": search_depth,
        "max_results": max_results,
        "max_iterations": max_iterations,
        "session_id": st.session_state.get("current_session", ""),
        # Phase 02-7: 추론 모드 설정
        "reasoning_mode": reasoning_mode,
        # Phase 03-3: 요약 압축률
        "compression_rate": compression_rate,
        # Phase 03-5: thinking 설정
        "thinking_budget": thinking_budget,
        "show_thoughts": show_thoughts,
    }
