import streamlit as st
import csv
import io
from domain.session import Session


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


def render_sidebar() -> dict:
    st.sidebar.title("Settings")

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

        max_output_tokens = st.slider(
            "Max Output Tokens",
            min_value=256,
            max_value=65536,
            value=8192,
            step=256,
            help="최대 출력 토큰 수 (Gemini 2.5: 최대 65,536)",
        )

        # Phase 03-1: seed 파라미터 추가
        seed = st.number_input(
            "Seed (재현성)",
            min_value=-1,
            max_value=2147483647,
            value=-1,
            step=1,
            help="응답 재현성 제어. -1은 랜덤, 양수는 고정 시드",
        )

        st.divider()

        # Phase 02-7: 추론 모드 설정
        reasoning_mode = st.toggle(
            "추론 모드 (Reasoning Mode)",
            value=False,
            help="복잡한 추론이 필요한 질문에 thinking 활성화",
        )

        auto_reasoning = st.toggle(
            "자동 추론 모드 감지",
            value=True,
            help="질문 유형에 따라 자동으로 추론 모드 활성화",
        )

        # Phase 03-5: thinking 설정
        if reasoning_mode:
            thinking_budget = st.slider(
                "Thinking Budget",
                min_value=0,
                max_value=8192,
                value=1024,
                step=128,
                help="추론에 사용할 토큰 예산 (0: 비활성화, 128+: 활성화)",
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

    with st.sidebar.expander("Agent Settings", expanded=False):
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
            help="낮을수록 짧게 요약, 높을수록 상세하게 요약 (3턴마다 적용)",
        )

    st.sidebar.divider()

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
            st.info("No sessions yet. Create a new one.")

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
        "max_output_tokens": max_output_tokens,
        "seed": seed if seed >= 0 else None,  # Phase 03-1: -1은 None (랜덤)
        "search_enabled": search_enabled,
        "search_depth": search_depth,
        "max_results": max_results,
        "max_iterations": max_iterations,
        "session_id": st.session_state.get("current_session", ""),
        # Phase 02-7: 추론 모드 설정
        "reasoning_mode": reasoning_mode,
        "auto_reasoning": auto_reasoning,
        # Phase 03-3: 요약 압축률
        "compression_rate": compression_rate,
        # Phase 03-5: thinking 설정
        "thinking_budget": thinking_budget,
        "show_thoughts": show_thoughts,
    }
