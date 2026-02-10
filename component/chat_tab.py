import streamlit as st
from domain.message import Message
from component.education_tips import (
    get_prompt_education, get_streaming_education,
    get_summary_education, get_thinking_education,
    get_tool_education,
)


def format_summary_card(summary_entry: dict) -> str:
    """요약 히스토리 엔트리를 마크다운 카드 형식으로 포맷팅

    Phase 03-3-2: excluded_turns 표시 지원
    Phase 03-3-3: Turn 1 제외 표시 수정
    """
    turns = summary_entry.get("turns", [])
    excluded = summary_entry.get("excluded_turns", [])
    summarized = summary_entry.get("summarized_turns", turns)  # 실제 요약된 턴

    # 턴 범위 표시: "Turn 1-3" 또는 "Turn 1, 3, 4"
    if turns:
        if len(turns) > 1 and turns == list(range(min(turns), max(turns) + 1)):
            # 연속 범위
            turns_str = f"{min(turns)}-{max(turns)}"
        else:
            turns_str = ", ".join(str(t) for t in turns)
    else:
        turns_str = "?"

    summary = summary_entry.get("summary", "")

    # excluded 턴이 있으면 표시 (casual 턴)
    if excluded:
        excluded_str = f"\n*({', '.join(map(str, sorted(excluded)))}턴 casual)*"
    else:
        excluded_str = ""

    return f"**Turn {turns_str}**{excluded_str}\n\n{summary}"


def _render_welcome_state():
    """메시지가 없을 때 교육적 Welcome 가이드 표시"""
    st.markdown(
        '<div class="welcome-card">'
        "<h3>대화를 시작하세요</h3>"
        "<p>아래 입력창에 질문을 입력하면 AI가 응답합니다.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**예시 질문:**")
    cols = st.columns(3)
    examples = [
        "오늘 날짜가 뭐야?",
        "Python과 JavaScript의 차이점을 비교해줘",
        "최신 AI 트렌드를 검색해줘",
    ]
    for col, example in zip(cols, examples):
        col.markdown(f"- {example}")

    st.caption(
        "동작 원리: 사용자 입력 → 모드 감지 (casual/normal) "
        "→ LLM 호출 + 도구 사용 → 응답 생성"
    )

    # Phase 05: 스트리밍 개념 교육
    streaming_edu = get_streaming_education()
    with st.expander(f"⚡ {streaming_edu['title']}", expanded=False):
        st.caption(streaming_edu["explanation"])
        for term in streaming_edu["terms"]:
            st.caption(f"• **{term['term']}**: {term['desc']}")


def _handle_streaming_response(on_stream: callable, user_input: str) -> dict:
    """스트리밍 응답 처리"""
    response_placeholder = st.empty()
    status_placeholder = st.empty()
    thought_placeholder = st.empty()  # Phase 03-5

    full_response = ""
    full_thought = ""  # Phase 03-5
    tool_calls = []
    metadata = {}

    for chunk in on_stream(user_input):
        chunk_type = chunk.get("type")

        if chunk_type == "token":
            full_response += chunk.get("content", "")
            response_placeholder.markdown(full_response + "▌")

        elif chunk_type == "thought":
            # Phase 03-5: 사고 과정 실시간 표시
            full_thought += chunk.get("content", "")
            thought_placeholder.caption("🧠 사고 중...")

        elif chunk_type == "tool_call":
            tool_name = chunk.get("name", "unknown")
            status_placeholder.caption(f"🔧 {tool_name} 호출 중...")
            tool_calls.append({"name": tool_name, "result": None})

        elif chunk_type == "tool_result":
            tool_name = chunk.get("name")
            for tc in tool_calls:
                if tc["name"] == tool_name and tc["result"] is None:
                    tc["result"] = chunk.get("content", "")
                    break
            status_placeholder.empty()

        elif chunk_type == "done":
            metadata = chunk.get("metadata", {})
            status_placeholder.empty()
            thought_placeholder.empty()

    response_placeholder.markdown(full_response)

    # Phase 03-5: 사고 과정 expander (done 후)
    thought_process = full_thought or metadata.get("thought_process", "")
    if thought_process:
        with st.expander("🧠 모델의 사고 과정", expanded=False):
            st.markdown(thought_process)

    return {
        "text": full_response,
        "tool_calls": tool_calls,
        "tool_results": metadata.get("tool_results", {}),
        "model_used": metadata.get("model_used", ""),
        "summary": metadata.get("summary", ""),
        "summary_history": metadata.get("summary_history", []),
        "input_tokens": metadata.get("input_tokens", 0),
        "output_tokens": metadata.get("output_tokens", 0),
        "normal_turn_ids": metadata.get("normal_turn_ids", []),
        "normal_turn_count": metadata.get("normal_turn_count", 0),
        "thought_process": thought_process,  # Phase 03-5
        # Phase 04: metadata pass-through
        "mode": metadata.get("mode", "normal"),
        "graph_path": metadata.get("graph_path", []),
        "summary_triggered": metadata.get("summary_triggered", False),
        "is_casual": metadata.get("is_casual", False),
        # Phase 05: actual_prompts
        "actual_prompts": metadata.get("actual_prompts", {}),
    }


def _mode_badge(mode: str) -> str:
    """모드별 이모지 + 라벨"""
    badges = {
        "casual": "🟢 casual",
        "normal": "🔵 normal",
    }
    return badges.get(mode, f"⚪ {mode}")


def _format_graph_path(path: list[str]) -> str:
    """그래프 경로를 화살표 형식으로 포맷"""
    if not path:
        return "N/A"
    return " → ".join(path) + " → END"


def _render_turn_metadata(msg: Message) -> None:
    """턴 실행 메타데이터 패널 렌더링"""
    with st.expander("📊 실행 상세", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Turn", msg.turn_id)
        col2.markdown(f"**{_mode_badge(msg.mode)}**")
        col3.caption(f"🤖 {msg.model_used or 'N/A'}")

        st.markdown(f"**Graph Path:** `{_format_graph_path(msg.graph_path)}`")

        if msg.summary_triggered:
            st.markdown("📋 **Summary:** 트리거됨")

        if msg.function_calls:
            names = [fc.get('name', 'unknown') for fc in msg.function_calls]
            st.markdown(f"🔧 **Tools:** {', '.join(names)} ({len(names)}개)")

        if msg.thinking_budget > 0:
            st.markdown(f"🧠 **Thinking:** budget {msg.thinking_budget}")

        st.caption(f"📈 Tokens: {msg.input_tokens} in / {msg.output_tokens} out")

        # Phase 05: 교육 팁
        st.divider()

        # 1. 프롬프트 교육
        prompt_edu = get_prompt_education(msg.actual_prompts)
        if prompt_edu:
            st.markdown(f"**📜 {prompt_edu['title']}**")
            st.caption(prompt_edu["explanation"])
            with st.expander("시스템 프롬프트 보기", expanded=False):
                st.code(prompt_edu["system_prompt_preview"], language=None)

        # 2. 요약/컨텍스트 교육
        summary_edu = get_summary_education(msg.summary_triggered, [])
        if summary_edu:
            st.markdown(f"**🧠 {summary_edu['title']}**")
            st.caption(summary_edu["explanation"])

        # 3. Thinking 교육
        thinking_edu = get_thinking_education(msg.thinking_budget, msg.thought_process)
        if thinking_edu:
            st.markdown(f"**💭 {thinking_edu['title']}**")
            st.caption(thinking_edu["explanation"])

        # 4. 도구 교육
        tool_names = [fc.get("name", "") for fc in msg.function_calls]
        tool_edu = get_tool_education(tool_names)
        if tool_edu:
            st.markdown(f"**🔧 {tool_edu['title']}**")
            for item in tool_edu["explanations"]:
                st.caption(f"• **{item['tool']}**: {item['desc']}")


def render_chat_tab(
    on_send: callable,
    on_stream: callable = None,
    messages: list[Message] = None,
    summary_history: list[dict] = None,
    turn_count: int = None,
    use_streaming: bool = True,
) -> None:
    if messages is None:
        messages = []

    # 턴 수 계산 (전달되지 않은 경우 user 메시지 수로 계산)
    if turn_count is None:
        turn_count = len([m for m in messages if m.role == "user"])

    # 헤더 + 턴 번호 표시
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Chat")
    with col2:
        st.metric("Turn", turn_count)

    # 2-Column 레이아웃 (3:1 비율)
    if summary_history is None:
        summary_history = []

    chat_col, summary_col = st.columns([3, 1])

    # 왼쪽 컬럼: 채팅 메시지
    with chat_col:
        chat_container = st.container()

        with chat_container:
            # Welcome State: 메시지가 없을 때 교육적 가이드 표시
            if not messages:
                _render_welcome_state()
            else:
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

                            # Phase 04: 실행 상세 메타데이터
                            _render_turn_metadata(msg)

    # 오른쪽 컬럼: 요약 히스토리
    with summary_col:
        st.markdown("#### 📋 Summary")
        if summary_history:
            for entry in summary_history:
                with st.container(border=True):
                    st.markdown(format_summary_card(entry))
        else:
            st.caption("대화가 진행되면 3턴마다 자동으로 요약이 생성됩니다. "
                       "요약은 컨텍스트 관리에 사용되어 긴 대화에서도 일관된 응답을 유지합니다.")

    # 채팅 입력창 (컬럼 외부에 배치)
    user_input = st.chat_input("메시지를 입력하세요...")

    if user_input:
        with chat_col:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                if use_streaming and on_stream:
                    response = _handle_streaming_response(on_stream, user_input)
                else:
                    with st.spinner("Thinking..."):
                        response = on_send(user_input)

                if response:
                    # 스트리밍에서는 텍스트 이미 표시됨, fallback만 표시
                    if not (use_streaming and on_stream):
                        st.markdown(response.get("text", ""))

                    # 도구 정보
                    if response.get("tool_calls"):
                        with st.expander("🔧 툴 사용 정보", expanded=False):
                            for tool in response["tool_calls"]:
                                st.markdown(f"**{tool['name']}**")
                                if tool.get("result"):
                                    st.code(tool["result"][:500], language=None)

                    # 모델 상세
                    if response.get("model_used"):
                        with st.expander("Details", expanded=False):
                            st.caption(f"Model: {response['model_used']}")
                            st.caption(f"Tokens: {response.get('input_tokens', 0)} in / {response.get('output_tokens', 0)} out")
