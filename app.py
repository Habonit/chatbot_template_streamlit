"""Phase 02-6: SQLite 단일 저장소로 통합된 Streamlit 앱"""
import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv

from domain.message import Message
from domain.session import Session
from repository.embedding_repo import EmbeddingRepository
from repository.pdf_extractor import PDFExtractor
from service.llm_service import LLMService
from service.embedding_service import EmbeddingService
from service.rag_service import RAGService
from service.react_graph import ReactGraphBuilder
from service.session_manager import SessionManager
from component.sidebar import render_sidebar
from component.chat_tab import render_chat_tab
from component.pdf_tab import render_pdf_tab
from component.overview_tab import render_overview_tab
from component.prompts_tab import render_prompts_tab
from component.styles import inject_custom_css

load_dotenv()

st.set_page_config(
    page_title="Gemini Hybrid Chatbot",
    page_icon="🤖",
    layout="wide",
)

DATA_PATH = Path("data/sessions")
UPLOAD_PATH = Path("data/uploads/temp")
DB_PATH = "data/langgraph.db"
DATA_PATH.mkdir(parents=True, exist_ok=True)
UPLOAD_PATH.mkdir(parents=True, exist_ok=True)

TOKEN_LIMIT_K = int(os.getenv("TOKEN_LIMIT_K", "256"))
TOKEN_LIMIT = TOKEN_LIMIT_K * 1000


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_session" not in st.session_state:
        st.session_state.current_session = Session.generate_id()
    if "sessions" not in st.session_state:
        st.session_state.sessions = [st.session_state.current_session]
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {"input": 0, "output": 0, "total": 0}
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "summary_history" not in st.session_state:
        st.session_state.summary_history = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "pdf_description" not in st.session_state:
        st.session_state.pdf_description = ""
    # Phase 03-3-2: normal 턴 ID 목록
    if "normal_turn_ids" not in st.session_state:
        st.session_state.normal_turn_ids = []


def get_turn_count(messages: list[Message]) -> int:
    return len([m for m in messages if m.role == "user"])


def _create_graph_builder(
    settings: dict,
    embed_repo: EmbeddingRepository,
) -> ReactGraphBuilder:
    """handle_chat_message와 handle_stream_message 공통 팩토리"""
    search_service = None
    embedding_service = None

    if settings.get("tavily_api_key"):
        from service.search_service import SearchService
        search_service = SearchService(api_key=settings["tavily_api_key"])

    if st.session_state.chunks:
        embedding_service = EmbeddingService(
            api_key=settings["gemini_api_key"],
            model=settings.get("embedding_model", "gemini-embedding-001"),
        )

    return ReactGraphBuilder(
        api_key=settings["gemini_api_key"],
        model=settings.get("model", "gemini-2.0-flash"),
        temperature=settings.get("temperature", 0.7),
        top_p=settings.get("top_p", 0.9),
        max_output_tokens=settings.get("max_output_tokens", 8192),
        seed=settings.get("seed"),
        max_iterations=settings.get("max_iterations", 5),
        thinking_budget=settings.get("thinking_budget", 0),
        show_thoughts=settings.get("show_thoughts", False),
        search_service=search_service,
        embedding_service=embedding_service,
        embedding_repo=embed_repo if st.session_state.chunks else None,
        db_path=DB_PATH,
        search_depth=settings.get("search_depth", "basic"),
        max_results=settings.get("max_results", 5),
    )


def handle_chat_message(
    user_input: str,
    settings: dict,
    embed_repo: EmbeddingRepository,
) -> dict:
    """LangGraph 기반 채팅 메시지 처리 (Phase 02-6: SqliteSaver가 자동 저장)"""
    if not settings.get("gemini_api_key"):
        return {"text": "Gemini API Key를 입력해주세요.", "error": True}

    session_id = st.session_state.current_session

    # 토큰 제한 체크
    current_tokens = st.session_state.token_usage["total"]
    if current_tokens >= TOKEN_LIMIT:
        return {"text": f"토큰 제한({TOKEN_LIMIT_K}k)을 초과했습니다. 새 세션을 시작해주세요.", "error": True}
    if current_tokens >= TOKEN_LIMIT * 0.8:
        st.warning(f"토큰 사용량이 80%를 초과했습니다 ({current_tokens:,}/{TOKEN_LIMIT:,})")

    turn_count = get_turn_count(st.session_state.messages) + 1

    # 사용자 메시지를 st.session_state에 저장 (UI 표시용)
    user_msg = Message(turn_id=turn_count, role="user", content=user_input)
    st.session_state.messages.append(user_msg)

    graph_builder = _create_graph_builder(settings, embed_repo)
    result = graph_builder.invoke(
        user_input=user_input,
        session_id=session_id,
        messages=st.session_state.messages[:-1],  # 현재 사용자 메시지 제외
        summary=st.session_state.summary,
        pdf_description=st.session_state.pdf_description,
        turn_count=turn_count,
        summary_history=st.session_state.summary_history,
        compression_rate=settings.get("compression_rate", 0.3),
        normal_turn_ids=st.session_state.normal_turn_ids,  # Phase 03-3-2
    )

    # 그래프에서 생성된 요약 업데이트
    if result.get("summary"):
        st.session_state.summary = result["summary"]
    if result.get("summary_history"):
        st.session_state.summary_history = result["summary_history"]
    # Phase 03-3-2: normal_turn_ids 업데이트
    if "normal_turn_ids" in result:
        st.session_state.normal_turn_ids = result["normal_turn_ids"]

    # 어시스턴트 메시지를 st.session_state에 저장 (UI 표시용)
    function_calls = [{"name": t, "args": {}} for t in result.get("tool_history", [])]
    assistant_msg = Message(
        turn_id=turn_count,
        role="assistant",
        content=result.get("text", ""),
        input_tokens=result.get("input_tokens", 0),
        output_tokens=result.get("output_tokens", 0),
        model_used=result.get("model_used", ""),
        function_calls=function_calls,
        tool_results=result.get("tool_results", {}),
        # Phase 04: 교육용 메타데이터
        mode=result.get("mode", "normal"),
        graph_path=result.get("graph_path", []),
        summary_triggered=result.get("summary_triggered", False),
        thought_process=result.get("thought_process"),
        thinking_budget=settings.get("thinking_budget", 0),
        is_casual=result.get("is_casual", False),
        actual_prompts=result.get("actual_prompts", {}),
    )
    st.session_state.messages.append(assistant_msg)
    # Phase 02-6: CSV 저장 제거 - SqliteSaver가 그래프 상태로 저장

    # 토큰 사용량 업데이트
    st.session_state.token_usage["input"] += result.get("input_tokens", 0)
    st.session_state.token_usage["output"] += result.get("output_tokens", 0)
    st.session_state.token_usage["total"] += result.get("total_tokens", 0)

    return result


def handle_stream_message(
    user_input: str,
    settings: dict,
    embed_repo: EmbeddingRepository,
):
    """스트리밍 채팅 핸들러

    Yields:
        dict: 스트리밍 청크 (token, tool_call, tool_result, done)
    """
    if not settings.get("gemini_api_key"):
        yield {"type": "token", "content": "Gemini API Key를 입력해주세요."}
        yield {"type": "done", "metadata": {"error": True}}
        return

    session_id = st.session_state.current_session
    current_tokens = st.session_state.token_usage["total"]
    if current_tokens >= TOKEN_LIMIT:
        yield {"type": "token", "content": f"토큰 제한({TOKEN_LIMIT_K}k)을 초과했습니다."}
        yield {"type": "done", "metadata": {"error": True}}
        return

    turn_count = get_turn_count(st.session_state.messages) + 1

    user_msg = Message(turn_id=turn_count, role="user", content=user_input)
    st.session_state.messages.append(user_msg)

    graph_builder = _create_graph_builder(settings, embed_repo)

    final_metadata = {}

    for chunk in graph_builder.stream(
        user_input=user_input,
        session_id=session_id,
        messages=st.session_state.messages[:-1],
        summary=st.session_state.summary,
        pdf_description=st.session_state.pdf_description,
        turn_count=turn_count,
        summary_history=st.session_state.summary_history,
        compression_rate=settings.get("compression_rate", 0.3),
        normal_turn_ids=st.session_state.normal_turn_ids,
    ):
        if chunk.get("type") == "done":
            final_metadata = chunk.get("metadata", {})
        yield chunk

    # done 이벤트 후 상태 업데이트
    if final_metadata and not final_metadata.get("error"):
        if final_metadata.get("summary"):
            st.session_state.summary = final_metadata["summary"]
        if final_metadata.get("summary_history"):
            st.session_state.summary_history = final_metadata["summary_history"]
        if "normal_turn_ids" in final_metadata:
            st.session_state.normal_turn_ids = final_metadata["normal_turn_ids"]

        function_calls = [{"name": t} for t in final_metadata.get("tool_history", [])]
        assistant_msg = Message(
            turn_id=turn_count,
            role="assistant",
            content=final_metadata.get("text", ""),
            input_tokens=final_metadata.get("input_tokens", 0),
            output_tokens=final_metadata.get("output_tokens", 0),
            model_used=final_metadata.get("model_used", ""),
            function_calls=function_calls,
            tool_results=final_metadata.get("tool_results", {}),
            # Phase 04: 교육용 메타데이터
            mode=final_metadata.get("mode", "normal"),
            graph_path=final_metadata.get("graph_path", []),
            summary_triggered=final_metadata.get("summary_triggered", False),
            thought_process=final_metadata.get("thought_process"),
            thinking_budget=settings.get("thinking_budget", 0),
            is_casual=final_metadata.get("is_casual", False),
            actual_prompts=final_metadata.get("actual_prompts", {}),
        )
        st.session_state.messages.append(assistant_msg)

        st.session_state.token_usage["input"] += final_metadata.get("input_tokens", 0)
        st.session_state.token_usage["output"] += final_metadata.get("output_tokens", 0)
        total = final_metadata.get("input_tokens", 0) + final_metadata.get("output_tokens", 0)
        st.session_state.token_usage["total"] += total


def handle_pdf_upload(uploaded_file, settings: dict) -> None:
    if uploaded_file:
        file_path = UPLOAD_PATH / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_pdf = file_path


def handle_pdf_process(step: str, settings: dict) -> dict:
    if "uploaded_pdf" not in st.session_state:
        return {"error": "PDF가 업로드되지 않았습니다."}

    if not settings.get("gemini_api_key"):
        return {"error": "Gemini API Key를 입력해주세요."}

    pdf_path = st.session_state.uploaded_pdf
    session_id = st.session_state.current_session

    try:
        if step == "텍스트 추출 중...":
            extractor = PDFExtractor()
            text, pages = extractor.extract_text(pdf_path)
            st.session_state.extracted_text = text
            st.session_state.pdf_pages = pages

        elif step == "청킹 중...":
            rag_service = RAGService()
            chunks = rag_service.chunk_text(
                st.session_state.extracted_text,
                chunk_size=1024,
                overlap=256,
                source_file=pdf_path.name,
            )
            st.session_state.chunks = chunks

        elif step == "정규화 중...":
            from service.prompt_loader import PromptLoader
            llm_service = LLMService(api_key=settings["gemini_api_key"])
            prompt_loader = PromptLoader()
            for chunk in st.session_state.chunks:
                prompt = prompt_loader.get_normalization_prompt(chunk_text=chunk.original_text)
                result = llm_service.generate(prompt, model="gemini-2.5-flash")
                chunk.normalized_text = result["text"]

            sample_text = st.session_state.chunks[0].normalized_text[:500] if st.session_state.chunks else ""
            desc_prompt = prompt_loader.get_description_prompt(sample_text=sample_text)
            desc_result = llm_service.generate(desc_prompt, model="gemini-2.5-flash")
            st.session_state.pdf_description = desc_result["text"]

        elif step == "임베딩 생성 중...":
            embed_service = EmbeddingService(
                api_key=settings["gemini_api_key"],
                model=settings["embedding_model"],
            )
            texts = [c.normalized_text for c in st.session_state.chunks]
            embeddings = embed_service.create_embeddings(texts)

            for chunk, embedding in zip(st.session_state.chunks, embeddings):
                chunk.embedding = embedding

            embed_repo = EmbeddingRepository(base_path=DATA_PATH)
            embed_repo.save_chunks(
                session_id,
                st.session_state.chunks,
                embedding_model=settings["embedding_model"],
                embedding_dim=768,
            )

        return {}

    except Exception as e:
        return {"error": str(e)}


def handle_pdf_delete(settings: dict) -> None:
    session_id = st.session_state.current_session
    embed_repo = EmbeddingRepository(base_path=DATA_PATH)
    embed_repo.delete_chunks(session_id)
    st.session_state.chunks = []
    st.session_state.pdf_description = ""


def load_session_data(
    session_id: str,
    session_manager: SessionManager,
    embed_repo: EmbeddingRepository,
) -> None:
    """세션 데이터를 로드하여 st.session_state에 설정 (Phase 02-6: SessionManager 사용)"""
    # SessionManager로 히스토리 조회
    history = session_manager.get_session_history(session_id)
    metadata = session_manager.get_session_metadata(session_id)

    if history:
        # 히스토리를 Message 객체로 변환
        messages = []
        turn = 0
        for item in history:
            if item["role"] == "user":
                turn += 1
            msg = Message(
                turn_id=turn,
                role=item["role"],
                content=item["content"],
            )
            messages.append(msg)

        st.session_state.messages = messages
        st.session_state.summary = metadata.get("summary", "")
        st.session_state.summary_history = metadata.get("summary_history", [])
        st.session_state.pdf_description = metadata.get("pdf_description", "")
        # Phase 03-3-2: normal_turn_ids 복원
        st.session_state.normal_turn_ids = metadata.get("normal_turn_ids", [])
        # 토큰 사용량은 현재 세션에서 리셋 (SqliteSaver에 저장되지 않음)
        st.session_state.token_usage = {"input": 0, "output": 0, "total": 0}

        # PDF 임베딩 로드
        chunks, _ = embed_repo.load_chunks(session_id)
        st.session_state.chunks = chunks
    else:
        # 새 세션 - 초기화
        st.session_state.messages = []
        st.session_state.token_usage = {"input": 0, "output": 0, "total": 0}
        st.session_state.summary = ""
        st.session_state.summary_history = []
        st.session_state.chunks = []
        st.session_state.normal_turn_ids = []  # Phase 03-3-2
        st.session_state.pdf_description = ""


def main():
    inject_custom_css()
    init_session_state()

    # Phase 02-6: SessionManager + EmbeddingRepository만 사용
    session_manager = SessionManager(db_path=DB_PATH)
    embed_repo = EmbeddingRepository(base_path=DATA_PATH)

    # 세션 변경 감지 및 처리
    if st.session_state.get("session_changed"):
        current_session = st.session_state.current_session

        # Phase 02-6: 이전 세션 저장 불필요 - SqliteSaver가 자동 저장
        # 새 세션/기존 세션 로드
        load_session_data(current_session, session_manager, embed_repo)

        # 플래그 초기화
        st.session_state.previous_session_id = current_session
        st.session_state.session_changed = False
        st.session_state.new_session_created = False

    # 저장된 세션 목록 로드 (앱 시작 시)
    if "sessions_loaded" not in st.session_state:
        saved_sessions = session_manager.list_sessions()
        for sid in saved_sessions:
            if sid not in st.session_state.sessions:
                st.session_state.sessions.append(sid)
        st.session_state.sessions_loaded = True

    settings = render_sidebar()

    st.title("Gemini Hybrid Chatbot")
    st.caption("AI 챗봇 핵심 개념 교육 데모")

    tab0, tab1, tab2, tab3 = st.tabs(["📖 Overview", "📝 Prompts", "💬 Chat", "📄 PDF Preprocessing"])

    with tab0:
        render_overview_tab()

    with tab1:
        render_prompts_tab()

    with tab2:
        render_chat_tab(
            on_send=lambda msg: handle_chat_message(msg, settings, embed_repo),
            on_stream=lambda msg: handle_stream_message(msg, settings, embed_repo),
            messages=st.session_state.messages,
            summary_history=st.session_state.summary_history,
            turn_count=get_turn_count(st.session_state.messages),
            use_streaming=True,
        )

    with tab3:
        render_pdf_tab(
            on_upload=lambda f: handle_pdf_upload(f, settings),
            on_process=lambda step: handle_pdf_process(step, settings),
            on_delete=lambda: handle_pdf_delete(settings),
            chunks=st.session_state.chunks,
        )


if __name__ == "__main__":
    main()
