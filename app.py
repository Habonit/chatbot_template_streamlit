import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv

from domain.message import Message
from domain.session import Session
from repository.conversation_repo import ConversationRepository
from repository.embedding_repo import EmbeddingRepository
from repository.session_repo import SessionRepository
from repository.pdf_extractor import PDFExtractor
from service.llm_service import LLMService
from service.embedding_service import EmbeddingService
from service.rag_service import RAGService
from service.summary_service import SummaryService
from service.search_service import SearchService
from service.tool_manager import ToolManager
from service.prompt_loader import PromptLoader
from component.sidebar import render_sidebar
from component.chat_tab import render_chat_tab
from component.pdf_tab import render_pdf_tab
from component.overview_tab import render_overview_tab
from component.prompts_tab import render_prompts_tab

load_dotenv()

st.set_page_config(
    page_title="Gemini Hybrid Chatbot",
    page_icon="🤖",
    layout="wide",
)

DATA_PATH = Path("data/sessions")
UPLOAD_PATH = Path("data/uploads/temp")
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
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "pdf_description" not in st.session_state:
        st.session_state.pdf_description = ""


def get_turn_count(messages: list[Message]) -> int:
    return len([m for m in messages if m.role == "user"])


def handle_chat_message(
    user_input: str,
    settings: dict,
    conv_repo: ConversationRepository,
    embed_repo: EmbeddingRepository,
) -> dict:
    if not settings.get("gemini_api_key"):
        return {"text": "Gemini API Key를 입력해주세요.", "error": True}

    session_id = st.session_state.current_session

    llm_service = LLMService(
        api_key=settings["gemini_api_key"],
        default_model=settings["model"],
    )
    summary_service = SummaryService()
    tool_manager = ToolManager()

    tool_manager.register_switch_tool()

    if st.session_state.chunks:
        embed_service = EmbeddingService(
            api_key=settings["gemini_api_key"],
            model=settings["embedding_model"],
        )

        def search_pdf_knowledge(query: str, top_k: int = 5) -> list[dict]:
            query_embedding = embed_service.create_embedding(query)
            results = embed_repo.search_similar(session_id, query_embedding, top_k)
            return [{"content": r["chunk"].normalized_text, "score": r["score"]} for r in results]

        tool_manager.register_tool(search_pdf_knowledge)

    if settings.get("search_enabled") and settings.get("tavily_api_key"):
        search_service = SearchService(api_key=settings["tavily_api_key"])

        def web_search(query: str) -> str:
            results = search_service.search(
                query,
                search_depth=settings["search_depth"],
                max_results=settings["max_results"],
            )
            return search_service.format_for_llm(results)

        tool_manager.register_tool(web_search)

    turn_count = get_turn_count(st.session_state.messages) + 1

    user_msg = Message(turn_id=turn_count, role="user", content=user_input)
    st.session_state.messages.append(user_msg)
    conv_repo.append_message(session_id, user_msg)

    if summary_service.should_summarize(turn_count):
        to_summarize, to_keep = summary_service.get_turns_to_summarize(
            st.session_state.messages[:-1], turn_count - 1
        )
        if to_summarize:
            summary_prompt = summary_service.build_summary_prompt(
                st.session_state.summary, to_summarize
            )
            summary_result = llm_service.generate(summary_prompt, model="gemini-2.5-flash")
            st.session_state.summary = summary_result["text"]
    else:
        to_keep = st.session_state.messages[:-1]

    context = summary_service.build_context(
        messages=to_keep,
        summary=st.session_state.summary,
        system_prompt=_get_system_prompt(st.session_state.pdf_description),
    )

    full_prompt = f"{context}\n\n[현재 사용자 입력]\n{user_input}"

    current_tokens = st.session_state.token_usage["total"]
    if current_tokens >= TOKEN_LIMIT:
        return {"text": f"토큰 제한({TOKEN_LIMIT_K}k)을 초과했습니다. 새 세션을 시작해주세요.", "error": True}
    if current_tokens >= TOKEN_LIMIT * 0.8:
        st.warning(f"토큰 사용량이 80%를 초과했습니다 ({current_tokens:,}/{TOKEN_LIMIT:,})")

    model_to_use = settings["model"]
    result = llm_service.generate(
        full_prompt,
        model=model_to_use,
        tools=tool_manager.get_tools() if tool_manager.get_tool_names() else None,
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        max_output_tokens=settings.get("max_output_tokens", 8192),
    )

    executed_function_calls = []

    if result.get("function_calls"):
        prompt_loader = PromptLoader()
        for fc in result["function_calls"]:
            executed_function_calls.append(fc)
            if fc["name"] == "switch_to_reasoning":
                model_to_use = "gemini-2.5-pro"
                cot_prompt = prompt_loader.get_cot_prompt(
                    user_input=user_input,
                    context=context,
                )
                result = llm_service.generate(
                    cot_prompt,
                    model=model_to_use,
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                    max_output_tokens=settings.get("max_output_tokens", 8192),
                )
                break
            else:
                tool_result = tool_manager.execute_tool(fc["name"], fc["args"])
                if fc["name"] == "web_search":
                    tavily_prompt = prompt_loader.get_tavily_prompt(
                        search_results=tool_result,
                        user_query=user_input,
                    )
                    enhanced_prompt = f"{context}\n\n{tavily_prompt}"
                else:
                    enhanced_prompt = f"{full_prompt}\n\n[Tool Result: {fc['name']}]\n{tool_result}"
                result = llm_service.generate(
                    enhanced_prompt,
                    model=model_to_use,
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                    max_output_tokens=settings.get("max_output_tokens", 8192),
                )

    assistant_msg = Message(
        turn_id=turn_count,
        role="assistant",
        content=result["text"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        model_used=result["model_used"],
        function_calls=executed_function_calls,
    )
    st.session_state.messages.append(assistant_msg)
    conv_repo.append_message(session_id, assistant_msg)

    st.session_state.token_usage["input"] += result["input_tokens"]
    st.session_state.token_usage["output"] += result["output_tokens"]
    st.session_state.token_usage["total"] += result["total_tokens"]

    return result


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


def save_current_session(
    session_repo: SessionRepository,
    conv_repo: ConversationRepository,
) -> None:
    """현재 세션 데이터를 저장"""
    session_id = st.session_state.get("current_session")
    if not session_id:
        return

    session = Session(
        session_id=session_id,
        total_turns=get_turn_count(st.session_state.messages),
        current_summary=st.session_state.get("summary", ""),
        token_usage=st.session_state.get("token_usage", {"input": 0, "output": 0, "total": 0}),
        pdf_description=st.session_state.get("pdf_description", ""),
        pdf_files=[],
    )
    session_repo.save_session(session)
    conv_repo.save_messages(session_id, st.session_state.messages)


def load_session_data(
    session_id: str,
    session_repo: SessionRepository,
    conv_repo: ConversationRepository,
    embed_repo: EmbeddingRepository,
) -> None:
    """세션 데이터를 로드하여 st.session_state에 설정"""
    session = session_repo.load_session(session_id)

    if session:
        st.session_state.messages = conv_repo.load_messages(session_id)
        st.session_state.token_usage = session.token_usage
        st.session_state.summary = session.current_summary
        st.session_state.pdf_description = session.pdf_description

        chunks, _ = embed_repo.load_chunks(session_id)
        st.session_state.chunks = chunks
    else:
        # 새 세션 - 초기화
        st.session_state.messages = []
        st.session_state.token_usage = {"input": 0, "output": 0, "total": 0}
        st.session_state.summary = ""
        st.session_state.chunks = []
        st.session_state.pdf_description = ""


def _get_system_prompt(pdf_description: str = "") -> str:
    prompt_loader = PromptLoader()
    return prompt_loader.get_system_prompt(pdf_description=pdf_description if pdf_description else None)


def main():
    init_session_state()

    conv_repo = ConversationRepository(base_path=DATA_PATH)
    embed_repo = EmbeddingRepository(base_path=DATA_PATH)
    session_repo = SessionRepository(base_path=DATA_PATH)

    # 세션 변경 감지 및 처리
    if st.session_state.get("session_changed"):
        previous_session = st.session_state.get("previous_session_id")
        current_session = st.session_state.current_session

        # 이전 세션 저장 (새 세션 생성이 아닌 경우)
        if previous_session and not st.session_state.get("new_session_created"):
            save_current_session(session_repo, conv_repo)

        # 새 세션/기존 세션 로드
        load_session_data(current_session, session_repo, conv_repo, embed_repo)

        # 플래그 초기화
        st.session_state.previous_session_id = current_session
        st.session_state.session_changed = False
        st.session_state.new_session_created = False

    # 저장된 세션 목록 로드 (앱 시작 시)
    if "sessions_loaded" not in st.session_state:
        saved_sessions = session_repo.list_sessions()
        for sid in saved_sessions:
            if sid not in st.session_state.sessions:
                st.session_state.sessions.append(sid)
        st.session_state.sessions_loaded = True

    settings = render_sidebar()

    tab0, tab1, tab2, tab3 = st.tabs(["📖 Overview", "📝 Prompts", "💬 Chat", "📄 PDF Preprocessing"])

    with tab0:
        render_overview_tab()

    with tab1:
        render_prompts_tab()

    with tab2:
        render_chat_tab(
            on_send=lambda msg: handle_chat_message(msg, settings, conv_repo, embed_repo),
            messages=st.session_state.messages,
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
