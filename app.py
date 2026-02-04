import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv

from domain.message import Message
from domain.session import Session
from repository.conversation_repo import ConversationRepository
from repository.embedding_repo import EmbeddingRepository
from repository.pdf_extractor import PDFExtractor
from service.llm_service import LLMService
from service.embedding_service import EmbeddingService
from service.rag_service import RAGService
from service.summary_service import SummaryService
from service.search_service import SearchService
from service.tool_manager import ToolManager
from component.sidebar import render_sidebar
from component.chat_tab import render_chat_tab
from component.pdf_tab import render_pdf_tab

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
    )

    if result.get("function_calls"):
        for fc in result["function_calls"]:
            if fc["name"] == "switch_to_reasoning":
                model_to_use = "gemini-2.5-pro"
                result = llm_service.generate(
                    full_prompt,
                    model=model_to_use,
                    tools=tool_manager.get_tools(),
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                )
                break
            else:
                tool_result = tool_manager.execute_tool(fc["name"], fc["args"])
                enhanced_prompt = f"{full_prompt}\n\n[Tool Result: {fc['name']}]\n{tool_result}"
                result = llm_service.generate(
                    enhanced_prompt,
                    model=model_to_use,
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                )

    assistant_msg = Message(
        turn_id=turn_count,
        role="assistant",
        content=result["text"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        model_used=result["model_used"],
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
            for chunk in st.session_state.chunks:
                prompt = f"""다음 텍스트를 검색에 최적화된 형태로 정규화하세요.
규칙:
1. 오탈자와 띄어쓰기 오류를 수정합니다.
2. 불필요한 특수문자와 중복 공백을 제거합니다.
3. 약어가 있다면 괄호 안에 풀이를 추가합니다.
4. 핵심 키워드는 그대로 유지합니다.
5. 원문의 의미를 변경하지 않습니다.

원본 텍스트:
{chunk.original_text}

정규화된 텍스트:"""
                result = llm_service.generate(prompt, model="gemini-2.5-flash")
                chunk.normalized_text = result["text"]

            desc_prompt = f"""다음 문서 내용을 바탕으로 이 PDF 문서에 대한 간단한 설명(description)을 작성하세요.
50자 이내로 작성하세요.

문서 내용 샘플:
{st.session_state.chunks[0].normalized_text[:500] if st.session_state.chunks else ''}

설명:"""
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


def _get_system_prompt(pdf_description: str = "") -> str:
    base_prompt = """당신은 두 가지 모드로 동작합니다:
1. 일반 모드: 간단한 질문, 일상 대화, 정보 조회
2. 추론 모드: 복잡한 분석, 다단계 추론, 비교/평가, 수학적 계산

다음 상황에서는 반드시 switch_to_reasoning 툴을 호출하세요:
- 여러 정보를 종합하여 결론을 도출해야 할 때
- "왜", "어떻게", "비교해줘", "분석해줘" 등의 심층 질문
- PDF 내용을 기반으로 추론이 필요할 때
- 수학적 계산이나 논리적 단계가 필요할 때"""

    if pdf_description:
        base_prompt += f"""

[업로드된 PDF 정보]
{pdf_description}
사용자가 이 문서와 관련된 질문을 하면 search_pdf_knowledge 툴을 사용하세요."""

    return base_prompt


def main():
    init_session_state()

    settings = render_sidebar()

    conv_repo = ConversationRepository(base_path=DATA_PATH)
    embed_repo = EmbeddingRepository(base_path=DATA_PATH)

    tab1, tab2 = st.tabs(["💬 Chat", "📄 PDF Preprocessing"])

    with tab1:
        render_chat_tab(
            on_send=lambda msg: handle_chat_message(msg, settings, conv_repo, embed_repo),
            messages=st.session_state.messages,
        )

    with tab2:
        render_pdf_tab(
            on_upload=lambda f: handle_pdf_upload(f, settings),
            on_process=lambda step: handle_pdf_process(step, settings),
            on_delete=lambda: handle_pdf_delete(settings),
            chunks=st.session_state.chunks,
        )


if __name__ == "__main__":
    main()
