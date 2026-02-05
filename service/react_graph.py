"""Phase 03-3: Tool Calling 리팩토링

LangChain 표준 Tool Calling 패턴 사용:
- bind_tools()로 LLM에 도구 바인딩
- ToolNode로 자동 도구 실행
- tools_condition으로 조건부 라우팅
"""
from typing import TypedDict, Any, Annotated
from pathlib import Path
import sqlite3

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from prompt.summary.summary_generator import get_prompt as get_summary_prompt
from service.tools import create_tools_with_services


def extract_text_from_content(content) -> str:
    """AIMessage.content에서 텍스트만 추출

    Phase 03-3-3: Gemini API 응답 구조 처리
    - content가 str인 경우: 그대로 반환
    - content가 list인 경우: text 타입 요소만 추출하여 결합
      - {"type": "text", "text": "..."} 형식
      - 문자열 요소
    - extras, signature 등 메타데이터 제외

    Args:
        content: AIMessage.content (str 또는 list)

    Returns:
        추출된 텍스트 문자열
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                # {"type": "text", "text": "..."} 형식
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                # extras, signature 등은 무시
        return "".join(text_parts)

    # 기타 타입은 문자열로 변환 시도
    return str(content)


class ChatState(TypedDict):
    """LangGraph 표준 상태 (Tool Calling 패턴)"""

    # === 메시지 (LangGraph 표준) - add_messages 리듀서 사용 ===
    messages: Annotated[list[BaseMessage], add_messages]

    # === 세션 정보 ===
    session_id: str

    # === 요약 관련 ===
    summary: str
    summary_history: list[dict]
    turn_count: int
    compression_rate: float  # 압축률 (0.1 ~ 0.5, 사이드바에서 설정, 기본값 0.3)

    # === Phase 03-3-2: Casual Mode 턴 관리 ===
    normal_turn_count: int  # normal 턴만 카운트 (요약 트리거 기준)
    normal_turn_ids: list[int]  # normal 턴의 turn_id 목록

    # === PDF 컨텍스트 ===
    pdf_description: str

    # === 메타데이터 ===
    input_tokens: int
    output_tokens: int
    model_used: str


def should_summarize(normal_turn_count: int, total_turn_count: int = None) -> bool:
    """요약 필요 여부 판단

    Phase 03-3-2: Casual Mode 대응
    - 기본: normal 턴 4, 7, 10, ... 에서 요약 트리거
    - Fallback: 전체 턴 10, 20, 30, ... 에서 강제 요약 (토큰 관리)

    Args:
        normal_turn_count: normal 모드 턴 카운트 (casual 제외)
        total_turn_count: 전체 턴 카운트 (Fallback용, 생략 시 normal_turn_count 사용)

    Returns:
        bool: 요약 필요 여부
    """
    if total_turn_count is None:
        total_turn_count = normal_turn_count

    # 기본 조건: normal 턴 4, 7, 10...
    if normal_turn_count >= 4 and (normal_turn_count - 1) % 3 == 0:
        return True

    # Fallback: 전체 턴 10개마다 강제 요약
    if total_turn_count >= 10 and total_turn_count % 10 == 0:
        return True

    return False


def extract_messages_by_turn_ids(messages: list, turn_ids: list[int]) -> list:
    """특정 turn_id에 해당하는 메시지만 추출

    Phase 03-3-2: Casual Mode에서 비연속 턴 추출용

    Args:
        messages: 전체 메시지 리스트 (additional_kwargs에 turn_id 포함)
        turn_ids: 추출할 턴 ID 목록 (예: [1, 3, 4])

    Returns:
        해당 턴의 메시지만 포함한 리스트
    """
    if not turn_ids:
        return []

    result = []
    turn_ids_set = set(turn_ids)

    for msg in messages:
        turn_id = getattr(msg, "additional_kwargs", {}).get("turn_id")
        if turn_id in turn_ids_set:
            result.append(msg)

    return result


def extract_last_n_turns(messages: list, n: int) -> list:
    """마지막 n턴의 완료된 메시지를 추출

    1턴의 끝 = AIMessage이면서 tool_calls가 없거나 빈 리스트인 경우

    Args:
        messages: 전체 메시지 리스트
        n: 추출할 턴 수

    Returns:
        완료된 턴들의 메시지 리스트 (진행 중인 턴은 제외)
    """
    if n <= 0:
        return []

    turns = []
    current_turn = []

    for msg in messages:
        current_turn.append(msg)

        # 턴 종료 조건: AIMessage이고 tool_calls가 없거나 빈 리스트
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            turns.append(current_turn)
            current_turn = []

    # current_turn에 남은 메시지 = 아직 완료되지 않은 진행 중인 턴
    # 이는 별도로 처리해야 함 (extract_current_turn에서)

    # 마지막 n턴 반환 (완료된 턴만)
    selected_turns = turns[-n:] if n <= len(turns) else turns
    return [msg for turn in selected_turns for msg in turn]


def extract_current_turn(messages: list) -> list:
    """현재 진행 중인 턴의 메시지 추출

    마지막 완료된 턴 이후의 모든 메시지를 반환

    Args:
        messages: 전체 메시지 리스트

    Returns:
        현재 진행 중인 턴의 메시지 리스트
    """
    if not messages:
        return []

    # 뒤에서부터 마지막 완료된 턴 찾기
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        # 완료된 턴의 끝 = AIMessage이고 tool_calls 없음
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            # 그 다음부터 끝까지가 현재 턴
            return messages[i + 1:]

    # 완료된 턴이 없으면 전체가 현재 턴
    return messages


class ReactGraphBuilder:
    """Tool Calling 패턴 기반 ReAct 그래프 빌더 (Phase 03-3)"""

    DEFAULT_DB_PATH = "data/langgraph.db"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_output_tokens: int = 8192,
        seed: int = None,
        max_iterations: int = 5,
        search_service: Any = None,
        embedding_service: Any = None,
        embedding_repo: Any = None,
        db_path: str = None,
    ):
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self.max_iterations = max_iterations
        self.search_service = search_service
        self.embedding_service = embedding_service
        self.embedding_repo = embedding_repo
        self.db_path = db_path or self.DEFAULT_DB_PATH

        # DB 디렉토리 생성
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # SqliteSaver 초기화
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._conn)
        self._graph = None

        # LLM 초기화
        llm_kwargs = {
            "model": model,
            "google_api_key": api_key,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
        }
        if seed is not None:
            llm_kwargs["seed"] = seed

        self._llm = ChatGoogleGenerativeAI(**llm_kwargs)
        self._llm_with_tools = None  # build()에서 초기화
        self._tools = None  # build()에서 초기화
        self._current_session_id = ""  # invoke 시 설정

    def _invoke_llm_with_token_tracking(
        self, messages: list[BaseMessage], llm=None
    ) -> tuple[str, int, int]:
        """LLM 호출 + 토큰 사용량 추적

        Phase 03-3-3: content가 list인 경우 텍스트만 추출
        """
        target_llm = llm or self._llm
        response = target_llm.invoke(messages)

        input_tokens = 0
        output_tokens = 0

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = response.usage_metadata.get("input_tokens", 0)
            output_tokens = response.usage_metadata.get("output_tokens", 0)
        elif hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata
            if "usage_metadata" in metadata:
                input_tokens = metadata["usage_metadata"].get("prompt_token_count", 0)
                output_tokens = metadata["usage_metadata"].get("candidates_token_count", 0)

        # Phase 03-3-3: content가 list인 경우 텍스트만 추출
        text_content = extract_text_from_content(response.content)
        return text_content, input_tokens, output_tokens

    def _build_system_prompt(self, state: ChatState) -> str:
        """System prompt 구성

        Phase 03-3: summary_history의 모든 요약을 포함
        """
        parts = ["당신은 유용한 AI 어시스턴트입니다."]
        parts.append("필요한 경우 도구를 사용하여 정확한 정보를 제공하세요.")

        # summary_history의 모든 요약을 포함
        summary_history = state.get("summary_history", [])
        if summary_history:
            parts.append("\n[이전 대화 요약]")
            for s in summary_history:
                turns = s.get("turns", [])
                if turns:
                    turns_str = f"{turns[0]}-{turns[-1]}턴"
                else:
                    turns_str = "이전 턴"
                parts.append(f"[{turns_str}] {s.get('summary', '')}")

        if state.get("pdf_description"):
            parts.append(f"\n[업로드된 PDF]\n{state['pdf_description']}")

        return "\n".join(parts)

    def _summary_node(self, state: ChatState) -> dict:
        """Summary Node: 요약 필요 여부 판단 및 생성

        Phase 03-3-2: Casual Mode 대응
        - normal_turn_count 4, 7, 10... 에서 직전 3 normal 턴 요약 생성
        - Fallback: 전체 턴 10, 20, 30... 에서 강제 요약
        - compression_rate 적용하여 요약 길이 조절
        - summary_history에 summarized_turns, excluded_turns 포함
        """
        turn_count = state.get("turn_count", 0)
        normal_turn_count = state.get("normal_turn_count", turn_count)
        normal_turn_ids = state.get("normal_turn_ids", [])
        messages = state.get("messages", [])
        session_id = state.get("session_id", "")
        compression_rate = state.get("compression_rate", 0.3)
        summary_history = state.get("summary_history", []).copy()

        # 요약 불필요한 경우 (normal_turn_count 기준, Fallback 포함)
        if not should_summarize(normal_turn_count, turn_count):
            return {
                "summary_history": summary_history,
            }

        # Phase 03-3-2: 요약할 normal 턴 ID (현재 턴 제외, 직전 3개)
        # normal_turn_ids에는 현재 턴이 포함되어 있으므로 [-4:-1] 사용
        if len(normal_turn_ids) >= 4:
            turns_to_summarize = normal_turn_ids[-4:-1]  # 현재 턴 제외한 직전 3개
        else:
            turns_to_summarize = normal_turn_ids[:-1] if len(normal_turn_ids) > 1 else []

        if not turns_to_summarize:
            return {
                "summary_history": summary_history,
            }

        # Phase 03-3-3: 전체 턴 범위 및 제외 턴 계산
        # 이전 요약이 있으면 그 다음 턴부터, 없으면 1턴부터
        if summary_history:
            last_summary = summary_history[-1]
            last_summarized_turns = last_summary.get("turns", [])
            first_turn = max(last_summarized_turns) + 1 if last_summarized_turns else 1
        else:
            first_turn = 1

        end_turn = turns_to_summarize[-1]
        all_turns_in_range = list(range(first_turn, end_turn + 1))
        excluded_turns = [t for t in all_turns_in_range if t not in turns_to_summarize]

        # Phase 03-3-2: turn_id 기반 메시지 추출
        messages_to_summarize = extract_messages_by_turn_ids(messages, turns_to_summarize)

        if not messages_to_summarize:
            # Fallback: turn_id 없는 경우 기존 방식 사용
            messages_without_current = extract_last_n_turns(messages, n=100)
            messages_to_summarize = extract_last_n_turns(messages_without_current, n=3)

        if not messages_to_summarize:
            return {
                "summary_history": summary_history,
            }

        # 원본 텍스트 길이 계산
        original_text = "".join(
            getattr(msg, "content", "") or "" for msg in messages_to_summarize
        )
        original_chars = len(original_text)

        if original_chars == 0:
            return {
                "summary_history": summary_history,
            }

        # 요약할 대화 텍스트 구성
        conversation_parts = []
        for msg in messages_to_summarize:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            if content:
                conversation_parts.append(f"{role}: {content}")
        conversation = "\n".join(conversation_parts)

        # LLM으로 요약 생성 (compression_rate 적용)
        target_chars = int(original_chars * compression_rate)
        prompt = get_summary_prompt(
            previous_summary="",  # 새 요약은 이전 요약 없이 생성
            conversation=conversation,
            target_length=target_chars,
        )
        content, in_tokens, out_tokens = self._invoke_llm_with_token_tracking(
            [HumanMessage(content=prompt)]
        )
        new_summary = content.strip()
        summary_chars = len(new_summary)

        # 토큰 누적
        total_input = state.get("input_tokens", 0) + in_tokens
        total_output = state.get("output_tokens", 0) + out_tokens

        # Phase 03-3-2: summary_history에 summarized_turns, excluded_turns 포함
        summary_history.append({
            "thread_id": session_id,
            "turns": all_turns_in_range,  # UI 표시용: [1, 2, 3]
            "summarized_turns": turns_to_summarize,  # 실제 요약된 턴: [1, 3]
            "excluded_turns": excluded_turns,  # casual로 제외된 턴: [2]
            "turn_length": len(turns_to_summarize),
            "original_chars": original_chars,
            "summary_chars": summary_chars,
            "compression_rate": compression_rate,
            "summary": new_summary,
        })

        return {
            "summary_history": summary_history,
            "input_tokens": total_input,
            "output_tokens": total_output,
        }

    def _llm_node(self, state: ChatState) -> dict:
        """LLM 노드: 도구 호출 또는 최종 응답 생성

        Phase 03-3: Context 구성
        - System prompt (요약 포함)
        - 완료된 raw 턴들 (요약되지 않은 턴)
        - 현재 진행 중인 턴
        """
        turn_count = state.get("turn_count", 0)
        summary_history = state.get("summary_history", [])
        all_messages = state.get("messages", [])

        # 요약되지 않은 raw 턴 수 계산
        summarized_turns = len(summary_history) * 3
        raw_turn_count = turn_count - summarized_turns

        # System prompt 구성 (요약 포함)
        system_content = self._build_system_prompt(state)
        system_prompt = SystemMessage(content=system_content)

        # Context 구성
        if raw_turn_count <= 1:
            # 현재 턴만 있는 경우 (Turn 4, 7, 10 등)
            # 현재 진행 중인 턴의 메시지만 사용
            current_turn = extract_current_turn(all_messages)
            context_messages = [system_prompt] + current_turn
        else:
            # 여러 raw 턴이 있는 경우 (Turn 5, 6, 8, 9 등)
            # 완료된 이전 raw 턴들 + 현재 턴
            completed_raw_turns = max(0, raw_turn_count - 1)
            recent_completed = extract_last_n_turns(all_messages, n=completed_raw_turns) if completed_raw_turns > 0 else []
            current_turn = extract_current_turn(all_messages)
            context_messages = [system_prompt] + recent_completed + current_turn

        # LLM 호출 (bind_tools 된 LLM)
        response = self._llm_with_tools.invoke(context_messages)

        # 토큰 추적
        in_tokens, out_tokens = 0, 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            in_tokens = response.usage_metadata.get("input_tokens", 0)
            out_tokens = response.usage_metadata.get("output_tokens", 0)
        elif hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata
            if "usage_metadata" in metadata:
                in_tokens = metadata["usage_metadata"].get("prompt_token_count", 0)
                out_tokens = metadata["usage_metadata"].get("candidates_token_count", 0)

        return {
            "messages": [response],  # AIMessage 추가 (tool_calls 포함 가능)
            "input_tokens": state.get("input_tokens", 0) + in_tokens,
            "output_tokens": state.get("output_tokens", 0) + out_tokens,
        }

    def build(self):
        """그래프 빌드 및 컴파일 (Tool Calling 패턴)"""

        # 서비스가 주입된 도구 생성
        self._tools = create_tools_with_services(
            search_service=self.search_service,
            embedding_service=self.embedding_service,
            embedding_repo=self.embedding_repo,
            session_id=self._current_session_id,
            llm=self._llm,
        )

        # LLM에 도구 바인딩
        self._llm_with_tools = self._llm.bind_tools(self._tools)

        # ToolNode 생성
        tool_node = ToolNode(self._tools)

        # 그래프 빌더
        builder = StateGraph(ChatState)

        # 노드 추가
        builder.add_node("summary_node", self._summary_node)
        builder.add_node("llm_node", self._llm_node)
        builder.add_node("tool_node", tool_node)

        # 엣지 정의
        builder.add_edge(START, "summary_node")
        builder.add_edge("summary_node", "llm_node")

        # 조건부 엣지: LLM이 도구 호출했으면 tool_node로, 아니면 END
        builder.add_conditional_edges(
            "llm_node",
            tools_condition,
            {
                "tools": "tool_node",
                END: END,
            }
        )

        # tool_node 실행 후 다시 llm_node로
        builder.add_edge("tool_node", "llm_node")

        self._graph = builder.compile(checkpointer=self._checkpointer)
        return self._graph

    def invoke(
        self,
        user_input: str,
        session_id: str,
        messages: list = None,
        summary: str = "",
        pdf_description: str = "",
        turn_count: int = 0,
        summary_history: list = None,
        compression_rate: float = 0.3,
        normal_turn_ids: list = None,
    ) -> dict:
        """그래프 실행

        Args:
            user_input: 사용자 질문
            session_id: 세션 ID
            messages: 이전 메시지
            summary: 대화 요약
            pdf_description: PDF 설명
            turn_count: 현재 턴 수
            summary_history: 요약 히스토리
            compression_rate: 요약 압축률 (0.1 ~ 0.5, 기본값 0.3)
            normal_turn_ids: normal 턴 ID 목록 (Phase 03-3-2)

        Returns:
            dict: 응답 결과
        """
        # Phase 03-3-2: normal_turn_ids 초기화
        if normal_turn_ids is None:
            normal_turn_ids = []

        # Fast-path: 일상적 대화는 그래프 실행 없이 바로 응답
        from service.reasoning_detector import detect_reasoning_need

        mode = detect_reasoning_need(user_input)
        if mode == "casual":
            casual_prompt = f"""사용자가 "{user_input}"라고 말했습니다.
자연스럽고 친근하게 짧게 응답해주세요. 분석이나 설명 없이요."""

            content, in_tokens, out_tokens = self._invoke_llm_with_token_tracking(
                [HumanMessage(content=casual_prompt)]
            )

            # Phase 03-3-2: casual은 normal_turn_ids 변경 없이 반환
            return {
                "text": content,
                "tool_history": [],
                "tool_results": {},
                "iteration": 0,
                "model_used": self.model_name,
                "summary": summary,
                "summary_history": summary_history or [],
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens,
                "is_casual": True,
                "normal_turn_ids": normal_turn_ids,  # 변경 없음
                "normal_turn_count": len(normal_turn_ids),  # 변경 없음
                "error": None,
            }

        # Phase 03-3-2: normal 모드 - turn_id 추가
        updated_normal_turn_ids = normal_turn_ids + [turn_count]
        normal_turn_count = len(updated_normal_turn_ids)

        # 세션 ID 저장 (도구에서 사용)
        self._current_session_id = session_id

        if self._graph is None:
            self.build()

        # domain.message.Message를 LangChain BaseMessage로 변환
        converted_messages = []
        for msg in (messages or []):
            if hasattr(msg, "role") and hasattr(msg, "content"):
                # domain.message.Message 타입
                if msg.role == "user":
                    converted_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    converted_messages.append(AIMessage(content=msg.content))
            elif isinstance(msg, BaseMessage):
                # 이미 LangChain 메시지 타입
                converted_messages.append(msg)

        # Phase 03-3-2: 사용자 메시지에 turn_id 메타데이터 추가
        user_message = HumanMessage(
            content=user_input,
            additional_kwargs={"turn_id": turn_count, "mode": "normal"}
        )
        all_messages = converted_messages + [user_message]

        initial_state: ChatState = {
            "messages": all_messages,
            "session_id": session_id,
            "summary": summary,
            "summary_history": summary_history or [],
            "turn_count": turn_count,
            "compression_rate": compression_rate,
            "pdf_description": pdf_description,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_used": self.model_name,
            # Phase 03-3-2: normal 턴 관리
            "normal_turn_count": normal_turn_count,
            "normal_turn_ids": updated_normal_turn_ids,
        }

        config = {"configurable": {"thread_id": session_id}}
        result = self._graph.invoke(initial_state, config)

        # 결과 추출
        result_messages = result.get("messages", [])
        final_message = result_messages[-1] if result_messages else None
        final_text = ""
        if final_message:
            # Phase 03-3-3: content가 list인 경우 처리 (extras/signature 제외)
            raw_content = getattr(final_message, "content", "")
            final_text = extract_text_from_content(raw_content)

        # Phase 03-3-3: 현재 턴 메시지만 추출하여 tool_history 계산
        # turn_id를 사용하여 현재 턴의 시작점을 정확히 찾음
        # (SqliteSaver 체크포인트에 이전 턴의 ToolMessage가 포함되어 있어 메시지 수로는 불정확)
        current_turn_start = None
        for i, msg in enumerate(result_messages):
            if (isinstance(msg, HumanMessage) and
                getattr(msg, "additional_kwargs", {}).get("turn_id") == turn_count):
                current_turn_start = i
                break

        if current_turn_start is not None:
            current_turn_messages = result_messages[current_turn_start:]
        else:
            # Fallback: 이전 방식 (turn_id가 없는 경우)
            prev_message_count = len(converted_messages)
            current_turn_messages = result_messages[prev_message_count:]

        tool_history = []
        tool_results = {}
        iteration = 0
        for msg in current_turn_messages:  # 현재 턴만!
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_history.append(tc["name"])
                    iteration += 1
            if hasattr(msg, "type") and msg.type == "tool":
                tool_results[msg.name] = msg.content

        input_tokens = result.get("input_tokens", 0)
        output_tokens = result.get("output_tokens", 0)

        return {
            "text": final_text,
            "tool_history": tool_history,
            "tool_results": tool_results,
            "iteration": iteration,
            "model_used": self.model_name,
            "summary": result.get("summary", ""),
            "summary_history": result.get("summary_history", []),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            # Phase 03-3-2: normal 턴 정보 반환
            "normal_turn_ids": updated_normal_turn_ids,
            "normal_turn_count": normal_turn_count,
            "error": None,
        }
