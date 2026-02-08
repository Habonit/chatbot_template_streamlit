"""Phase 03-3: Tool Calling 리팩토링

LangChain 표준 Tool Calling 패턴 사용:
- bind_tools()로 LLM에 도구 바인딩
- ToolNode로 자동 도구 실행
- tools_condition으로 조건부 라우팅
"""
from typing import TypedDict, Any, Annotated, Generator
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


def extract_thought_from_content(content) -> tuple[str, str]:
    """AIMessage.content에서 사고 과정과 응답 텍스트를 분리

    include_thoughts=True 시 content가 list:
      [{"type": "text", "text": "...", "thought": True}, ...]

    Returns:
        (thought_text, response_text)
    """
    if not isinstance(content, list):
        return "", str(content) if content else ""

    thought_parts = []
    response_parts = []

    for item in content:
        if isinstance(item, dict):
            text = item.get("text", "")
            if item.get("thought"):
                thought_parts.append(text)
            elif item.get("type") == "text":
                response_parts.append(text)
        elif isinstance(item, str):
            response_parts.append(item)

    return "\n".join(thought_parts), "".join(response_parts)


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

    # === Phase 04: 교육용 메타데이터 ===
    graph_path: list[str]  # 실행된 노드 경로 추적
    summary_triggered: bool  # 요약 실행 여부

    # === Router Node 통합 ===
    mode: str  # "casual", "normal" — router_node이 설정
    is_casual: bool  # mode == "casual" 편의 플래그

    # === Phase 05: 프롬프트 캡처 ===
    actual_prompts: dict

    # === PDF 컨텍스트 ===
    pdf_description: str

    # === 메타데이터 ===
    input_tokens: int
    output_tokens: int
    model_used: str


def should_summarize(normal_turn_count: int, total_turn_count: int = None) -> bool:
    """요약 필요 여부 판단

    summary_node가 router_node 이전에 실행되므로
    현재 턴이 아직 normal_turn_count에 포함되지 않은 상태에서 판단합니다.
    - 기본: normal 턴 3, 6, 9, ... 에서 요약 트리거 (이전 3턴 누적 기준)
    - Fallback: 전체 턴 10, 20, 30, ... 에서 강제 요약 (토큰 관리)

    Args:
        normal_turn_count: normal 모드 턴 카운트 (casual 제외, 현재 턴 미포함)
        total_turn_count: 전체 턴 카운트 (Fallback용, 생략 시 normal_turn_count 사용)

    Returns:
        bool: 요약 필요 여부
    """
    if total_turn_count is None:
        total_turn_count = normal_turn_count

    # 기본 조건: normal 턴 3, 6, 9... (이전 normal 턴 3개 누적 시)
    if normal_turn_count >= 3 and normal_turn_count % 3 == 0:
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
        thinking_budget: int = 0,
        show_thoughts: bool = False,
        search_service: Any = None,
        embedding_service: Any = None,
        embedding_repo: Any = None,
        db_path: str = None,
        search_depth: str = "basic",
        max_results: int = 5,
    ):
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self.max_iterations = max_iterations
        self.thinking_budget = thinking_budget
        self.show_thoughts = show_thoughts
        self.search_service = search_service
        self.embedding_service = embedding_service
        self.embedding_repo = embedding_repo
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.search_depth = search_depth
        self.max_results = max_results

        # Phase 03-5: thinking 지원 모델 검증
        THINKING_SUPPORTED_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"]
        if thinking_budget > 0 and model not in THINKING_SUPPORTED_MODELS:
            import warnings
            warnings.warn(f"{model}은 thinking을 지원하지 않습니다. thinking_budget 무시됨.")
            self.thinking_budget = 0

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

        # Phase 03-5: thinking 설정
        if self.thinking_budget > 0:
            llm_kwargs["thinking_budget"] = self.thinking_budget
            if self.show_thoughts:
                llm_kwargs["include_thoughts"] = True

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
        graph_path = state.get("graph_path", []) + ["summary_node"]

        # 요약 불필요한 경우 (normal_turn_count 기준, Fallback 포함)
        if not should_summarize(normal_turn_count, turn_count):
            return {
                "summary_history": summary_history,
                "summary_triggered": False,
                "graph_path": graph_path,
            }

        # 요약할 normal 턴 ID (직전 3개)
        # summary_node가 router_node 이전에 실행되므로 현재 턴은 아직 미포함
        if len(normal_turn_ids) >= 3:
            turns_to_summarize = normal_turn_ids[-3:]  # 직전 3개 normal 턴
        else:
            turns_to_summarize = list(normal_turn_ids) if normal_turn_ids else []

        if not turns_to_summarize:
            return {
                "summary_history": summary_history,
                "summary_triggered": False,
                "graph_path": graph_path,
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
                "summary_triggered": False,
                "graph_path": graph_path,
            }

        # 원본 텍스트 길이 계산
        original_text = "".join(
            getattr(msg, "content", "") or "" for msg in messages_to_summarize
        )
        original_chars = len(original_text)

        if original_chars == 0:
            return {
                "summary_history": summary_history,
                "summary_triggered": False,
                "graph_path": graph_path,
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
            "summary_triggered": True,
            "graph_path": graph_path,
        }

    def _llm_node(self, state: ChatState) -> dict:
        """LLM 노드: 도구 호출 또는 최종 응답 생성

        Context 구성 (단기/장기 기억 분리):
        - 장기 기억: System prompt에 요약 포함 (normal 턴만 요약)
        - 단기 기억: 최근 3턴 raw 메시지 (casual 포함) + 현재 턴
        """
        graph_path = state.get("graph_path", []) + ["llm_node"]
        all_messages = state.get("messages", [])

        # 장기 기억: System prompt (요약 포함)
        system_content = self._build_system_prompt(state)
        system_prompt = SystemMessage(content=system_content)

        # 단기 기억: 최근 3턴 (casual 포함) + 현재 진행 중인 턴
        recent_completed = extract_last_n_turns(all_messages, n=3)
        current_turn = extract_current_turn(all_messages)
        context_messages = [system_prompt] + recent_completed + current_turn

        # Phase 05: 프롬프트 캡처
        actual_prompts = {
            "system_prompt": system_content,
            "user_messages_count": len(current_turn),
            "context_turns": len(recent_completed),
        }

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
            "graph_path": graph_path,
            "actual_prompts": actual_prompts,
        }

    def _router_node(self, state: ChatState) -> dict:
        """Router Node: 입력을 분석하여 casual/normal/reasoning 모드 결정

        그래프의 첫 번째 노드로 실행되어 모든 라우팅을 그래프 내부에서 수행합니다.
        """
        from service.reasoning_detector import detect_reasoning_need

        messages = state.get("messages", [])
        turn_count = state.get("turn_count", 0)
        normal_turn_ids = list(state.get("normal_turn_ids", []))

        # 마지막 HumanMessage에서 user_input 추출
        user_input = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_input = msg.content
                break

        mode = detect_reasoning_need(user_input)
        is_casual = mode == "casual"

        # casual이 아닌 경우에만 normal_turn_ids에 turn_count 추가
        if not is_casual:
            normal_turn_ids = normal_turn_ids + [turn_count]

        return {
            "mode": mode,
            "is_casual": is_casual,
            "graph_path": state.get("graph_path", []) + ["router_node"],
            "normal_turn_ids": normal_turn_ids,
            "normal_turn_count": len(normal_turn_ids),
        }

    def _build_casual_context_from_state(
        self, user_input: str, summary_history: list, history_messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        """casual 모드용 context를 state의 BaseMessage 리스트로 구성

        Args:
            user_input: 사용자 입력 텍스트
            summary_history: 요약 히스토리
            history_messages: 이미 변환된 BaseMessage 리스트 (현재 턴의 HumanMessage 제외)

        Returns:
            list[BaseMessage]: SystemMessage(요약) + 히스토리 + casual HumanMessage
        """
        context = []

        # 요약문이 있으면 SystemMessage로 포함
        if summary_history:
            parts = ["당신은 유용한 AI 어시스턴트입니다."]
            parts.append("\n[이전 대화 요약]")
            for s in summary_history:
                turns = s.get("turns", [])
                if turns:
                    turns_str = f"{turns[0]}-{turns[-1]}턴"
                else:
                    turns_str = "이전 턴"
                parts.append(f"[{turns_str}] {s.get('summary', '')}")
            context.append(SystemMessage(content="\n".join(parts)))

        # 히스토리 메시지 (현재 턴의 HumanMessage 제외)
        for msg in history_messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                context.append(msg)

        # casual 프롬프트
        casual_prompt = f"""사용자가 "{user_input}"라고 말했습니다.
자연스럽고 친근하게 짧게 응답해주세요. 분석이나 설명 없이요."""
        context.append(HumanMessage(content=casual_prompt))

        return context

    def _casual_node(self, state: ChatState) -> dict:
        """Casual Node: casual 모드 LLM 호출

        self._llm.invoke(context)를 직접 호출합니다.
        LangGraph stream_mode="messages"가 BaseChatModel.invoke()를 자동 인터셉트하여
        토큰 단위 스트리밍을 제공합니다.

        컨텍스트 윈도우링: 전체 히스토리 대신 최근 3턴만 사용하여
        토큰 효율성을 확보합니다. 요약문은 SystemMessage로 포함됩니다.
        """
        messages = state.get("messages", [])
        summary_history = state.get("summary_history", [])
        graph_path = state.get("graph_path", []) + ["casual_node"]

        # 마지막 HumanMessage에서 user_input 추출
        user_input = ""
        # 히스토리 메시지 = 마지막 HumanMessage 이전의 모든 메시지
        history_messages = []
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage) and i == len(messages) - 1:
                user_input = msg.content
            else:
                history_messages.append(msg)

        # 컨텍스트 윈도우링: 최근 3턴만 사용
        recent_history = extract_last_n_turns(history_messages, n=3)

        context = self._build_casual_context_from_state(
            user_input, summary_history, recent_history
        )

        # Phase 05: 프롬프트 캡처
        casual_prompt = context[-1].content if context else ""
        system_prompt_text = context[0].content if context and isinstance(context[0], SystemMessage) else ""
        actual_prompts = {
            "system_prompt": system_prompt_text,
            "user_messages_count": 1,
            "context_turns": len(recent_history),
            "casual_prompt": casual_prompt,
        }

        response = self._llm.invoke(context)

        # 토큰 추적
        in_tokens, out_tokens = 0, 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            in_tokens = response.usage_metadata.get("input_tokens", 0)
            out_tokens = response.usage_metadata.get("output_tokens", 0)

        return {
            "messages": [response],
            "graph_path": graph_path,
            "input_tokens": state.get("input_tokens", 0) + in_tokens,
            "output_tokens": state.get("output_tokens", 0) + out_tokens,
            "actual_prompts": actual_prompts,
        }

    def _route_by_mode(self, state: ChatState) -> str:
        """router_node 이후 조건부 라우팅

        Returns:
            "casual_node" | "llm_node"
        """
        if state.get("is_casual"):
            return "casual_node"
        return "llm_node"

    def build(self):
        """그래프 빌드 및 컴파일 (Tool Calling 패턴)"""

        # 서비스가 주입된 도구 생성
        self._tools = create_tools_with_services(
            search_service=self.search_service,
            embedding_service=self.embedding_service,
            embedding_repo=self.embedding_repo,
            session_id=self._current_session_id,
            llm=self._llm,
            search_depth=self.search_depth,
            max_results=self.max_results,
        )

        # LLM에 도구 바인딩
        self._llm_with_tools = self._llm.bind_tools(self._tools)

        # ToolNode 생성
        tool_node = ToolNode(self._tools)

        # 그래프 빌더
        builder = StateGraph(ChatState)

        # 노드 추가 (5개)
        builder.add_node("summary_node", self._summary_node)
        builder.add_node("router_node", self._router_node)
        builder.add_node("casual_node", self._casual_node)
        builder.add_node("llm_node", self._llm_node)
        builder.add_node("tool_node", tool_node)

        # 엣지 정의: START → summary_node → router_node
        builder.add_edge(START, "summary_node")
        builder.add_edge("summary_node", "router_node")

        # router_node → casual_node 또는 llm_node
        builder.add_conditional_edges(
            "router_node",
            self._route_by_mode,
            {
                "casual_node": "casual_node",
                "llm_node": "llm_node",
            }
        )

        # casual_node → END
        builder.add_edge("casual_node", END)

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

    def _prepare_invocation(
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
    ) -> tuple[dict, dict]:
        """invoke()와 stream() 공통 전처리

        모드 감지는 router_node에서 수행하므로 여기서는 상태 구성만 담당합니다.

        Returns:
            (initial_state, config) 튜플. 항상 반환됩니다.
        """
        if normal_turn_ids is None:
            normal_turn_ids = []

        self._current_session_id = session_id
        if self._graph is None:
            self.build()

        # 메시지 변환 (domain.message.Message → LangChain BaseMessage)
        converted_messages = []
        for msg in (messages or []):
            if hasattr(msg, "role") and hasattr(msg, "content"):
                if msg.role == "user":
                    converted_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    converted_messages.append(AIMessage(content=msg.content))
            elif isinstance(msg, BaseMessage):
                converted_messages.append(msg)

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
            "normal_turn_count": len(normal_turn_ids),
            "normal_turn_ids": normal_turn_ids,
            # Phase 04: 교육용 메타데이터
            "graph_path": [],
            "summary_triggered": False,
            # Phase 05: 프롬프트 캡처
            "actual_prompts": {},
            # Router Node 통합: 기본값 (router_node이 덮어씀)
            "mode": "",
            "is_casual": False,
        }

        config = {"configurable": {"thread_id": session_id}}

        return initial_state, config

    def _extract_current_turn_messages(
        self, result_messages: list, turn_count: int
    ) -> list:
        """현재 턴의 메시지만 추출 (turn_id 기반)"""
        for i, msg in enumerate(result_messages):
            if (isinstance(msg, HumanMessage) and
                getattr(msg, "additional_kwargs", {}).get("turn_id") == turn_count):
                return result_messages[i:]

        # Fallback: 전체 반환
        return result_messages

    def _parse_result(
        self,
        result_messages: list,
        turn_count: int,
    ) -> dict:
        """invoke()와 stream() done 이벤트에서 공통 사용

        Returns:
            dict: text, tool_history, tool_results, (thought_process)
        """
        final_message = result_messages[-1] if result_messages else None
        final_text = ""
        thought_process = ""

        if final_message:
            raw_content = getattr(final_message, "content", "")
            if self.show_thoughts:
                thought_process, final_text = extract_thought_from_content(raw_content)
            else:
                final_text = extract_text_from_content(raw_content)

        current_turn_messages = self._extract_current_turn_messages(
            result_messages, turn_count
        )
        tool_history = []
        tool_results = {}
        for msg in current_turn_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_history.append(tc["name"])
            if hasattr(msg, "type") and msg.type == "tool":
                tool_results[msg.name] = msg.content

        result = {
            "text": final_text,
            "tool_history": tool_history,
            "tool_results": tool_results,
        }
        if thought_process:
            result["thought_process"] = thought_process
        return result

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
        if normal_turn_ids is None:
            normal_turn_ids = []

        initial_state, config = self._prepare_invocation(
            user_input, session_id, messages, summary, pdf_description,
            turn_count, summary_history, compression_rate, normal_turn_ids,
        )

        result = self._graph.invoke(initial_state, config)

        result_messages = result.get("messages", [])
        parsed = self._parse_result(result_messages, turn_count)

        input_tokens = result.get("input_tokens", 0)
        output_tokens = result.get("output_tokens", 0)
        mode = result.get("mode", "")
        is_casual = result.get("is_casual", False)
        updated_normal_turn_ids = result.get("normal_turn_ids", normal_turn_ids)

        # Phase 04: graph_path 강화 (tool_node 추론)
        graph_path = list(result.get("graph_path", []))
        if parsed["tool_history"]:
            enhanced = []
            llm_count = 0
            for node in graph_path:
                if node == "llm_node":
                    llm_count += 1
                    if llm_count > 1:  # 2번째+ llm_node = 이전에 tool 호출
                        enhanced.append("tool_node")
                enhanced.append(node)
            graph_path = enhanced

        return {
            **parsed,
            "iteration": len(parsed["tool_history"]),
            "model_used": self.model_name,
            "summary": result.get("summary", ""),
            "summary_history": result.get("summary_history", []),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "normal_turn_ids": updated_normal_turn_ids,
            "normal_turn_count": len(updated_normal_turn_ids),
            "mode": mode,
            "graph_path": graph_path,
            "summary_triggered": result.get("summary_triggered", False),
            "is_casual": is_casual,
            "actual_prompts": result.get("actual_prompts", {}),
            "error": None,
        }

    def stream(
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
    ) -> Generator[dict, None, None]:
        """그래프 스트리밍 실행

        Yields:
            dict: {"type": "token"|"tool_call"|"tool_result"|"thought"|"done", ...}
        """
        if normal_turn_ids is None:
            normal_turn_ids = []

        initial_state, config = self._prepare_invocation(
            user_input, session_id, messages, summary, pdf_description,
            turn_count, summary_history, compression_rate, normal_turn_ids,
        )

        tool_calls_buffer = []

        for chunk, metadata in self._graph.stream(
            initial_state, config, stream_mode="messages"
        ):
            for event in self._parse_message_chunk(chunk, metadata, tool_calls_buffer):
                yield event

        # 스트리밍 완료 후 공통 결과 파싱
        final_state = self._graph.get_state(config).values
        result_messages = final_state.get("messages", [])
        parsed = self._parse_result(result_messages, turn_count)

        mode = final_state.get("mode", "")
        is_casual = final_state.get("is_casual", False)
        updated_normal_turn_ids = final_state.get("normal_turn_ids", normal_turn_ids)

        # Phase 04: graph_path 강화 (tool_node 추론)
        graph_path = list(final_state.get("graph_path", []))
        if parsed["tool_history"]:
            enhanced = []
            llm_count = 0
            for node in graph_path:
                if node == "llm_node":
                    llm_count += 1
                    if llm_count > 1:
                        enhanced.append("tool_node")
                enhanced.append(node)
            graph_path = enhanced

        yield {
            "type": "done",
            "metadata": {
                **parsed,
                "model_used": self.model_name,
                "summary": final_state.get("summary", ""),
                "summary_history": final_state.get("summary_history", []),
                "input_tokens": final_state.get("input_tokens", 0),
                "output_tokens": final_state.get("output_tokens", 0),
                "normal_turn_ids": updated_normal_turn_ids,
                "normal_turn_count": len(updated_normal_turn_ids),
                "mode": mode,
                "graph_path": graph_path,
                "summary_triggered": final_state.get("summary_triggered", False),
                "is_casual": is_casual,
                "actual_prompts": final_state.get("actual_prompts", {}),
            }
        }

    def _parse_message_chunk(
        self, chunk, metadata: dict, tool_calls_buffer: list
    ) -> list[dict]:
        """stream_mode="messages" 이벤트 파싱

        Returns:
            list[dict]: 파싱된 이벤트 목록 (0~N개)
        """
        from langchain_core.messages import AIMessageChunk, ToolMessage

        events = []

        if isinstance(chunk, AIMessageChunk):
            # Phase 03-5: thought detection
            if self.show_thoughts and isinstance(chunk.content, list):
                for item in chunk.content:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if not text:
                            continue
                        if item.get("thought"):
                            events.append({"type": "thought", "content": text})
                        elif item.get("type") == "text":
                            events.append({"type": "token", "content": text})
                if events:
                    return events

            content = extract_text_from_content(chunk.content)
            if content:
                events.append({"type": "token", "content": content})

            if chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    name = tc.get("name")
                    if name:
                        tool_calls_buffer.append({"name": name})
                        events.append({"type": "tool_call", "name": name})

        elif isinstance(chunk, ToolMessage):
            events.append({
                "type": "tool_result",
                "name": getattr(chunk, "name", "unknown"),
                "content": str(chunk.content)[:500],
            })

        return events

