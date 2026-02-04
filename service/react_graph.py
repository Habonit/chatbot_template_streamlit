"""Phase 02-5: ReAct + 노드별 프롬프트 분리 그래프 + SqliteSaver + Summary Node

노드 구성:
- Summary Node: 요약 필요 여부 판단 및 생성
- Tool Selector: 다음에 실행할 툴 선택
- Time Tool / RAG Tool / Search Tool / Reasoning Tool: 툴 실행
- Result Processor: 결과 해석 + 추가 툴 필요 여부 판단
- Response Generator: 최종 응답 생성
"""
from typing import TypedDict, Literal, Any
from datetime import datetime
from pathlib import Path
import json
import sqlite3
import pytz

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from prompt.selector.tool_selector import get_prompt as get_tool_selector_prompt
from prompt.tools.reasoning import get_prompt as get_reasoning_prompt
from prompt.processor.result_processor import get_prompt as get_result_processor_prompt
from prompt.response.response_generator import get_prompt as get_response_generator_prompt
from prompt.summary.summary_generator import get_prompt as get_summary_prompt


class ChatState(TypedDict):
    """ReAct 루프 전체에서 공유되는 상태"""

    # === 입력 ===
    user_input: str
    session_id: str

    # === 컨텍스트 ===
    messages: list[BaseMessage]
    summary: str
    pdf_description: str

    # === Phase 02-5: 요약 관련 필드 ===
    turn_count: int
    summary_history: list[dict]
    messages_for_context: list[BaseMessage]

    # === ReAct 루프 상태 ===
    current_tool: str
    tool_history: list[str]
    iteration: int
    max_iterations: int

    # === 툴 실행 결과 ===
    tool_results: dict[str, str]

    # === 프로세서 판단 ===
    needs_more_tools: bool
    processor_summary: str

    # === 최종 출력 ===
    final_response: str

    # === 메타데이터 ===
    input_tokens: int
    output_tokens: int
    model_used: str


def route_to_selected_tool(state: ChatState) -> str:
    """Tool Selector가 선택한 툴로 라우팅"""
    return state["current_tool"]


def should_continue_loop(state: ChatState) -> Literal["continue", "finish"]:
    """Result Processor 판단에 따라 루프 계속 or 종료"""
    if state["needs_more_tools"] and state["iteration"] < state["max_iterations"]:
        return "continue"
    return "finish"


def should_summarize(turn_count: int) -> bool:
    """요약 필요 여부 판단

    Turn 4, 7, 10, 13, ... 에서 요약 트리거
    - Turn 4: 1,2,3 요약
    - Turn 7: 4,5,6 요약
    - Turn 10: 7,8,9 요약

    Args:
        turn_count: 현재 턴 수

    Returns:
        bool: 요약이 필요하면 True
    """
    if turn_count < 4:
        return False
    return (turn_count - 1) % 3 == 0


class ReactGraphBuilder:
    """ReAct + 노드별 프롬프트 분리 그래프 빌더 (Phase 02-5: SqliteSaver + Summary Node)"""

    DEFAULT_DB_PATH = "data/langgraph.db"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_iterations: int = 5,
        search_service: Any = None,
        embedding_service: Any = None,
        embedding_repo: Any = None,
        db_path: str = None,
    ):
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.search_service = search_service
        self.embedding_service = embedding_service
        self.embedding_repo = embedding_repo
        self.db_path = db_path or self.DEFAULT_DB_PATH

        # DB 디렉토리 생성 (파일 경로인 경우)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # SqliteSaver 초기화 (sqlite3 connection 사용)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._conn)
        self._graph = None

        # LLM 초기화
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
        )

    def _invoke_llm_with_token_tracking(
        self, messages: list[BaseMessage], llm=None
    ) -> tuple[str, int, int]:
        """LLM 호출 + 토큰 사용량 추적

        Args:
            messages: LLM에 전달할 메시지 목록
            llm: 사용할 LLM (None이면 self._llm 사용)

        Returns:
            tuple: (응답 텍스트, 입력 토큰, 출력 토큰)
        """
        target_llm = llm or self._llm
        response = target_llm.invoke(messages)

        input_tokens = 0
        output_tokens = 0

        # LangChain ChatGoogleGenerativeAI 응답에서 토큰 정보 추출
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = response.usage_metadata.get("input_tokens", 0)
            output_tokens = response.usage_metadata.get("output_tokens", 0)
        elif hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata
            if "usage_metadata" in metadata:
                input_tokens = metadata["usage_metadata"].get("prompt_token_count", 0)
                output_tokens = metadata["usage_metadata"].get("candidates_token_count", 0)

        return response.content, input_tokens, output_tokens

    def _summary_node(self, state: ChatState) -> dict:
        """Summary Node: 요약 필요 여부 판단 및 생성"""
        turn_count = state.get("turn_count", 0)
        messages = state.get("messages", [])
        current_summary = state.get("summary", "")
        summary_history = state.get("summary_history", []).copy()

        # 요약 불필요한 경우
        if not should_summarize(turn_count):
            return {
                "messages_for_context": messages,
                "summary": current_summary,
                "summary_history": summary_history,
            }

        # 요약 필요한 경우: 이전 3개 턴 요약
        # to_summarize: turn_count-3 ~ turn_count-1
        # to_keep: turn_count (현재 턴)
        if len(messages) < 4:
            return {
                "messages_for_context": messages,
                "summary": current_summary,
                "summary_history": summary_history,
            }

        # 메시지에서 요약할 부분과 유지할 부분 분리
        # 각 턴은 user + assistant 2개의 메시지
        to_summarize = messages[:-2]  # 마지막 턴 제외
        to_keep = messages[-2:]  # 마지막 턴 유지

        # 요약할 대화 텍스트 구성
        conversation_parts = []
        for msg in to_summarize[-6:]:  # 최대 3턴 (6개 메시지)
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
            conversation_parts.append(f"{role}: {content}")
        conversation = "\n".join(conversation_parts)

        # LLM으로 요약 생성
        prompt = get_summary_prompt(
            previous_summary=current_summary,
            conversation=conversation,
        )
        content, in_tokens, out_tokens = self._invoke_llm_with_token_tracking(
            [HumanMessage(content=prompt)]
        )
        new_summary = content.strip()

        # 토큰 누적
        total_input = state.get("input_tokens", 0) + in_tokens
        total_output = state.get("output_tokens", 0) + out_tokens

        # summary_history 업데이트
        start_turn = max(1, turn_count - 3)
        end_turn = turn_count - 1
        summary_history.append({
            "created_at_turn": turn_count,
            "covers_turns": f"{start_turn}-{end_turn}",
            "summary": new_summary,
        })

        # 요약을 SystemMessage로 포함
        summary_message = SystemMessage(content=f"[이전 대화 요약]\n{new_summary}")
        messages_for_context = [summary_message] + to_keep

        return {
            "summary": new_summary,
            "summary_history": summary_history,
            "messages_for_context": messages_for_context,
            "input_tokens": total_input,
            "output_tokens": total_output,
        }

    def _tool_selector_node(self, state: ChatState) -> dict:
        """Tool Selector 노드: 다음에 실행할 툴 선택"""
        # 툴 결과 요약
        tool_results_summary = ""
        if state.get("tool_results"):
            summaries = []
            for tool_name, result in state["tool_results"].items():
                summaries.append(f"[{tool_name}]: {result[:200]}...")
            tool_results_summary = "\n".join(summaries)

        prompt = get_tool_selector_prompt(
            user_input=state["user_input"],
            tool_history=state.get("tool_history", []),
            tool_results_summary=tool_results_summary,
            pdf_description=state.get("pdf_description", ""),
        )

        content, in_tokens, out_tokens = self._invoke_llm_with_token_tracking(
            [HumanMessage(content=prompt)]
        )
        selected_tool = content.strip().lower()

        # 유효한 툴 이름인지 확인
        valid_tools = ["get_current_time", "web_search", "search_pdf_knowledge", "reasoning", "none"]
        if selected_tool not in valid_tools:
            selected_tool = "none"

        return {
            "current_tool": selected_tool,
            "input_tokens": state.get("input_tokens", 0) + in_tokens,
            "output_tokens": state.get("output_tokens", 0) + out_tokens,
        }

    def _time_tool_node(self, state: ChatState) -> dict:
        """Time Tool 노드: 현재 시각 반환"""
        kst = pytz.timezone("Asia/Seoul")
        now = datetime.now(kst)
        result = now.strftime("%Y-%m-%d %H:%M:%S") + " (KST)"

        tool_results = state.get("tool_results", {}).copy()
        tool_results["get_current_time"] = result

        tool_history = state.get("tool_history", []).copy()
        tool_history.append("get_current_time")

        return {
            "tool_results": tool_results,
            "tool_history": tool_history,
            "iteration": state.get("iteration", 0) + 1,
        }

    def _search_tool_node(self, state: ChatState) -> dict:
        """Search Tool 노드: 웹 검색"""
        if self.search_service:
            results = self.search_service.search(state["user_input"])
            result = self.search_service.format_for_llm(results)
        else:
            result = "웹 검색 서비스가 설정되지 않았습니다."

        tool_results = state.get("tool_results", {}).copy()
        tool_results["web_search"] = result

        tool_history = state.get("tool_history", []).copy()
        tool_history.append("web_search")

        return {
            "tool_results": tool_results,
            "tool_history": tool_history,
            "iteration": state.get("iteration", 0) + 1,
        }

    def _rag_tool_node(self, state: ChatState) -> dict:
        """RAG Tool 노드: PDF 검색"""
        if self.embedding_service and self.embedding_repo:
            query_embedding = self.embedding_service.embed_query(state["user_input"])
            results = self.embedding_repo.search_similar(
                state["session_id"], query_embedding, top_k=5
            )

            if results:
                formatted = []
                for i, r in enumerate(results, 1):
                    formatted.append(f"[{i}] (유사도: {r['score']:.2f})\n{r['chunk'].normalized_text}")
                result = "\n\n".join(formatted)
            else:
                result = "PDF에서 관련 내용을 찾을 수 없습니다."
        else:
            result = "PDF 검색 서비스가 설정되지 않았습니다."

        tool_results = state.get("tool_results", {}).copy()
        tool_results["search_pdf_knowledge"] = result

        tool_history = state.get("tool_history", []).copy()
        tool_history.append("search_pdf_knowledge")

        return {
            "tool_results": tool_results,
            "tool_history": tool_history,
            "iteration": state.get("iteration", 0) + 1,
        }

    def _reasoning_tool_node(self, state: ChatState) -> dict:
        """Reasoning Tool 노드: 단계별 추론"""
        # 컨텍스트 구성
        context = ""
        if state.get("tool_results"):
            context_parts = []
            for tool_name, result in state["tool_results"].items():
                context_parts.append(f"[{tool_name} 결과]\n{result}")
            context = "\n\n".join(context_parts)

        prompt = get_reasoning_prompt(
            user_input=state["user_input"],
            context=context,
        )

        content, in_tokens, out_tokens = self._invoke_llm_with_token_tracking(
            [HumanMessage(content=prompt)]
        )
        result = content

        tool_results = state.get("tool_results", {}).copy()
        tool_results["reasoning"] = result

        tool_history = state.get("tool_history", []).copy()
        tool_history.append("reasoning")

        return {
            "tool_results": tool_results,
            "tool_history": tool_history,
            "iteration": state.get("iteration", 0) + 1,
            "input_tokens": state.get("input_tokens", 0) + in_tokens,
            "output_tokens": state.get("output_tokens", 0) + out_tokens,
        }

    def _result_processor_node(self, state: ChatState) -> dict:
        """Result Processor 노드: 결과 해석 + 추가 툴 필요 여부 판단"""
        prompt = get_result_processor_prompt(
            user_input=state["user_input"],
            tool_history=state.get("tool_history", []),
            tool_results=state.get("tool_results", {}),
            iteration=state.get("iteration", 0),
            max_iterations=state.get("max_iterations", self.max_iterations),
        )

        content, in_tokens, out_tokens = self._invoke_llm_with_token_tracking(
            [HumanMessage(content=prompt)]
        )

        # JSON 파싱 시도
        try:
            result_text = content.strip()
            # JSON 부분만 추출 시도
            if "{" in result_text and "}" in result_text:
                json_start = result_text.find("{")
                json_end = result_text.rfind("}") + 1
                json_str = result_text[json_start:json_end]
                parsed = json.loads(json_str)
            else:
                parsed = json.loads(result_text)

            needs_more_tools = parsed.get("needs_more_tools", False)
            processor_summary = parsed.get("summary", "")
        except (json.JSONDecodeError, AttributeError):
            # 파싱 실패 시 기본값
            needs_more_tools = False
            processor_summary = content

        return {
            "needs_more_tools": needs_more_tools,
            "processor_summary": processor_summary,
            "input_tokens": state.get("input_tokens", 0) + in_tokens,
            "output_tokens": state.get("output_tokens", 0) + out_tokens,
        }

    def _response_generator_node(self, state: ChatState) -> dict:
        """Response Generator 노드: 최종 응답 생성"""
        # 수집된 정보 구성
        collected_info = ""
        if state.get("tool_results"):
            info_parts = []
            for tool_name, result in state["tool_results"].items():
                info_parts.append(f"[{tool_name} 결과]\n{result}")
            collected_info = "\n\n".join(info_parts)

        prompt = get_response_generator_prompt(
            user_input=state["user_input"],
            collected_info=collected_info,
            processor_summary=state.get("processor_summary", ""),
        )

        content, in_tokens, out_tokens = self._invoke_llm_with_token_tracking(
            [HumanMessage(content=prompt)]
        )

        return {
            "final_response": content,
            "input_tokens": state.get("input_tokens", 0) + in_tokens,
            "output_tokens": state.get("output_tokens", 0) + out_tokens,
        }

    def build(self):
        """그래프 빌드 및 컴파일"""
        builder = StateGraph(ChatState)

        # 노드 추가 (Phase 02-5: summary_node 추가)
        builder.add_node("summary_node", self._summary_node)
        builder.add_node("tool_selector", self._tool_selector_node)
        builder.add_node("time_tool", self._time_tool_node)
        builder.add_node("search_tool", self._search_tool_node)
        builder.add_node("rag_tool", self._rag_tool_node)
        builder.add_node("reasoning_tool", self._reasoning_tool_node)
        builder.add_node("result_processor", self._result_processor_node)
        builder.add_node("response_generator", self._response_generator_node)

        # START → Summary Node → Tool Selector
        builder.add_edge(START, "summary_node")
        builder.add_edge("summary_node", "tool_selector")

        # Tool Selector → 선택된 툴 (조건부)
        builder.add_conditional_edges(
            "tool_selector",
            route_to_selected_tool,
            {
                "get_current_time": "time_tool",
                "web_search": "search_tool",
                "search_pdf_knowledge": "rag_tool",
                "reasoning": "reasoning_tool",
                "none": "response_generator",
            }
        )

        # 각 툴 → Result Processor
        builder.add_edge("time_tool", "result_processor")
        builder.add_edge("search_tool", "result_processor")
        builder.add_edge("rag_tool", "result_processor")
        builder.add_edge("reasoning_tool", "result_processor")

        # Result Processor → 루프 or 응답 (조건부)
        builder.add_conditional_edges(
            "result_processor",
            should_continue_loop,
            {
                "continue": "tool_selector",
                "finish": "response_generator",
            }
        )

        # Response Generator → END
        builder.add_edge("response_generator", END)

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

        Returns:
            dict: 응답 결과
        """
        # Phase 02-7: Fast-path - 일상적 대화는 그래프 실행 없이 바로 응답
        from service.reasoning_detector import detect_reasoning_need

        mode = detect_reasoning_need(user_input)
        if mode == "casual":
            # 간단한 LLM 호출로 자연스러운 응답 생성
            casual_prompt = f"""사용자가 "{user_input}"라고 말했습니다.
자연스럽고 친근하게 짧게 응답해주세요. 분석이나 설명 없이요."""

            content, in_tokens, out_tokens = self._invoke_llm_with_token_tracking(
                [HumanMessage(content=casual_prompt)]
            )

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
                "error": None,
            }

        if self._graph is None:
            self.build()

        initial_state: ChatState = {
            "user_input": user_input,
            "session_id": session_id,
            "messages": messages or [],
            "summary": summary,
            "pdf_description": pdf_description,
            # Phase 02-5: 요약 관련 필드
            "turn_count": turn_count,
            "summary_history": summary_history or [],
            "messages_for_context": messages or [],
            # ReAct 루프 상태
            "current_tool": "",
            "tool_history": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "tool_results": {},
            "needs_more_tools": False,
            "processor_summary": "",
            "final_response": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "model_used": self.model_name,
        }

        config = {"configurable": {"thread_id": session_id}}
        result = self._graph.invoke(initial_state, config)

        input_tokens = result.get("input_tokens", 0)
        output_tokens = result.get("output_tokens", 0)

        return {
            "text": result.get("final_response", ""),
            "tool_history": result.get("tool_history", []),
            "tool_results": result.get("tool_results", {}),
            "iteration": result.get("iteration", 0),
            "model_used": self.model_name,
            # Phase 02-5: 요약 결과 반환
            "summary": result.get("summary", ""),
            "summary_history": result.get("summary_history", []),
            # Phase 02-7: 토큰 사용량 반환
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "error": None,
        }
