from domain.message import Message


class SummaryService:
    TURNS_PER_SUMMARY = 3

    def should_summarize(self, turn_count: int) -> bool:
        return turn_count > self.TURNS_PER_SUMMARY

    def get_turns_to_summarize(
        self,
        messages: list[Message],
        turn_count: int,
    ) -> tuple[list[Message], list[Message]]:
        if turn_count <= self.TURNS_PER_SUMMARY:
            return [], messages

        messages_per_turn = 2
        keep_count = (turn_count % self.TURNS_PER_SUMMARY) or self.TURNS_PER_SUMMARY
        keep_messages = keep_count * messages_per_turn

        to_summarize = messages[:-keep_messages] if keep_messages < len(messages) else []
        to_keep = messages[-keep_messages:] if keep_messages <= len(messages) else messages

        return to_summarize, to_keep

    def build_summary_prompt(
        self,
        previous_summary: str,
        conversation: list[Message],
    ) -> str:
        conv_text = "\n".join([f"{m.role}: {m.content}" for m in conversation])

        return f"""다음 대화 내용을 간결하게 요약하세요.
핵심 정보, 사용자의 요청 사항, 중요한 결정 사항을 포함해야 합니다.
200자 이내로 작성하세요.

이전 요약 (있는 경우):
{previous_summary}

추가할 대화:
{conv_text}

통합 요약:"""

    def build_context(
        self,
        messages: list[Message],
        summary: str = "",
        system_prompt: str = "",
    ) -> str:
        parts = []

        if system_prompt:
            parts.append(f"[System Prompt]\n{system_prompt}")

        if summary:
            parts.append(f"[누적 요약문]\n{summary}")
            parts.append("---")

        if messages:
            parts.append("[최근 대화 원문]")
            for msg in messages:
                role = "User" if msg.role == "user" else "Assistant"
                parts.append(f"- {role}: {msg.content}")
            parts.append("---")

        return "\n".join(parts)
