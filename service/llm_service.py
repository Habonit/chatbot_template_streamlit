from google import genai
from google.genai import types
from typing import Callable


class LLMService:
    def __init__(
        self,
        api_key: str,
        default_model: str = "gemini-2.5-flash",
    ):
        self.client = genai.Client(api_key=api_key)
        self.default_model = default_model

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        tools: list[Callable] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict:
        model = model or self.default_model

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
        )

        if tools:
            config.tools = tools

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        result = {
            "text": response.text if hasattr(response, "text") else "",
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
            "model_used": model,
            "function_calls": [],
        }

        if hasattr(response, "function_calls") and response.function_calls:
            result["function_calls"] = [
                {"name": fc.name, "args": dict(fc.args)}
                for fc in response.function_calls
            ]

        return result

    def generate_with_history(
        self,
        messages: list[dict],
        model: str | None = None,
        tools: list[Callable] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict:
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        model = model or self.default_model

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
        )

        if tools:
            config.tools = tools

        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        result = {
            "text": response.text if hasattr(response, "text") else "",
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
            "model_used": model,
            "function_calls": [],
        }

        if hasattr(response, "function_calls") and response.function_calls:
            result["function_calls"] = [
                {"name": fc.name, "args": dict(fc.args)}
                for fc in response.function_calls
            ]

        return result

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        try:
            client = genai.Client(api_key=api_key)
            client.models.list()
            return True
        except Exception:
            return False
