"""OpenAI-compatible LLM client wrapper.

Wraps the OpenAI SDK for use with Groq or Ollama endpoints.
Provides:
- Outbound request recording (for audit logs and PII leak evaluation)
- Configurable timeouts and retries
- Token budgeting and tracking
- Mocking support for zero-network testing
"""
import time
from typing import Any, Dict, List, Optional
import httpx
from openai import OpenAI
from app.config import settings


class LLMClient:
    """Wrapper around OpenAI-compatible API client with request recording and retries."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.top_p = top_p if top_p is not None else settings.llm_top_p
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = timeout or settings.llm_timeout_seconds

        # Outbound request history for safety verification and PII leak evals
        self.outbound_requests: List[Dict[str, Any]] = []

        # Initialize underlying OpenAI SDK client
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(self.timeout),
        )

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = "auto",
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
    ) -> Any:
        """Call chat completion with automatic retries and outbound request recording."""
        temp = self.temperature if temperature is None else temperature

        # Prepare request payload
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format

        # Record outbound payload BEFORE making the network call
        request_record: Dict[str, Any] = {
            "timestamp": time.time(),
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": temp,
        }

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(**payload)
                # Record response metadata
                request_record["status"] = "success"
                if hasattr(response, "usage") and response.usage:
                    request_record["prompt_tokens"] = response.usage.prompt_tokens
                    request_record["completion_tokens"] = response.usage.completion_tokens
                self.outbound_requests.append(request_record)
                return response
            except Exception as e:
                last_error = e
                time.sleep(0.5 * (attempt + 1))

        request_record["status"] = "failed"
        request_record["error"] = str(last_error)
        self.outbound_requests.append(request_record)
        raise RuntimeError(f"LLM chat completion failed after {max_retries + 1} attempts: {last_error}")

    def clear_outbound_history(self) -> None:
        """Clear recorded requests."""
        self.outbound_requests.clear()

    def get_outbound_requests(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded outbound requests."""
        return list(self.outbound_requests)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Approximate token count for budgeting context window (~4 chars per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)


# Default shared client instance
default_llm_client = LLMClient()

