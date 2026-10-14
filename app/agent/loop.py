"""Hand-written tool-calling agent loop.

No LangChain. No LangGraph. No high-level frameworks.
A transparent, from-scratch ReAct/Tool-Calling agent loop implementing:
1. Message construction with system prompt and history
2. Tool schema injection via OpenAI function calling protocol
3. Tool call execution against registered deterministic financial tools
4. Error recovery: feeds tool error back to model once, aborting cleanly if persistent
5. Hard iteration cap (default 5)
6. Comprehensive execution trace for UI observability and automated evals
7. Input/output guardrails and numeric grounding validation
"""
import json
import time
from typing import Any, Dict, List, Optional
import pandas as pd
from app.config import settings
from app.llm.client import LLMClient, default_llm_client
from app.llm.prompts import AGENT_SYSTEM_PROMPT
from app.safety.pii import PIIVault
from app.safety.injection import sanitize_untrusted_text
from app.tools.registry import ToolRegistry, default_tool_registry
from app.agent.guardrails import (
    validate_input_query,
    validate_output_safety,
    validate_numeric_grounding,
)
from app.schemas import AgentQueryResponse, AgentTrace, ToolCallRecord


class AgentLoop:
    """Hand-written tool-calling agent loop over financial statements."""

    def __init__(
        self,
        df: pd.DataFrame,
        vault: Optional[PIIVault] = None,
        registry: Optional[ToolRegistry] = None,
        client: Optional[LLMClient] = None,
        max_iterations: Optional[int] = None,
    ):
        self.df = df
        self.vault = vault or PIIVault()
        self.registry = registry or default_tool_registry
        self.client = client or default_llm_client
        self.max_iterations = max_iterations or settings.max_agent_iterations

    def run(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> AgentQueryResponse:
        """Execute the agent loop for a user question."""
        start_time = time.time()
        trace = AgentTrace()
        tool_results_for_grounding: List[Any] = []

        # 1. Input Guardrail
        is_valid_input, input_warnings = validate_input_query(question)
        if input_warnings:
            trace.security_flags.extend(input_warnings)

        if not is_valid_input:
            return AgentQueryResponse(
                question=question,
                answer=f"Invalid request: {', '.join(input_warnings)}",
                trace=trace,
                grounded=False,
            )

        # Redact any PII the user may have typed into their question
        redacted_question = self.vault.redact_text(question)
        sanitized_question = sanitize_untrusted_text(redacted_question)

        # 2. Build Initial Messages
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]

        if chat_history:
            for item in chat_history:
                messages.append({"role": item["role"], "content": item["content"]})

        messages.append({"role": "user", "content": sanitized_question})

        openai_tools = self.registry.get_openai_tools()
        iteration = 0
        final_answer = ""
        consecutive_tool_errors = 0

        # 3. Execution Loop
        while iteration < self.max_iterations:
            iteration += 1

            # Call LLM with tool schemas
            try:
                resp = self.client.chat_completion(
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    temperature=0.0,
                )
            except Exception as e:
                trace.security_flags.append(f"LLM communication error: {str(e)}")
                final_answer = "I encountered an error communicating with the language model. Please try again."
                break

            msg = resp.choices[0].message

            # Track tokens
            if hasattr(resp, "usage") and resp.usage:
                trace.total_prompt_tokens += resp.usage.prompt_tokens or 0
                trace.total_completion_tokens += resp.usage.completion_tokens or 0

            # Case A: Model wants to invoke one or more tools
            if msg.tool_calls:
                # Append assistant's tool-call intent to conversation
                assistant_tool_call_dicts = []
                for tc in msg.tool_calls:
                    assistant_tool_call_dicts.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })

                messages.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": assistant_tool_call_dicts,
                })

                # Execute each tool
                has_error_in_step = False
                for tc in msg.tool_calls:
                    t_name = tc.function.name
                    t_args_raw = tc.function.arguments
                    try:
                        t_args = json.loads(t_args_raw) if isinstance(t_args_raw, str) else t_args_raw
                    except Exception:
                        t_args = {}

                    t0 = time.time()
                    try:
                        res = self.registry.execute(t_name, df=self.df, **t_args)
                        status = "success"
                        err_msg = None
                        tool_results_for_grounding.append(res)
                        res_str = json.dumps(res, default=str)
                        consecutive_tool_errors = 0
                    except Exception as err:
                        status = "error"
                        err_msg = str(err)
                        res_str = json.dumps({"error": f"Tool execution failed: {str(err)}"})
                        has_error_in_step = True
                        consecutive_tool_errors += 1

                    latency = (time.time() - t0) * 1000.0

                    # Record trace
                    trace.steps.append(
                        ToolCallRecord(
                            iteration=iteration,
                            tool_name=t_name,
                            arguments=t_args,
                            result=res if status == "success" else {"error": err_msg},
                            latency_ms=round(latency, 2),
                            status=status,
                            error_message=err_msg,
                        )
                    )

                    # Append tool response
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": t_name,
                        "content": res_str,
                    })

                # If same tool error repeated, abort cleanly to avoid infinite loop
                if consecutive_tool_errors >= 2:
                    final_answer = "I was unable to complete the analysis because the required tool encountered an error."
                    break

            # Case B: Model returned a final textual answer
            elif msg.content:
                final_answer = msg.content
                break

        # Check iteration cap
        if not final_answer and iteration >= self.max_iterations:
            final_answer = (
                f"I reached the maximum limit of {self.max_iterations} analytical steps. "
                "Based on the tools called, please review the execution trace for preliminary data."
            )

        # 4. Output Guardrails
        is_safe, final_answer, leak_flags = validate_output_safety(final_answer, self.vault)
        if leak_flags:
            trace.security_flags.extend(leak_flags)

        # 5. Numeric Grounding Guardrail
        is_grounded, grounding_warnings = validate_numeric_grounding(
            final_answer,
            tool_results_for_grounding,
        )
        if grounding_warnings:
            trace.grounding_warnings.extend(grounding_warnings)

        trace.total_iterations = iteration
        trace.total_latency_ms = round((time.time() - start_time) * 1000.0, 2)

        return AgentQueryResponse(
            question=question,
            answer=final_answer,
            trace=trace,
            grounded=is_grounded,
        )
