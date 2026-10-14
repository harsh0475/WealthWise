"""Smoke test executing 5 real questions against the local Ollama model.

Verifies:
1. Local Ollama OpenAI-compatible endpoint connectivity (/v1)
2. Tool calling schema parsing and execution
3. Output extraction and numeric grounding
4. Trace generation
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from app.config import settings
from app.ingestion.synthetic import generate_and_save_synthetic_statement
from app.safety.pii import PIIVault
from app.categorization.categorizer import categorize_statement
from app.agent.loop import AgentLoop
from app.llm.client import LLMClient


SMOKE_QUESTIONS = [
    "What was my total spending in 2024?",
    "Who are my top 3 merchants by spend?",
    "What are my recurring monthly payments and subscriptions?",
    "Were there any unusual transactions or anomalies in my statement?",
    "If I cut Food & Dining spending by 20%, how much will I save each month?",
]


def run_smoke_test(model_name: str = "qwen2.5:1.5b"):
    print(f"==================================================")
    print(f" Starting 5-Question Smoke Test on Local Ollama")
    print(f" Model: {model_name} | Endpoint: http://localhost:11434/v1")
    print(f"==================================================\n")

    # 1. Prepare data
    raw_df = generate_and_save_synthetic_statement()
    vault = PIIVault()
    redacted_df = vault.redact_dataframe(raw_df, account_holder_name="Aarav Sharma")
    categorized_df = categorize_statement(redacted_df)

    # 2. Configure client pointing to local Ollama
    client = LLMClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model=model_name,
        temperature=0.0,
    )
    client.clear_outbound_history()

    agent = AgentLoop(
        df=categorized_df,
        vault=vault,
        client=client,
        max_iterations=4,
    )

    passed_count = 0
    for idx, q in enumerate(SMOKE_QUESTIONS, 1):
        print(f"[{idx}/5] Question: {q}")
        try:
            resp = agent.run(q)
            tools_called = [step.tool_name for step in resp.trace.steps]
            print(f"      -> Tools Called: {tools_called}")
            print(f"      -> Grounded: {resp.grounded}")
            print(f"      -> Answer: {resp.answer[:160]}...")
            print(f"      -> Latency: {resp.trace.total_latency_ms:.0f} ms | Steps: {resp.trace.total_iterations}\n")
            if len(tools_called) > 0 and len(resp.answer.strip()) > 10:
                passed_count += 1
        except Exception as e:
            print(f"      -> ERROR: {e}\n")

    print(f"Smoke Test Complete: {passed_count}/5 questions successfully executed tool calls.")
    return passed_count


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:1.5b"
    run_smoke_test(model)

