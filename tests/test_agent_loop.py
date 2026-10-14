"""Unit tests for the hand-written agent loop using mocked LLM responses."""
import json
from unittest.mock import MagicMock
import pandas as pd
import pytest
from app.agent.loop import AgentLoop
from app.ingestion.synthetic import generate_synthetic_transactions
from app.safety.pii import PIIVault
from app.tools.registry import create_default_tool_registry


@pytest.fixture
def mock_df():
    """Create small DataFrame for loop testing."""
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-03"],
        "description": ["TechCorp Salary", "Landlord Rent"],
        "amount": [125000.0, 32000.0],
        "type": ["credit", "debit"],
        "balance": [125000.0, 93000.0],
        "category": ["Salary", "Housing & Rent"],
    })


def test_agent_loop_successful_tool_call(mock_df):
    """Test standard tool call flow: user question -> model calls tool -> tool executes -> model returns final answer."""
    mock_client = MagicMock()

    # Step 1 response: Tool call request
    resp1 = MagicMock()
    choice1 = MagicMock()
    tc1 = MagicMock()
    tc1.id = "call_abc123"
    tc1.function.name = "get_spending_summary"
    tc1.function.arguments = json.dumps({"start_date": "2024-01-01"})
    choice1.message.content = None
    choice1.message.tool_calls = [tc1]
    resp1.choices = [choice1]
    resp1.usage.prompt_tokens = 50
    resp1.usage.completion_tokens = 20

    # Step 2 response: Final answer referencing tool result
    resp2 = MagicMock()
    choice2 = MagicMock()
    choice2.message.content = "Your total spending was ₹32,000.00 and total income was ₹125,000.00."
    choice2.message.tool_calls = None
    resp2.choices = [choice2]
    resp2.usage.prompt_tokens = 100
    resp2.usage.completion_tokens = 30

    mock_client.chat_completion.side_effect = [resp1, resp2]

    loop = AgentLoop(df=mock_df, client=mock_client)
    res = loop.run("How much did I spend in January?")

    assert "₹32,000.00" in res.answer
    assert res.trace.total_iterations == 2
    assert len(res.trace.steps) == 1
    assert res.trace.steps[0].tool_name == "get_spending_summary"
    assert res.trace.steps[0].status == "success"
    assert res.grounded is True


def test_agent_loop_iteration_limit(mock_df):
    """Verify agent loop enforces hard iteration cap and terminates gracefully."""
    mock_client = MagicMock()

    # Infinite tool calling response
    resp = MagicMock()
    choice = MagicMock()
    tc = MagicMock()
    tc.id = "call_loop"
    tc.function.name = "get_spending_summary"
    tc.function.arguments = "{}"
    choice.message.content = None
    choice.message.tool_calls = [tc]
    resp.choices = [choice]
    resp.usage.prompt_tokens = 20
    resp.usage.completion_tokens = 10

    mock_client.chat_completion.return_value = resp

    loop = AgentLoop(df=mock_df, client=mock_client, max_iterations=3)
    res = loop.run("Tell me everything about my account")

    assert res.trace.total_iterations == 3
    assert len(res.trace.steps) == 3
    assert "maximum limit of 3 analytical steps" in res.answer


def test_agent_loop_tool_error_handling(mock_df):
    """Verify agent handles tool errors by feeding error message to LLM and recovering."""
    mock_client = MagicMock()

    # Model attempts to call tool with bad parameters or invalid tool
    resp1 = MagicMock()
    choice1 = MagicMock()
    tc1 = MagicMock()
    tc1.id = "call_err"
    tc1.function.name = "simulate_budget"
    tc1.function.arguments = json.dumps({"category": "NonExistent", "reduction_percent": -50.0})
    choice1.message.content = None
    choice1.message.tool_calls = [tc1]
    resp1.choices = [choice1]
    resp1.usage.prompt_tokens = 30
    resp1.usage.completion_tokens = 15

    # Model provides explanation after receiving error
    resp2 = MagicMock()
    choice2 = MagicMock()
    choice2.message.content = "I could not simulate the budget because the reduction percentage must be positive."
    choice2.message.tool_calls = None
    resp2.choices = [choice2]
    resp2.usage.prompt_tokens = 60
    resp2.usage.completion_tokens = 20

    mock_client.chat_completion.side_effect = [resp1, resp2]

    loop = AgentLoop(df=mock_df, client=mock_client)
    res = loop.run("Simulate -50% budget")

    assert "could not simulate" in res.answer
    assert res.trace.steps[0].status == "success" or res.trace.steps[0].result.get("error") is not None

