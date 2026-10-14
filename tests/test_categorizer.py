"""Unit tests for transaction categorizer (rules-first + mocked LLM fallback)."""
from unittest.mock import MagicMock
import pandas as pd
import pytest
from app.categorization.categorizer import match_rules, categorize_statement, categorize_unmatched_with_llm
from app.schemas import StandardCategory


def test_match_rules_obvious_merchants():
    """Verify deterministic regex rules categorize standard merchants correctly."""
    assert match_rules("TechCorp India Pvt Ltd Salary credit") == StandardCategory.SALARY.value
    assert match_rules("UPI/DR/123/Rent transfer to Landlord") == StandardCategory.HOUSING_AND_RENT.value
    assert match_rules("ACH DEBIT-HDFC BANK PERSONAL LOAN EMI") == StandardCategory.EMI_AND_LOANS.value
    assert match_rules("Blinkit Quick Commerce Bangalore") == StandardCategory.GROCERIES.value
    assert match_rules("UPI/DR/3910/Swiggy Food Bangalore") == StandardCategory.FOOD_AND_DINING.value
    assert match_rules("AUTOPAY-Spotify India Premium") == StandardCategory.SUBSCRIPTIONS.value
    assert match_rules("BILLPAY-BESCOM Electricity") == StandardCategory.UTILITIES.value
    assert match_rules("POS/DR/CARD/Uber India Systems") == StandardCategory.TRAVEL_AND_COMMUTE.value
    assert match_rules("Tanishq Jewellers Indiranagar") == StandardCategory.SHOPPING.value
    assert match_rules("ACH DEBIT-Zerodha Broking Coin SIP") == StandardCategory.INVESTMENTS.value
    assert match_rules("UNKNOWN_MYSTERIOUS_STORE_XYZ") is None


def test_categorize_statement_rules_only():
    """Verify statement categorization executes rules without requiring LLM for obvious rows."""
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-03", "2024-01-05"],
        "description": [
            "TechCorp Salary credit",
            "Landlord Rent payment",
            "Swiggy Food dinner"
        ],
        "amount": [125000.0, 32000.0, 450.0],
        "type": ["credit", "debit", "debit"],
        "balance": [125000.0, 93000.0, 92550.0]
    })
    cat_df = categorize_statement(df, client=None)
    assert cat_df["category"].tolist() == [
        StandardCategory.SALARY.value,
        StandardCategory.HOUSING_AND_RENT.value,
        StandardCategory.FOOD_AND_DINING.value
    ]


def test_categorize_unmatched_with_mocked_llm():
    """Verify LLM fallback is called and structured Pydantic output is parsed for unknown merchants."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = """{
      "predictions": [
        {"id": 0, "category": "Shopping", "confidence": 0.88, "reasoning": "Specialty boutique store"}
      ]
    }"""
    mock_response.choices = [mock_choice]
    mock_client.chat_completion.return_value = mock_response

    items = [{"id": 0, "description": "POS/DR/FANCY_BOUTIQUE_9921", "amount": 2500.0}]
    results = categorize_unmatched_with_llm(items, client=mock_client)
    assert results[0] == "Shopping"


def test_categorize_llm_malformed_json_fallback():
    """Verify that if LLM returns invalid JSON, it retries and falls back to Uncategorized."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "INVALID NON-JSON RESPONSE"
    mock_response.choices = [mock_choice]
    mock_client.chat_completion.return_value = mock_response

    items = [{"id": 5, "description": "WEIRD_TXN_007", "amount": 100.0}]
    results = categorize_unmatched_with_llm(items, client=mock_client, max_retries=1)
    assert results[5] == StandardCategory.UNCATEGORIZED.value

