"""Unit tests for all pure financial analytics tools and ToolRegistry."""
import pandas as pd
import pytest
from app.ingestion.synthetic import generate_synthetic_transactions
from app.tools import finance_tools
from app.tools.registry import create_default_tool_registry, ToolRegistry


@pytest.fixture(scope="module")
def sample_df():
    """Generate synthetic DataFrame for tool tests."""
    df = generate_synthetic_transactions(seed=42)
    # Assign deterministic categories for testing
    cats = []
    for desc in df["description"]:
        desc_l = desc.lower()
        if "salary" in desc_l:
            cats.append("Salary")
        elif "rent" in desc_l:
            cats.append("Housing & Rent")
        elif "emi" in desc_l:
            cats.append("EMI & Loans")
        elif "bescom" in desc_l or "airtel" in desc_l:
            cats.append("Utilities")
        elif "netflix" in desc_l or "spotify" in desc_l:
            cats.append("Subscriptions")
        elif "blinkit" in desc_l or "zepto" in desc_l or "dmart" in desc_l:
            cats.append("Groceries")
        elif "swiggy" in desc_l or "zomato" in desc_l:
            cats.append("Food & Dining")
        elif "uber" in desc_l or "metro" in desc_l:
            cats.append("Travel & Commute")
        elif "tanishq" in desc_l:
            cats.append("Shopping")
        else:
            cats.append("Miscellaneous")
    df["category"] = cats
    return df


def test_spending_summary(sample_df):
    """Test get_spending_summary calculates correct totals and breakdown."""
    summary = finance_tools.get_spending_summary(sample_df)
    assert summary["total_spend"] > 0
    assert summary["total_income"] == 125000.0 * 12
    assert summary["net_savings"] > 0
    assert len(summary["category_breakdown"]) > 0

    # Date range filtered
    filtered = finance_tools.get_spending_summary(sample_df, "2024-01-01", "2024-01-31")
    assert filtered["total_income"] == 125000.0
    assert filtered["transaction_count"] < len(sample_df)


def test_top_merchants(sample_df):
    """Test get_top_merchants returns top merchants."""
    res = finance_tools.get_top_merchants(sample_df, n=3)
    merchants = res["top_merchants"]
    assert len(merchants) == 3
    assert merchants[0]["total_spend"] >= merchants[1]["total_spend"]


def test_monthly_trend(sample_df):
    """Test get_monthly_trend computes monthly spend and MoM percentage."""
    res = finance_tools.get_monthly_trend(sample_df)
    trends = res["monthly_trend"]
    assert len(trends) == 12
    assert trends[0]["month"] == "2024-01"
    assert trends[0]["mom_change_percent"] == 0.0
    assert trends[1]["mom_change_percent"] is not None

    # Filtered by category
    food_res = finance_tools.get_monthly_trend(sample_df, category="Food & Dining")
    assert food_res["category"] == "Food & Dining"
    assert len(food_res["monthly_trend"]) == 12


def test_detect_recurring_payments(sample_df):
    """Test recurring payment detection finds regular subscriptions and EMIs."""
    res = finance_tools.detect_recurring_payments(sample_df)
    recurring = res["recurring_payments"]
    assert len(recurring) >= 3
    amounts = [r["amount"] for r in recurring]
    # House Rent (32,000), EMI (14,500), Netflix (649), Spotify (119)
    assert 32000.0 in amounts
    assert 14500.0 in amounts
    assert 649.0 in amounts
    assert 119.0 in amounts


def test_detect_anomalies(sample_df):
    """Test anomaly detection identifies Tanishq Jewellery and duplicate Swiggy charge."""
    res = finance_tools.detect_anomalies(sample_df, method="iqr")
    anomalies = res["anomalies"]
    assert len(anomalies) >= 2

    # Verify Tanishq 1,85,000 was flagged as statistical outlier
    tanishq = next((a for a in anomalies if a["amount"] == 185000.0), None)
    assert tanishq is not None
    assert tanishq["anomaly_type"] == "statistical_outlier"

    # Verify Swiggy 640 duplicate charge on 2024-07-14 was flagged
    swiggy_dup = next((a for a in anomalies if a["amount"] == 640.0 and a["date"] == "2024-07-14"), None)
    assert swiggy_dup is not None
    assert swiggy_dup["anomaly_type"] == "duplicate_charge"


def test_simulate_budget(sample_df):
    """Test simulate_budget computes monthly and annual savings."""
    res = finance_tools.simulate_budget(sample_df, category="Food & Dining", reduction_percent=20.0)
    assert res["reduction_percent"] == 20.0
    assert res["projected_monthly_savings"] > 0
    assert res["projected_annual_savings"] == round(res["projected_monthly_savings"] * 12, 2)

    # Invalid reduction percent
    err_res = finance_tools.simulate_budget(sample_df, category="Food & Dining", reduction_percent=150.0)
    assert "error" in err_res


def test_compare_periods(sample_df):
    """Test compare_periods returns comparison metrics between two date spans."""
    res = finance_tools.compare_periods(
        sample_df,
        period_a_start="2024-01-01",
        period_a_end="2024-03-31",
        period_b_start="2024-04-01",
        period_b_end="2024-06-30"
    )
    assert "period_a" in res
    assert "period_b" in res
    assert "absolute_spend_difference" in res
    assert res["trend"] in ["increased", "decreased", "neutral"]


def test_tool_registry_execution(sample_df):
    """Test ToolRegistry schemas and execution dispatch."""
    registry = create_default_tool_registry()
    schemas = registry.get_openai_tools()
    assert len(schemas) == 7

    # Execute spending summary through registry
    res = registry.execute("get_spending_summary", df=sample_df, start_date="2024-01-01")
    assert res["total_spend"] > 0

    # Unregistered tool raises ValueError
    with pytest.raises(ValueError, match="is not registered"):
        registry.execute("unknown_hack_tool", df=sample_df)
