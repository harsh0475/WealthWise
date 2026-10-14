"""Pure financial analytics tools executed over standardized DataFrames.

Every tool is deterministic, unit-tested, validates input arguments, and returns
a compact JSON-serializable dict to protect LLM context windows.
"""
from typing import Any, Dict, List, Optional
import re
import numpy as np
import pandas as pd


def clean_merchant_name(desc: str) -> str:
    """Extract a concise merchant identifier from raw bank narrations.
    
    Why: Bank narrations include raw reference IDs, channel codes (UPI/DR, POS),
    and terminal locations that would prevent accurate groupby aggregations.
    """
    s = str(desc).strip()

    # Handle UPI format: UPI/DR/ref/Merchant/channel
    if s.startswith("UPI/"):
        parts = s.split("/")
        if len(parts) >= 4:
            return parts[3].strip()

    # Handle POS format: POS 1234... Merchant City IN
    if s.startswith("POS"):
        # Strip POS, card placeholders, and trailing country code
        cleaned = re.sub(r"POS(?:/DR/CARD\s*[^\/]+)?\s*", "", s, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:NOIDA|MUMBAI|BANGALORE|DELHI|PUNE|IN)\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        return cleaned.strip() or s

    # Handle ACH / NEFT / BILLPAY formats
    for prefix in ["ACH DEBIT-", "NEFT CR-", "BILLPAY-", "AUTOPAY-", "BBPS-"]:
        if prefix in s:
            tail = s.split(prefix, 1)[1]
            return tail.split("-")[0].strip()

    # Strip placeholders and extra punctuation
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip()[:40]


def get_spending_summary(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """Calculate aggregated spending by category within an optional date range.
    
    Returns:
    - total_spend: float
    - total_income: float
    - net_savings: float
    - category_breakdown: list of {category, total, count, percentage}
    - transaction_count: int
    """
    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered["date"] >= start_date]
    if end_date:
        filtered = filtered[filtered["date"] <= end_date]

    if filtered.empty:
        return {
            "total_spend": 0.0,
            "total_income": 0.0,
            "net_savings": 0.0,
            "category_breakdown": [],
            "transaction_count": 0,
        }

    debits = filtered[filtered["type"] == "debit"]
    credits = filtered[filtered["type"] == "credit"]

    total_spend = round(float(debits["amount"].sum()), 2)
    total_income = round(float(credits["amount"].sum()), 2)
    net_savings = round(total_income - total_spend, 2)

    cat_group = debits.groupby("category")["amount"].agg(total="sum", count="count").reset_index()
    cat_group["total"] = cat_group["total"].round(2)
    cat_group["percentage"] = ((cat_group["total"] / (total_spend or 1.0)) * 100).round(1)
    cat_group = cat_group.sort_values(by="total", ascending=False)

    return {
        "start_date": start_date or (str(filtered["date"].min()) if not filtered.empty else None),
        "end_date": end_date or (str(filtered["date"].max()) if not filtered.empty else None),
        "total_spend": total_spend,
        "total_income": total_income,
        "net_savings": net_savings,
        "category_breakdown": cat_group.to_dict(orient="records"),
        "transaction_count": len(filtered),
    }


def get_top_merchants(
    df: pd.DataFrame,
    n: int = 5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve top N merchants by total debit spend."""
    if not isinstance(n, int) or n <= 0:
        n = 5

    debits = df[df["type"] == "debit"].copy()
    if start_date:
        debits = debits[debits["date"] >= start_date]
    if end_date:
        debits = debits[debits["date"] <= end_date]

    if debits.empty:
        return {"top_merchants": [], "count": 0}

    debits["merchant"] = debits["description"].apply(clean_merchant_name)
    m_group = debits.groupby("merchant")["amount"].agg(total_spend="sum", transaction_count="count").reset_index()
    m_group["total_spend"] = m_group["total_spend"].round(2)
    top = m_group.sort_values(by="total_spend", ascending=False).head(n)

    return {
        "top_merchants": top.to_dict(orient="records"),
        "total_merchants_analyzed": len(m_group),
    }


def get_monthly_trend(
    df: pd.DataFrame,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """Calculate monthly spending totals and month-over-month rate of change."""
    debits = df[df["type"] == "debit"].copy()
    if category:
        debits = debits[debits["category"].str.lower() == category.lower()]

    if debits.empty:
        return {"monthly_trend": [], "category": category}

    debits["month"] = debits["date"].str.slice(0, 7)
    trend = debits.groupby("month")["amount"].sum().reset_index(name="total_spend")
    trend["total_spend"] = trend["total_spend"].round(2)
    trend = trend.sort_values(by="month").reset_index(drop=True)

    # Calculate MoM growth percentage
    mom_change = [0.0]
    for i in range(1, len(trend)):
        prev = trend.loc[i - 1, "total_spend"]
        curr = trend.loc[i, "total_spend"]
        pct = round(((curr - prev) / (prev or 1.0)) * 100, 1)
        mom_change.append(pct)
    trend["mom_change_percent"] = mom_change

    return {
        "category": category or "All",
        "monthly_trend": trend.to_dict(orient="records"),
        "average_monthly_spend": round(float(trend["total_spend"].mean()), 2),
    }


def detect_recurring_payments(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect regular subscriptions, EMIs, and bills based on recurrence cadence.
    
    Why: Recurring debits have identical amounts and recur every ~25-35 days
    (monthly cadence) across 2 or more distinct occurrences.
    """
    debits = df[df["type"] == "debit"].copy()
    if len(debits) < 2:
        return {"recurring_payments": [], "total_recurring_monthly": 0.0}

    debits["date_dt"] = pd.to_datetime(debits["date"])
    debits["merchant"] = debits["description"].apply(clean_merchant_name)

    recurring = []
    # Group by merchant and rounded amount
    for (merchant, amt), group in debits.groupby(["merchant", "amount"]):
        if len(group) >= 2:
            sorted_dates = group["date_dt"].sort_values()
            deltas = sorted_dates.diff().dropna().dt.days
            # Check if median delta is roughly monthly (25 to 35 days) or bi-weekly (12 to 16 days)
            median_delta = deltas.median()
            if 25 <= median_delta <= 35 or 12 <= median_delta <= 16:
                cat = group["category"].iloc[0] if "category" in group.columns else "Subscriptions"
                recurring.append({
                    "merchant": merchant,
                    "amount": round(float(amt), 2),
                    "category": cat,
                    "frequency_days": int(round(median_delta)),
                    "occurrences": len(group),
                    "first_seen": str(sorted_dates.min().date()),
                    "last_seen": str(sorted_dates.max().date()),
                })

    recurring.sort(key=lambda x: x["amount"], reverse=True)
    total_monthly = round(sum(r["amount"] for r in recurring if r["frequency_days"] >= 25), 2)

    return {
        "recurring_payments": recurring,
        "count": len(recurring),
        "total_estimated_monthly": total_monthly,
    }


def detect_anomalies(
    df: pd.DataFrame,
    method: str = "iqr"
) -> Dict[str, Any]:
    """Detect financial anomalies using statistical IQR threshold and exact duplicate detection.
    
    Why: Bank anomalies include:
    1. Statistical outliers: Unusually large non-recurring purchases (exceeding Q3 + 1.5 * IQR)
    2. Same-day duplicate debit charges: Accidental double-billings on the same calendar day
    Note: Regular recurring fixed expenses (Rent, EMI, monthly subscriptions) are tracked
    by detect_recurring_payments and excluded here to eliminate false positive noise.
    """
    debits = df[df["type"] == "debit"].copy()
    if debits.empty:
        return {"anomalies": [], "total_flagged": 0}

    anomalies: List[Dict[str, Any]] = []

    # 1. Filter out regular recurring charges (amounts occurring >= 3 times)
    amt_counts = debits["amount"].value_counts()
    recurring_amts = set(amt_counts[amt_counts >= 3].index)
    non_recurring = debits[~debits["amount"].isin(recurring_amts)]

    # Compute IQR on non-recurring transactions
    if not non_recurring.empty:
        amounts = non_recurring["amount"].values
        q25 = float(np.percentile(amounts, 25))
        q75 = float(np.percentile(amounts, 75))
        iqr = q75 - q25
        upper_bound = q75 + (1.5 * iqr)

        outliers = non_recurring[non_recurring["amount"] > upper_bound]
        for _, row in outliers.iterrows():
            anomalies.append({
                "date": str(row["date"]),
                "description": str(row["description"]),
                "amount": round(float(row["amount"]), 2),
                "reason": f"Statistical outlier: Amount ₹{row['amount']} exceeds IQR threshold ₹{round(upper_bound, 2)}",
                "anomaly_type": "statistical_outlier",
            })
    else:
        upper_bound = 0.0

    # 2. Same-Day Duplicate Charges
    # Group by date and exact amount where count > 1
    dup_groups = debits.groupby(["date", "amount"])
    for (d, amt), grp in dup_groups:
        if len(grp) > 1:
            for _, r in grp.iterrows():
                # Avoid duplicate entries if already added as IQR outlier
                if not any(a["date"] == str(r["date"]) and a["amount"] == round(float(r["amount"]), 2) and a["anomaly_type"] == "duplicate_charge" for a in anomalies):
                    anomalies.append({
                        "date": str(r["date"]),
                        "description": str(r["description"]),
                        "amount": round(float(r["amount"]), 2),
                        "reason": f"Potential duplicate charge: {len(grp)} identical debits of ₹{amt} on {d}",
                        "anomaly_type": "duplicate_charge",
                    })

    # Sort newest or highest amount first
    anomalies.sort(key=lambda x: x["amount"], reverse=True)
    return {
        "anomalies": anomalies,
        "total_flagged": len(anomalies),
        "iqr_upper_bound": round(upper_bound, 2),
    }


def simulate_budget(
    df: pd.DataFrame,
    category: str,
    reduction_percent: float
) -> Dict[str, Any]:
    """Simulate projected monthly and annual savings from reducing spending in a category.
    
    Why: Allows users to run what-if scenarios (e.g. 'What if I cut Food & Dining by 20%?').
    """
    if reduction_percent <= 0 or reduction_percent > 100:
        return {"error": "reduction_percent must be between 1 and 100"}

    debits = df[df["type"] == "debit"].copy()
    cat_debits = debits[debits["category"].str.lower() == category.lower()]
    if cat_debits.empty:
        return {
            "category": category,
            "message": f"No spending found in category '{category}' to simulate",
            "current_monthly_spend": 0.0,
            "projected_monthly_savings": 0.0,
            "projected_annual_savings": 0.0,
        }

    # Number of months spanned
    cat_debits["month"] = cat_debits["date"].str.slice(0, 7)
    num_months = max(1, cat_debits["month"].nunique())

    total_cat_spend = float(cat_debits["amount"].sum())
    monthly_avg = round(total_cat_spend / num_months, 2)
    monthly_savings = round(monthly_avg * (reduction_percent / 100.0), 2)
    annual_savings = round(monthly_savings * 12, 2)
    new_monthly_budget = round(monthly_avg - monthly_savings, 2)

    return {
        "category": category,
        "reduction_percent": reduction_percent,
        "current_monthly_average": monthly_avg,
        "new_projected_monthly_spend": new_monthly_budget,
        "projected_monthly_savings": monthly_savings,
        "projected_annual_savings": annual_savings,
    }


def compare_periods(
    df: pd.DataFrame,
    period_a_start: str,
    period_a_end: str,
    period_b_start: str,
    period_b_end: str
) -> Dict[str, Any]:
    """Compare spending aggregates between two distinct date ranges."""
    sum_a = get_spending_summary(df, period_a_start, period_a_end)
    sum_b = get_spending_summary(df, period_b_start, period_b_end)

    spend_a = sum_a["total_spend"]
    spend_b = sum_b["total_spend"]
    diff = round(spend_b - spend_a, 2)
    pct_change = round(((spend_b - spend_a) / (spend_a or 1.0)) * 100, 1)

    return {
        "period_a": {
            "range": f"{period_a_start} to {period_a_end}",
            "total_spend": spend_a,
            "total_income": sum_a["total_income"],
            "net_savings": sum_a["net_savings"],
        },
        "period_b": {
            "range": f"{period_b_start} to {period_b_end}",
            "total_spend": spend_b,
            "total_income": sum_b["total_income"],
            "net_savings": sum_b["net_savings"],
        },
        "absolute_spend_difference": diff,
        "percentage_change": pct_change,
        "trend": "increased" if diff > 0 else ("decreased" if diff < 0 else "neutral"),
    }
