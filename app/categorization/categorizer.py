"""Two-stage transaction categorization engine.

Stage 1: Deterministic keyword and regex rules for high-precision, zero-latency matching.
Stage 2: LLM fallback for ambiguous merchants with Pydantic JSON validation and retries.
Respects context window budgeting via batching.
"""
from typing import Dict, List, Optional
import json
import re
import pandas as pd
from app.llm.client import LLMClient, default_llm_client
from app.llm.prompts import CATEGORIZATION_SYSTEM_PROMPT
from app.schemas import BatchCategorizationOutput, StandardCategory


# Deterministic rule definitions: compiled regex patterns mapped to standard categories
RULE_PATTERNS: List[tuple[re.Pattern, str]] = [
    # Salary
    (re.compile(r"\b(salary|payroll|stipend|bonus credit|techcorp)\b", re.IGNORECASE), StandardCategory.SALARY.value),
    # Housing & Rent
    (re.compile(r"\b(rent|landlord|flat rent|maintenance charge)\b", re.IGNORECASE), StandardCategory.HOUSING_AND_RENT.value),
    # EMI & Loans
    (re.compile(r"\b(loan emi|personal loan|home loan|ach debit.*loan|emi-a/c)\b", re.IGNORECASE), StandardCategory.EMI_AND_LOANS.value),
    # Utilities
    (re.compile(r"\b(bescom|airtel|jio|tata play|broadband|electricity|billpay|bbps|water supply|gas)\b", re.IGNORECASE), StandardCategory.UTILITIES.value),
    # Subscriptions
    (re.compile(r"\b(netflix|spotify|amazon prime|hotstar|youtube premium|disney|apple\.com/bill)\b", re.IGNORECASE), StandardCategory.SUBSCRIPTIONS.value),
    # Groceries
    (re.compile(r"\b(blinkit|zepto|dmart|d-mart|bigbasket|instamart|supermarket|groceries|nature's basket)\b", re.IGNORECASE), StandardCategory.GROCERIES.value),
    # Food & Dining
    (re.compile(r"\b(swiggy|zomato|starbucks|third wave|mcdonald|restaurant|cafe|burger king|kfc|pizza)\b", re.IGNORECASE), StandardCategory.FOOD_AND_DINING.value),
    # Commute & Travel
    (re.compile(r"\b(uber|ola|metro|bmrcl|irctc|makemytrip|indigo|petrol|fuel|indian oil|bpcl|hpcl|fastag)\b", re.IGNORECASE), StandardCategory.TRAVEL_AND_COMMUTE.value),
    # Shopping
    (re.compile(r"\b(tanishq|jewellers|amazon\.in|flipkart|myntra|zara|h&m|clothing|apparel)\b", re.IGNORECASE), StandardCategory.SHOPPING.value),
    # Healthcare
    (re.compile(r"\b(apollo|pharmacy|pharmeasy|1mg|medplus|hospital|clinic|diagnostic|doctor)\b", re.IGNORECASE), StandardCategory.HEALTHCARE.value),
    # Investments
    (re.compile(r"\b(zerodha|groww|coin sip|mutual fund|sip pan|nps|upstox|uti mf|hdfc mf)\b", re.IGNORECASE), StandardCategory.INVESTMENTS.value),
]


def match_rules(description: str) -> Optional[str]:
    """Test description against deterministic merchant rules."""
    for pattern, category in RULE_PATTERNS:
        if pattern.search(description):
            return category
    return None


def categorize_unmatched_with_llm(
    items: List[Dict[str, object]],
    client: LLMClient,
    max_retries: int = 2
) -> Dict[int, str]:
    """Batch-categorize ambiguous transactions using LLM structured output.
    
    Why: Keeps LLM calls minimal and validates strictly with Pydantic,
    re-prompting on JSON/Schema failure before falling back to Uncategorized.
    """
    if not items:
        return {}

    user_content = f"Categorize this batch of transactions:\n{json.dumps(items, indent=2)}"
    messages = [
        {"role": "system", "content": CATEGORIZATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    predictions_map: Dict[int, str] = {}

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw_content = resp.choices[0].message.content or "{}"
            
            # Clean possible markdown fence ```json ... ```
            clean_json = raw_content.strip()
            if clean_json.startswith("```"):
                clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
                clean_json = re.sub(r"\s*```$", "", clean_json)
            
            parsed_data = json.loads(clean_json)
            # Validate with Pydantic
            validated = BatchCategorizationOutput.model_validate(parsed_data)
            
            for pred in validated.predictions:
                predictions_map[pred.id] = pred.category
            return predictions_map
        except Exception:
            if attempt == max_retries:
                # Fallback to Uncategorized on persistent failure
                for itm in items:
                    predictions_map[int(itm["id"])] = StandardCategory.UNCATEGORIZED.value
                return predictions_map

    return predictions_map


def categorize_statement(
    df: pd.DataFrame,
    client: Optional[LLMClient] = None,
    batch_size: int = 25
) -> pd.DataFrame:
    """Assign categories to a normalized, redacted statement DataFrame.
    
    1. First passes each transaction through deterministic rules (fast, free, 100% reliable).
    2. Collects unmatched transactions and queries LLM in budgeted chunks.
    """
    result_df = df.copy()
    categories: List[str] = []
    unmatched_indices: List[int] = []
    unmatched_payload: List[Dict[str, object]] = []

    # Stage 1: Rules pass
    for idx, row in result_df.iterrows():
        cat = match_rules(str(row["description"]))
        if cat:
            categories.append(cat)
        else:
            categories.append(StandardCategory.UNCATEGORIZED.value)
            unmatched_indices.append(idx)
            unmatched_payload.append({
                "id": idx,
                "description": str(row["description"]),
                "amount": float(row["amount"]),
            })

    # Stage 2: LLM pass for unmatched (if client provided and unmatched exist)
    if unmatched_payload and client is not None:
        for i in range(0, len(unmatched_payload), batch_size):
            chunk = unmatched_payload[i : i + batch_size]
            llm_results = categorize_unmatched_with_llm(chunk, client=client)
            for idx_val, cat_val in llm_results.items():
                categories[idx_val] = cat_val

    result_df["category"] = categories
    return result_df

