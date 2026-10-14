"""Agent guardrails for input validation, output safety, and numeric grounding.

Ensures:
1. Input safety: Detects prompt injections and validates question bounds.
2. Output privacy & safety: Blocks system prompt exfiltration and unredacted PII leaks.
3. Numeric Grounding (Amendment 4): Validates that every financial number in the agent's
   final answer was derived from a tool result.
"""
from typing import Any, List, Set, Tuple
import re
from app.llm.prompts import AGENT_SYSTEM_PROMPT
from app.safety.injection import detect_injection
from app.safety.pii import PIIVault


# Key prompt phrases that should never be exfiltrated in the final response
SYSTEM_PROMPT_FINGERPRINTS = [
    "CORE OPERATIONAL RULES",
    "PROMPT INJECTION DEFENSE",
    "PRIVACY GUARANTEE",
    "NUMERIC ACCURACY IS PARAMOUNT",
    "<<<TRANSACTION_DATA",
]


def extract_numbers_from_obj(obj: Any) -> Set[float]:
    """Recursively extract all numeric values from tool results (dicts, lists, primitives)."""
    numbers: Set[float] = set()
    if isinstance(obj, (int, float)):
        numbers.add(round(float(obj), 2))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            numbers.update(extract_numbers_from_obj(v))
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            numbers.update(extract_numbers_from_obj(item))
    elif isinstance(obj, str):
        # Extract numeric tokens from string values
        clean_s = obj.replace(",", "").replace(r"\u20b9", " ").replace("\u20b9", " ")
        clean_s = re.sub(r"[₹$€£]|Rs\.?|INR", " ", clean_s, flags=re.IGNORECASE)
        found = re.findall(r"(?<![a-zA-Z0-9_])\d+(?:\.\d+)?(?![a-zA-Z0-9_])", clean_s)
        for num_str in found:
            try:
                numbers.add(round(float(num_str), 2))
            except ValueError:
                pass
    return numbers


def extract_numbers_from_text(text: str) -> Set[float]:
    """Extract numeric values from generated answer text, ignoring dates, currency prefixes, and list markers."""
    numbers: Set[float] = set()
    clean_text = text.replace(",", "").replace(r"\u20b9", " ").replace("\u20b9", " ")
    clean_text = re.sub(r"[₹$€£]|Rs\.?|INR", " ", clean_text, flags=re.IGNORECASE)
    
    # Strip ISO dates (YYYY-MM-DD or DD/MM/YYYY) and timestamps so calendar days aren't mistaken for amounts
    clean_text = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", " ", clean_text)
    clean_text = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", " ", clean_text)
    
    # Strip numbered list markers like "1. ", "2. ", "10. "
    clean_text = re.sub(r"(?:^|\n|\s)\d+\.\s+", " ", clean_text)

    matches = re.findall(r"(?<![a-zA-Z0-9_])\d+(?:\.\d+)?(?![a-zA-Z0-9_])", clean_text)
    for m in matches:
        try:
            num = round(float(m), 2)
            # Ignore trivial structural/conversational numbers (0 to 10 for counts/lists, and common years)
            if num in {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 2023.0, 2024.0, 2025.0, 2026.0}:
                continue
            numbers.add(num)
        except ValueError:
            pass
    return numbers


def validate_input_query(query: str) -> Tuple[bool, List[str]]:
    """Validate query length and check for injection attempts."""
    warnings: List[str] = []
    if not query or len(query.strip()) < 2:
        return False, ["Query is too short."]
    if len(query) > 2000:
        return False, ["Query exceeds maximum length of 2000 characters."]

    is_injected, patterns = detect_injection(query)
    if is_injected:
        warnings.append(f"Potential prompt injection pattern detected: {', '.join(patterns)}")

    return True, warnings


def validate_numeric_grounding(
    answer: str,
    tool_results: List[Any],
    tolerance_pct: float = 0.02
) -> Tuple[bool, List[str]]:
    """Verify that all non-trivial financial numbers in the answer originated from tool results.
    
    Why: LLMs are prone to hallucinating plausible-sounding monetary numbers.
    This guardrail enforces that the agent strictly quotes factual analytics.
    """
    answer_numbers = extract_numbers_from_text(answer)
    ground_numbers: Set[float] = set()
    for tr in tool_results:
        ground_numbers.update(extract_numbers_from_obj(tr))

    warnings: List[str] = []
    ungrounded = []

    for num in answer_numbers:
        # Check if num matches any number in ground_numbers within tolerance or rounding
        matched = False
        for gn in ground_numbers:
            diff = abs(num - gn)
            if diff <= 1.0 or (gn > 0 and (diff / gn) <= tolerance_pct):
                matched = True
                break
        if not matched:
            ungrounded.append(num)

    if ungrounded:
        warnings.append(f"Ungrounded numbers detected in response: {ungrounded}")
        return False, warnings

    return True, []


def validate_output_safety(
    answer: str,
    vault: PIIVault
) -> Tuple[bool, str, List[str]]:
    """Verify output does not leak system prompts or unredacted raw PII."""
    flags: List[str] = []
    sanitized_answer = answer

    # 1. System Prompt Exfiltration check
    for fp in SYSTEM_PROMPT_FINGERPRINTS:
        if fp.lower() in answer.lower():
            flags.append("Response contained system prompt fingerprints and was blocked.")
            return False, "I am unable to display internal system instructions. How can I assist with your finances?", flags

    # 2. Raw PII Leak check
    if vault.contains_pii(answer):
        flags.append("Raw PII pattern was detected in output. Redacting before presentation.")
        sanitized_answer = vault.redact_text(answer)

    return True, sanitized_answer, flags

