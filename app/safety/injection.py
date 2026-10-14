"""Prompt injection detection, input sanitization, and data block containment.

Protects against untrusted transaction descriptions and adversarial user queries:
- Heuristic detection of jailbreak / system instruction overrides
- Strict data block encapsulation (`<<<TRANSACTION_DATA ... >>>`)
- Control character and delimiter sanitization
"""
from typing import List, Tuple
import re


INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(?:the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"system\s+prompt\s*:", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:in\s+)?(?:dan|evil|unrestricted|god\s+mode)", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?prior\s+rules", re.IGNORECASE),
    re.compile(r"bypass\s+(?:all\s+)?guardrails", re.IGNORECASE),
    re.compile(r"act\s+as\s+an\s+unfiltered", re.IGNORECASE),
    re.compile(r"exfiltrate\s+(?:data|prompt|keys)", re.IGNORECASE),
    re.compile(r"print\s+your\s+initial\s+instructions", re.IGNORECASE),
]


def detect_injection(text: str) -> Tuple[bool, List[str]]:
    """Inspect text for heuristic prompt-injection signatures.
    
    Returns:
    - (is_injected: bool, matched_patterns: list[str])
    """
    if not text:
        return False, []

    matched = []
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            matched.append(match.group(0))

    return len(matched) > 0, matched


def sanitize_untrusted_text(text: str) -> str:
    """Sanitize untrusted text before LLM consumption.
    
    Strips zero-width spaces, harmful non-printable control characters,
    and escapes raw delimiter tokens that could break data framing.
    """
    if not text:
        return ""

    # Remove zero-width spaces and control characters (except standard newlines/tabs)
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\u200B-\u200D\uFEFF]", "", text)

    # Neutralize any attempts to close data delimiter blocks prematurely
    cleaned = cleaned.replace(">>>", "&gt;&gt;&gt;").replace("<<<", "&lt;&lt;&lt;")
    return cleaned.strip()


def wrap_transaction_data(text: str) -> str:
    """Encapsulate untrusted transaction text in explicit containment delimiters.
    
    Why: Instructs LLM attention that contents are static data records,
    not instructional control directives.
    """
    sanitized = sanitize_untrusted_text(text)
    return f"<<<TRANSACTION_DATA\n{sanitized}\n>>>"

