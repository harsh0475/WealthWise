"""Personally Identifiable Information (PII) detection, redaction, and vaulting.

This module guarantees that no raw customer PII ever reaches the LLM.
Supported PII types:
- Bank Account Numbers (9-18 digits)
- Debit/Credit Card Numbers (16-digit or masked 1234XXXXXXXX5678)
- Indian Mobile Numbers (+91 or 10-digit starting with 6-9)
- Email Addresses
- UPI IDs (e.g. user@okhdfcbank, user@paytm)
- Indian PAN (e.g. ABCDE1234F)
- Aadhaar-like 12-digit UID numbers
- Account Holder Name(s)

Redacted entities are replaced by deterministic, sequential placeholders:
`<ACCOUNT_1>`, `<PHONE_1>`, `<UPI_1>`, `<NAME_1>`.
A local bidirectional PIIVault allows restoration strictly in user-facing UI contexts.
"""
from typing import Dict, List, Optional, Tuple
import re
import pandas as pd


# Regex Patterns for Financial & Indian PII
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_UPI = re.compile(
    r"\b[a-zA-Z0-9.\-_]{2,50}@(okhdfcbank|okaxis|oksbi|paytm|ybl|apl|upi|axl|ibl|icici|kotak|postbank|sbi|hdfcbank)\b",
    re.IGNORECASE
)
RE_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
RE_CARD = re.compile(
    r"\b(?:\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|\d{4}[Xx\*\-\s]{4,10}\d{4})\b"
)
RE_AADHAAR = re.compile(r"\b[2-9]\d{3}[\s-]\d{4}[\s-]\d{4}\b")
RE_PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
RE_ACCOUNT_PREFIXED = re.compile(
    r"(?:A/C|A/c|Account|acct)[^\w\d]*([0-9]{9,18})\b",
    re.IGNORECASE
)
RE_ACCOUNT_RAW = re.compile(r"(?<!\d)[0-9]{11,18}(?!\d)")


class PIIVault:
    """Manages bidirectional PII redaction and placeholder restoration."""

    def __init__(self):
        # placeholder -> original_value (e.g., '<ACCOUNT_1>' -> '50100492817291')
        self._vault: Dict[str, str] = {}
        # original_value -> placeholder (e.g., '50100492817291' -> '<ACCOUNT_1>')
        self._reverse_vault: Dict[str, str] = {}
        # Counter for sequential stable placeholders
        self._counters: Dict[str, int] = {
            "ACCOUNT": 1,
            "CARD": 1,
            "PHONE": 1,
            "EMAIL": 1,
            "UPI": 1,
            "PAN": 1,
            "AADHAAR": 1,
            "NAME": 1,
        }

    def _get_or_create_placeholder(self, pii_type: str, original_val: str) -> str:
        """Return existing placeholder for a value or create a new sequential one."""
        cleaned_key = original_val.strip()
        if cleaned_key in self._reverse_vault:
            return self._reverse_vault[cleaned_key]

        idx = self._counters.get(pii_type, 1)
        placeholder = f"<{pii_type}_{idx}>"
        self._counters[pii_type] = idx + 1

        self._vault[placeholder] = cleaned_key
        self._reverse_vault[cleaned_key] = placeholder
        return placeholder

    def redact_text(
        self,
        text: str,
        account_holder_name: Optional[str] = None,
        additional_names: Optional[List[str]] = None,
    ) -> str:
        """Redact all detected PII patterns in string and substitute placeholders.
        
        Order of substitution is deliberate:
        1. Names (specific multi-word strings first)
        2. Prefixed account numbers
        3. Email addresses
        4. UPI IDs
        5. PAN numbers
        6. Credit/Debit card numbers
        7. Aadhaar numbers
        8. Phone numbers
        9. Standalone account numbers
        """
        if not text:
            return text

        redacted = str(text)

        # 1. Names
        names_to_redact = []
        if account_holder_name and len(account_holder_name.strip()) > 2:
            names_to_redact.append(account_holder_name.strip())
        if additional_names:
            for n in additional_names:
                if n and len(n.strip()) > 2:
                    names_to_redact.append(n.strip())

        for name in names_to_redact:
            if re.search(re.escape(name), redacted, flags=re.IGNORECASE):
                ph = self._get_or_create_placeholder("NAME", name)
                redacted = re.sub(re.escape(name), ph, redacted, flags=re.IGNORECASE)

        # 2. Prefixed Account Numbers
        def replace_prefixed_account(match: re.Match) -> str:
            raw_acc = match.group(1)
            ph = self._get_or_create_placeholder("ACCOUNT", raw_acc)
            return match.group(0).replace(raw_acc, ph)

        redacted = RE_ACCOUNT_PREFIXED.sub(replace_prefixed_account, redacted)

        # 3. Email Addresses
        def replace_email(match: re.Match) -> str:
            return self._get_or_create_placeholder("EMAIL", match.group(0))

        redacted = RE_EMAIL.sub(replace_email, redacted)

        # 4. UPI IDs
        def replace_upi(match: re.Match) -> str:
            return self._get_or_create_placeholder("UPI", match.group(0))

        redacted = RE_UPI.sub(replace_upi, redacted)

        # 5. Indian PAN
        def replace_pan(match: re.Match) -> str:
            return self._get_or_create_placeholder("PAN", match.group(0))

        redacted = RE_PAN.sub(replace_pan, redacted)

        # 6. Card Numbers
        def replace_card(match: re.Match) -> str:
            return self._get_or_create_placeholder("CARD", match.group(0))

        redacted = RE_CARD.sub(replace_card, redacted)

        # 7. Aadhaar Numbers
        def replace_aadhaar(match: re.Match) -> str:
            return self._get_or_create_placeholder("AADHAAR", match.group(0))

        redacted = RE_AADHAAR.sub(replace_aadhaar, redacted)

        # 8. Phone Numbers
        def replace_phone(match: re.Match) -> str:
            return self._get_or_create_placeholder("PHONE", match.group(0))

        redacted = RE_PHONE.sub(replace_phone, redacted)

        # 9. Standalone 11-18 digit account numbers (if not already part of placeholder)
        def replace_raw_account(match: re.Match) -> str:
            raw_num = match.group(0)
            return self._get_or_create_placeholder("ACCOUNT", raw_num)

        redacted = RE_ACCOUNT_RAW.sub(replace_raw_account, redacted)

        return redacted

    def restore_text(self, text: str) -> str:
        """Restore placeholders in text back to original values for UI presentation only.
        
        Why: LLM answers can be made readable to the authenticated user in the frontend,
        while ensuring the LLM never received the true PII.
        """
        if not text:
            return text
        restored = str(text)
        for placeholder, original in self._vault.items():
            restored = restored.replace(placeholder, original)
        return restored

    def redact_dataframe(
        self,
        df: pd.DataFrame,
        account_holder_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Redact all text columns in a DataFrame, producing a sanitized copy."""
        clean_df = df.copy()
        if "description" in clean_df.columns:
            clean_df["description"] = clean_df["description"].apply(
                lambda s: self.redact_text(s, account_holder_name=account_holder_name)
            )
        return clean_df

    def contains_pii(self, text: str) -> bool:
        """Check whether a text string contains unredacted raw PII patterns."""
        if not text:
            return False
        # Test all patterns
        if RE_EMAIL.search(text):
            return True
        if RE_UPI.search(text):
            return True
        if RE_PAN.search(text):
            return True
        if RE_CARD.search(text):
            return True
        if RE_AADHAAR.search(text):
            return True
        if RE_PHONE.search(text):
            return True
        if RE_ACCOUNT_PREFIXED.search(text):
            return True
        if RE_ACCOUNT_RAW.search(text):
            # Check if this raw number is not part of a placeholder like <ACCOUNT_1>
            matches = RE_ACCOUNT_RAW.findall(text)
            for m in matches:
                if f"<{m}>" not in text:
                    return True
        return False

    @property
    def vault_mappings(self) -> Dict[str, str]:
        """Return read-only copy of placeholder mappings."""
        return dict(self._vault)
