"""Unit tests for PII detection, redaction, and restoration vault."""
import pandas as pd
import pytest
from app.safety.pii import PIIVault


def test_redact_account_numbers():
    """Verify bank account numbers (both prefixed and raw) are redacted."""
    vault = PIIVault()
    text = "NEFT transfer from A/C 50100492817291 to 987654321098."
    redacted = vault.redact_text(text)
    assert "50100492817291" not in redacted
    assert "987654321098" not in redacted
    assert "<ACCOUNT_1>" in redacted
    assert "<ACCOUNT_2>" in redacted


def test_redact_cards_masked_and_full():
    """Verify 16-digit cards and masked cards are redacted."""
    vault = PIIVault()
    text = "POS 4111-2222-3333-4444 and POS 5241XXXXXXXX1234"
    redacted = vault.redact_text(text)
    assert "4111-2222-3333-4444" not in redacted
    assert "5241XXXXXXXX1234" not in redacted
    assert "<CARD_1>" in redacted
    assert "<CARD_2>" in redacted


def test_redact_phones():
    """Verify Indian phone numbers with and without +91 are redacted."""
    vault = PIIVault()
    text = "Call mobile +91 9876543210 or 8765432109 for OTP."
    redacted = vault.redact_text(text)
    assert "9876543210" not in redacted
    assert "8765432109" not in redacted
    assert "<PHONE_1>" in redacted
    assert "<PHONE_2>" in redacted


def test_redact_emails_and_upi():
    """Verify email addresses and Indian UPI IDs are redacted."""
    vault = PIIVault()
    text = "Send confirmation to user.name@example.com or UPI payment to rohit.sharma@okhdfcbank"
    redacted = vault.redact_text(text)
    assert "user.name@example.com" not in redacted
    assert "rohit.sharma@okhdfcbank" not in redacted
    assert "<EMAIL_1>" in redacted
    assert "<UPI_1>" in redacted


def test_redact_pan_and_aadhaar():
    """Verify PAN and Aadhaar formatted numbers are redacted."""
    vault = PIIVault()
    text = "Tax PAN: ABCDE1234F, Aadhaar Ref: 4321 8765 1234"
    redacted = vault.redact_text(text)
    assert "ABCDE1234F" not in redacted
    assert "4321 8765 1234" not in redacted
    assert "<PAN_1>" in redacted
    assert "<AADHAAR_1>" in redacted


def test_redact_account_holder_name():
    """Verify account holder's name is redacted."""
    vault = PIIVault()
    text = "Salary credit for Aarav Sharma from TechCorp"
    redacted = vault.redact_text(text, account_holder_name="Aarav Sharma")
    assert "Aarav Sharma" not in redacted
    assert "<NAME_1>" in redacted


def test_stable_placeholder_reuse():
    """Verify repeated instances of the same PII reuse the same placeholder index."""
    vault = PIIVault()
    text = "A/C 50100492817291 debited. Also verify A/C 50100492817291."
    redacted = vault.redact_text(text)
    assert "<ACCOUNT_1>" in redacted
    assert "<ACCOUNT_2>" not in redacted
    assert redacted.count("<ACCOUNT_1>") == 2


def test_vault_restoration():
    """Verify vault can accurately restore redacted placeholders back to original text."""
    vault = PIIVault()
    original = "Salary for Aarav Sharma, A/C 50100492817291, UPI test@okhdfcbank"
    redacted = vault.redact_text(original, account_holder_name="Aarav Sharma")
    restored = vault.restore_text(redacted)
    assert restored == original


def test_redact_dataframe():
    """Verify DataFrame descriptions are sanitized in batch."""
    vault = PIIVault()
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "description": [
            "UPI/DR/30291/aarav.sharma@okhdfcbank",
            "BILLPAY/BESCOM/Mobile +91 9876543210"
        ],
        "amount": [1500.0, 2100.0],
    })
    clean_df = vault.redact_dataframe(df, account_holder_name="Aarav Sharma")
    assert "aarav.sharma@okhdfcbank" not in clean_df.iloc[0]["description"]
    assert "+91 9876543210" not in clean_df.iloc[1]["description"]
    assert "<UPI_1>" in clean_df.iloc[0]["description"]
    assert "<PHONE_1>" in clean_df.iloc[1]["description"]

