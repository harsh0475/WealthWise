"""Bank statement parser for CSV and XLSX files.

Normalizes diverse bank statement formats into a standardized DataFrame with columns:
`date, description, amount, type, balance`.
Handles currency symbols, multiple date formats, separate debit/credit columns,
and duplicate transactions.
"""
from io import BytesIO
from typing import BinaryIO, Union
import re
import pandas as pd


# Column name aliases for standardizing messy bank statements
DATE_ALIASES = ["date", "txn_date", "transaction_date", "trans_date", "value_date", "posting_date"]
DESC_ALIASES = ["description", "narration", "particulars", "details", "remarks", "memo", "transaction_details"]
AMOUNT_ALIASES = ["amount", "txn_amount", "transaction_amount", "net_amount"]
DEBIT_ALIASES = ["debit", "dr", "withdrawal", "withdrawals", "debit_amount", "dr_amount"]
CREDIT_ALIASES = ["credit", "cr", "deposit", "deposits", "credit_amount", "cr_amount"]
TYPE_ALIASES = ["type", "txn_type", "transaction_type", "dr_cr"]
BALANCE_ALIASES = ["balance", "closing_balance", "running_balance", "available_balance", "net_balance"]


def clean_amount_str(val: object) -> float:
    """Extract a float amount from currency strings, e.g., '₹1,500.50', 'Rs. 250', '(100.00)'.
    
    Why: Bank statements commonly embed currency prefixes, comma separators, or accounting parentheses
    for negatives that throw ValueError in default float converters.
    """
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    
    s = str(val).strip()
    if not s or s == "-":
        return 0.0

    # Handle accounting format: (1,234.50) -> -1234.50
    is_paren_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_paren_negative = True
        s = s[1:-1].strip()

    # Strip currency signs (₹, $, €, Rs, INR) and comma formatting
    s = re.sub(r"[₹$€£\s]|INR|Rs\.?|Cr\.?|Dr\.?", "", s, flags=re.IGNORECASE)
    s = s.replace(",", "")

    try:
        val_float = float(s)
        return -val_float if is_paren_negative else val_float
    except ValueError:
        return 0.0


def _parse_single_date(val: object) -> pd.Timestamp:
    """Parse an individual date value handling ISO vs day-first formats."""
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    if not s:
        return pd.NaT
    # If starting with 4 digits (e.g. 2024-01-05), it's ISO format
    if re.match(r"^\d{4}[-/]", s):
        return pd.to_datetime(s, yearfirst=True, errors="coerce")
    # Otherwise Indian / UK standard DD/MM/YYYY or DD-MM-YYYY
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def parse_date_robust(series: pd.Series) -> pd.Series:
    """Robustly parse diverse date formats into standard YYYY-MM-DD strings.
    
    Why: Bank statements mix DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, and DD-Mon-YYYY.
    We check whether year or day appears first to avoid ambiguous swap errors.
    """
    parsed = series.apply(_parse_single_date)
    if parsed.isna().any():
        parsed = parsed.bfill().ffill()
    return parsed.dt.strftime("%Y-%m-%d")


def parse_statement(
    file_source: Union[str, BinaryIO, bytes, BytesIO],
    file_type: str = "csv"
) -> pd.DataFrame:
    """Ingest CSV or XLSX bank statement and normalize to standard schema.
    
    Columns returned:
    - date: str ('YYYY-MM-DD')
    - description: str
    - amount: float (positive magnitude)
    - type: str ('debit' or 'credit')
    - balance: float (running balance)
    """
    if isinstance(file_source, bytes):
        file_source = BytesIO(file_source)

    # 1. Load into DataFrame
    if file_type.lower() in ["xlsx", "xls"]:
        df = pd.read_excel(file_source)
    else:
        # Try reading CSV with utf-8, fallback to latin-1
        try:
            df = pd.read_csv(file_source)
        except UnicodeDecodeError:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            df = pd.read_csv(file_source, encoding="latin-1")

    if df.empty:
        return pd.DataFrame(columns=["date", "description", "amount", "type", "balance"])

    # Clean raw column headers
    df.columns = [re.sub(r"[^\w\s]", "", str(c).strip().lower()).replace(" ", "_") for c in df.columns]

    # Map column headers to standard canonical roles
    date_col = next((c for c in df.columns if c in DATE_ALIASES), None)
    desc_col = next((c for c in df.columns if c in DESC_ALIASES), None)
    amt_col = next((c for c in df.columns if c in AMOUNT_ALIASES), None)
    debit_col = next((c for c in df.columns if c in DEBIT_ALIASES), None)
    credit_col = next((c for c in df.columns if c in CREDIT_ALIASES), None)
    type_col = next((c for c in df.columns if c in TYPE_ALIASES), None)
    bal_col = next((c for c in df.columns if c in BALANCE_ALIASES), None)

    if not date_col:
        # Fallback to the first column if date-like or present
        date_col = df.columns[0]
    if not desc_col:
        # Fallback to second column or any textual column
        desc_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    normalized = pd.DataFrame()
    normalized["date"] = parse_date_robust(df[date_col])
    normalized["description"] = df[desc_col].astype(str).str.strip()

    # Determine amounts and types (debit vs credit)
    if debit_col and credit_col:
        # Two-column layout (Withdrawal vs Deposit)
        debits = df[debit_col].apply(clean_amount_str).abs()
        credits = df[credit_col].apply(clean_amount_str).abs()
        
        types = []
        amounts = []
        for d, c in zip(debits, credits):
            if c > 0 and d == 0:
                types.append("credit")
                amounts.append(c)
            else:
                types.append("debit")
                amounts.append(d if d > 0 else c)
        normalized["amount"] = amounts
        normalized["type"] = types
    elif amt_col:
        # Single amount column
        raw_amts = df[amt_col].apply(clean_amount_str)
        if type_col:
            types = []
            for t, val in zip(df[type_col].astype(str).str.lower(), raw_amts):
                if "cr" in t or "credit" in t or "deposit" in t or val > 0 and "dr" not in t:
                    types.append("credit")
                else:
                    types.append("debit")
            normalized["type"] = types
            normalized["amount"] = raw_amts.abs()
        else:
            # Infer from signs: positive is credit, negative is debit
            normalized["type"] = raw_amts.apply(lambda x: "credit" if x > 0 else "debit")
            normalized["amount"] = raw_amts.abs()
    else:
        # Fallback: look for any numeric column
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0:
            first_num = df[num_cols[0]].apply(clean_amount_str)
            normalized["amount"] = first_num.abs()
            normalized["type"] = first_num.apply(lambda x: "credit" if x > 0 else "debit")
        else:
            normalized["amount"] = 0.0
            normalized["type"] = "debit"

    # Handle running balance
    if bal_col:
        normalized["balance"] = df[bal_col].apply(clean_amount_str)
    else:
        # Reconstruct synthetic running balance starting at base 50,000
        running_bal = [50000.0]
        for _, row in normalized.iterrows():
            curr = running_bal[-1]
            delta = row["amount"] if row["type"] == "credit" else -row["amount"]
            running_bal.append(curr + delta)
        normalized["balance"] = running_bal[1:]

    # Remove zero or invalid amount rows and drop duplicate rows
    normalized = normalized[normalized["amount"] > 0]
    normalized = normalized.drop_duplicates(subset=["date", "description", "amount", "type"])
    
    # Sort chronologically by date
    normalized = normalized.sort_values(by="date", ascending=True).reset_index(drop=True)
    return normalized
