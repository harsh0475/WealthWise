"""Unit tests for statement parser and normalizer."""
import io
import pandas as pd
import pytest
from app.ingestion.parser import parse_statement, clean_amount_str, parse_date_robust


def test_clean_amount_str():
    """Verify various messy currency strings are cleaned to correct float."""
    assert clean_amount_str("1500.50") == 1500.50
    assert clean_amount_str("₹1,25,000.00") == 125000.00
    assert clean_amount_str("Rs. 450.75") == 450.75
    assert clean_amount_str("$2,500") == 2500.0
    assert clean_amount_str("(1,200.00)") == -1200.00
    assert clean_amount_str("-350.00") == -350.00
    assert clean_amount_str(None) == 0.0
    assert clean_amount_str("") == 0.0


def test_parse_date_robust():
    """Verify various date strings parse into standard ISO YYYY-MM-DD."""
    dates = pd.Series(["15/08/2024", "2024-01-05", "03-11-2024"])
    parsed = parse_date_robust(dates)
    assert parsed.tolist() == ["2024-08-15", "2024-01-05", "2024-11-03"]


def test_parse_statement_csv_standard():
    """Test standard CSV format with Debit / Credit columns."""
    csv_data = """Date,Narration,Withdrawal,Deposit,Balance
01/01/2024,Salary TechCorp,0,125000,125000
03/01/2024,House Rent UPI,32000,0,93000
10/01/2024,Swiggy Food,450.50,0,92549.50
"""
    df = parse_statement(csv_data.encode("utf-8"), file_type="csv")
    assert len(df) == 3
    assert list(df.columns) == ["date", "description", "amount", "type", "balance"]
    assert df.iloc[0]["type"] == "credit"
    assert df.iloc[0]["amount"] == 125000.0
    assert df.iloc[1]["type"] == "debit"
    assert df.iloc[1]["amount"] == 32000.0
    assert df.iloc[0]["date"] == "2024-01-01"


def test_parse_statement_messy_amounts_and_signs():
    """Test CSV with currency signs, commas, and single signed Amount column."""
    csv_data = """Txn Date,Particulars,Amount,Running Balance
2024-03-01,Bonus Credit,"₹25,000.00","₹75,000"
2024-03-05,Amazon Shopping,"-₹1,499.00","₹73,501"
2024-03-10,BESCOM Electricity,"-Rs. 2,150.50","₹71,350.50"
"""
    df = parse_statement(csv_data.encode("utf-8"), file_type="csv")
    assert len(df) == 3
    assert df.iloc[0]["type"] == "credit"
    assert df.iloc[0]["amount"] == 25000.0
    assert df.iloc[1]["type"] == "debit"
    assert df.iloc[1]["amount"] == 1499.0
    assert df.iloc[2]["type"] == "debit"
    assert df.iloc[2]["amount"] == 2150.50


def test_parse_statement_duplicate_dropping():
    """Verify exact duplicate rows are safely deduplicated."""
    csv_data = """date,description,amount,type,balance
2024-05-01,Netflix,649,debit,50000
2024-05-01,Netflix,649,debit,50000
2024-05-02,Spotify,119,debit,49881
"""
    df = parse_statement(csv_data.encode("utf-8"), file_type="csv")
    assert len(df) == 2


def test_parse_statement_xlsx():
    """Verify Excel parsing."""
    raw_df = pd.DataFrame({
        "Date": ["2024-04-01", "2024-04-02"],
        "Details": ["Grocery Blinkit", "Uber Ride"],
        "Debit": [1200, 350],
        "Credit": [0, 0],
        "Balance": [45000, 44650]
    })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        raw_df.to_excel(writer, index=False)
    output.seek(0)

    parsed = parse_statement(output.getvalue(), file_type="xlsx")
    assert len(parsed) == 2
    assert parsed.iloc[0]["amount"] == 1200.0
    assert parsed.iloc[1]["amount"] == 350.0

