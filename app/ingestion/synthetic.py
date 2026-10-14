"""Synthetic bank statement generator for WealthWise.

Generates 12 months of realistic Indian-context financial transactions:
- Monthly Salary credit
- Rent via UPI
- Groceries (Blinkit, Zepto, D-Mart)
- Food Delivery (Swiggy, Zomato)
- Utilities (BESCOM, Airtel Broadband, Tata Play)
- Subscriptions (Netflix, Spotify, Amazon Prime)
- Loan EMI (HDFC Personal Loan)
- Commute/Travel (Uber, Metro, MakeMyTrip)
- Deliberate anomalies:
  * 1 unusually large purchase (Tanishq Jewellery)
  * 1 duplicate charge (Swiggy double debit on same day)
  * 1 unknown merchant (PAYTM*UNKNOWN_TXN_8841)
- Realistic embedded Indian PII tokens for safety verification.
"""
import os
import random
from datetime import date, timedelta
from typing import List, Dict, Any
import pandas as pd


def generate_synthetic_transactions(
    start_date: date = date(2024, 1, 1),
    months: int = 12,
    seed: int = 42
) -> pd.DataFrame:
    """Generate deterministic 12-month synthetic transaction history."""
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    running_balance = 85000.0

    # Account holder personal details embedded in transaction narrations
    account_holder = "Aarav Sharma"
    account_no = "50100492817291"
    upi_id = "aarav.sharma@okhdfcbank"
    phone_no = "+91 9876543210"
    pan_no = "ABCDE1234F"

    current_month_start = start_date

    for month_idx in range(months):
        year = current_month_start.year + (current_month_start.month + month_idx - 1) // 12
        month = (current_month_start.month + month_idx - 1) % 12 + 1

        # 1. Monthly Salary (1st of month)
        sal_date = date(year, month, 1)
        sal_amount = 125000.00
        running_balance += sal_amount
        records.append({
            "date": sal_date.strftime("%Y-%m-%d"),
            "description": f"NEFT CR-HDFC0000128-TechCorp India Pvt Ltd Salary-{account_holder}-{account_no}",
            "amount": sal_amount,
            "type": "credit",
            "balance": round(running_balance, 2),
        })

        # 2. House Rent via UPI (3rd of month)
        rent_date = date(year, month, 3)
        rent_amount = 32000.00
        running_balance -= rent_amount
        records.append({
            "date": rent_date.strftime("%Y-%m-%d"),
            "description": f"UPI/DR/4039281726/Rent transfer to Landlord/{upi_id}/HDFC",
            "amount": rent_amount,
            "type": "debit",
            "balance": round(running_balance, 2),
        })

        # 3. Monthly EMI: HDFC Personal Loan (5th of month)
        emi_date = date(year, month, 5)
        emi_amount = 14500.00
        running_balance -= emi_amount
        records.append({
            "date": emi_date.strftime("%Y-%m-%d"),
            "description": f"ACH DEBIT-HDFC BANK PERSONAL LOAN EMI-A/C {account_no}",
            "amount": emi_amount,
            "type": "debit",
            "balance": round(running_balance, 2),
        })

        # 4. Utilities (10th of month)
        # BESCOM Electricity
        bescom_amt = round(rng.uniform(1800, 2400), 2)
        running_balance -= bescom_amt
        records.append({
            "date": date(year, month, 10).strftime("%Y-%m-%d"),
            "description": f"BILLPAY-BESCOM Electricity Bangalore Consumer ID 9827361-Mobile {phone_no}",
            "amount": bescom_amt,
            "type": "debit",
            "balance": round(running_balance, 2),
        })

        # Airtel Broadband
        airtel_amt = 1179.00
        running_balance -= airtel_amt
        records.append({
            "date": date(year, month, 11).strftime("%Y-%m-%d"),
            "description": f"BBPS-Airtel Fiber Broadband 08041239876-Receipt 98213",
            "amount": airtel_amt,
            "type": "debit",
            "balance": round(running_balance, 2),
        })

        # 5. Monthly Subscriptions (15th of month)
        netflix_amt = 649.00
        running_balance -= netflix_amt
        records.append({
            "date": date(year, month, 15).strftime("%Y-%m-%d"),
            "description": "POS 4111222233334444 NETFLIX ENTERTAINMENT MUMBAI IN",
            "amount": netflix_amt,
            "type": "debit",
            "balance": round(running_balance, 2),
        })

        spotify_amt = 119.00
        running_balance -= spotify_amt
        records.append({
            "date": date(year, month, 15).strftime("%Y-%m-%d"),
            "description": "AUTOPAY-Spotify India Premium Monthly Subscription",
            "amount": spotify_amt,
            "type": "debit",
            "balance": round(running_balance, 2),
        })

        # 6. Groceries & Quick Commerce (distributed throughout the month)
        for day in [4, 12, 19, 26]:
            groc_store = rng.choice(["Blinkit Quick Commerce Bangalore", "Zepto Marketplace", "DMart Supermarket Koramangala"])
            groc_amt = round(rng.uniform(650, 2200), 2)
            running_balance -= groc_amt
            records.append({
                "date": date(year, month, day).strftime("%Y-%m-%d"),
                "description": f"UPI/DR/398172938/{groc_store}/UPI-REF-8921",
                "amount": groc_amt,
                "type": "debit",
                "balance": round(running_balance, 2),
            })

        # 7. Food Delivery & Dining
        for day in [6, 14, 21, 28]:
            food_vendor = rng.choice(["Swiggy Food Bangalore", "Zomato Restaurant Order", "Third Wave Coffee Roasters"])
            food_amt = round(rng.uniform(320, 950), 2)
            running_balance -= food_amt
            records.append({
                "date": date(year, month, day).strftime("%Y-%m-%d"),
                "description": f"UPI/DR/928172819/{food_vendor}/Bangalore",
                "amount": food_amt,
                "type": "debit",
                "balance": round(running_balance, 2),
            })

        # 8. Commute (Uber / Metro)
        for day in [8, 17, 24]:
            commute_vendor = rng.choice(["Uber India Systems Pvt Ltd", "BMRCL Namma Metro Smart Card Reload"])
            commute_amt = round(rng.uniform(180, 550), 2)
            running_balance -= commute_amt
            records.append({
                "date": date(year, month, day).strftime("%Y-%m-%d"),
                "description": f"POS/DR/CARD 5241XXXXXXXX1234/{commute_vendor}",
                "amount": commute_amt,
                "type": "debit",
                "balance": round(running_balance, 2),
            })

        # 9. Quarterly Investment (SIP in Mutual Fund in Mar, Jun, Sep, Dec)
        if month in [3, 6, 9, 12]:
            sip_amt = 15000.00
            running_balance -= sip_amt
            records.append({
                "date": date(year, month, 18).strftime("%Y-%m-%d"),
                "description": f"ACH DEBIT-Zerodha Broking Limited Coin SIP PAN-{pan_no}",
                "amount": sip_amt,
                "type": "debit",
                "balance": round(running_balance, 2),
            })

    # --- Deliberate Anomalies Injected ---
    
    # Anomaly 1: Unusually large purchase in October (Festive season / Dhanteras)
    tanishq_amt = 185000.00
    running_balance -= tanishq_amt
    records.append({
        "date": "2024-10-24",
        "description": f"POS/DR/CARD 5241XXXXXXXX1234/Tanishq Jewellers Indiranagar Bangalore/Aadhaar Ref 4321 8765 1234",
        "amount": tanishq_amt,
        "type": "debit",
        "balance": round(running_balance, 2),
    })

    # Anomaly 2: Accidental duplicate transaction on same day (July 14)
    # Swiggy debited twice with exact same amount 640.00
    dup_amt = 640.00
    running_balance -= dup_amt
    records.append({
        "date": "2024-07-14",
        "description": "UPI/DR/8812938192/Swiggy Food Bangalore/Duplicate Ref A",
        "amount": dup_amt,
        "type": "debit",
        "balance": round(running_balance, 2),
    })
    running_balance -= dup_amt
    records.append({
        "date": "2024-07-14",
        "description": "UPI/DR/8812938193/Swiggy Food Bangalore/Duplicate Ref B",
        "amount": dup_amt,
        "type": "debit",
        "balance": round(running_balance, 2),
    })

    # Anomaly 3: New unknown/uncommon merchant in November
    unknown_amt = 3200.00
    running_balance -= unknown_amt
    records.append({
        "date": "2024-11-09",
        "description": "POS/DR/PAYTM*UNKNOWN_TXN_8841 NOIDA UP IN",
        "amount": unknown_amt,
        "type": "debit",
        "balance": round(running_balance, 2),
    })

    df = pd.DataFrame(records)
    # Sort chronologically by date
    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
    return df


def generate_and_save_synthetic_statement(output_path: str = "data/synthetic_statement.csv") -> pd.DataFrame:
    """Generate synthetic statement DataFrame and persist to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_synthetic_transactions()
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic_statement.csv")
    df_gen = generate_and_save_synthetic_statement(os.path.abspath(out_file))
    print(f"Generated {len(df_gen)} transactions saved to {out_file}")

