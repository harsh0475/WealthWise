"""FastAPI backend server for WealthWise.

Provides endpoints for:
- Health check (/health)
- Statement file upload (/upload)
- Synthetic statement loading (/load-synthetic)
- Dashboard spending summary (/summary)
- Conversational agent query with full execution trace (/ask)
"""
from typing import Dict, Any, Optional
import os
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.ingestion.parser import parse_statement
from app.ingestion.synthetic import generate_and_save_synthetic_statement
from app.safety.pii import PIIVault
from app.categorization.categorizer import categorize_statement
from app.agent.loop import AgentLoop
from app.tools import finance_tools
from app.schemas import AgentQueryRequest, AgentQueryResponse, UploadSummary


app = FastAPI(
    title="WealthWise API",
    description="Privacy-preserving AI personal finance agent with tool calling and measurable evals.",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory session store
class SessionState:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.vault: PIIVault = PIIVault()
        self.agent: Optional[AgentLoop] = None
        self.chat_history: list = []


sessions: Dict[str, SessionState] = {}


def get_or_create_session(session_id: str = "default") -> SessionState:
    """Retrieve existing session or initialize a new one."""
    if session_id not in sessions:
        sessions[session_id] = SessionState()
    return sessions[session_id]


def compute_session_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level summary KPIs for UI dashboard."""
    summary = finance_tools.get_spending_summary(df)
    anomalies = finance_tools.detect_anomalies(df)
    recurring = finance_tools.detect_recurring_payments(df)

    top_cats = {}
    for item in summary.get("category_breakdown", [])[:5]:
        top_cats[item["category"]] = item["total"]

    return {
        "total_transactions": len(df),
        "date_range": {
            "start": summary.get("start_date") or "",
            "end": summary.get("end_date") or "",
        },
        "total_debit": summary["total_spend"],
        "total_credit": summary["total_income"],
        "net_savings": summary["net_savings"],
        "top_categories": top_cats,
        "recurring_count": recurring.get("count", 0),
        "anomaly_count": anomalies.get("total_flagged", 0),
    }


@app.get("/health")
def health_check():
    """Health check endpoint indicating LLM provider and model configuration."""
    return {
        "status": "healthy",
        "service": "WealthWise API",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "base_url": settings.llm_base_url,
    }


@app.post("/upload", response_model=UploadSummary)
async def upload_statement(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
):
    """Upload bank statement (CSV or XLSX), redact PII, categorize, and initialize agent."""
    filename = file.filename or "statement.csv"
    ext = filename.split(".")[-1].lower()
    if ext not in ["csv", "xlsx", "xls"]:
        raise HTTPException(status_code=400, detail="Only CSV and XLSX files are supported.")

    content = await file.read()
    session = get_or_create_session(session_id)

    # 1. Parse statement
    try:
        raw_df = parse_statement(content, file_type=ext)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse statement: {str(e)}")

    if raw_df.empty:
        raise HTTPException(status_code=400, detail="Parsed statement is empty.")

    # 2. Pipeline Order: Redact PII BEFORE any categorization or LLM call
    session.vault = PIIVault()
    redacted_df = session.vault.redact_dataframe(raw_df)

    # 3. Categorize transactions (rules first, LLM fallback only sees redacted text)
    categorized_df = categorize_statement(redacted_df)

    # 4. Attach agent loop
    session.df = categorized_df
    session.agent = AgentLoop(df=categorized_df, vault=session.vault)
    session.chat_history = []

    summary_data = compute_session_summary(categorized_df)
    return UploadSummary(**summary_data)


@app.post("/load-synthetic", response_model=UploadSummary)
def load_synthetic_statement(session_id: str = "default"):
    """Load or regenerate 12 months of synthetic Indian statement data."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_statement.csv")
    if os.path.exists(csv_path):
        raw_df = pd.read_csv(csv_path)
    else:
        raw_df = generate_and_save_synthetic_statement(csv_path)

    session = get_or_create_session(session_id)

    # Pipeline: Redact PII first
    session.vault = PIIVault()
    redacted_df = session.vault.redact_dataframe(raw_df, account_holder_name="Aarav Sharma")

    # Categorize
    categorized_df = categorize_statement(redacted_df)

    session.df = categorized_df
    session.agent = AgentLoop(df=categorized_df, vault=session.vault)
    session.chat_history = []

    summary_data = compute_session_summary(categorized_df)
    return UploadSummary(**summary_data)


@app.get("/summary")
def get_statement_summary(session_id: str = "default"):
    """Retrieve current summary metrics and analytics for the active statement."""
    session = get_or_create_session(session_id)
    if session.df is None or session.df.empty:
        raise HTTPException(status_code=404, detail="No active statement found. Please upload or load synthetic data.")
    return compute_session_summary(session.df)


@app.post("/ask", response_model=AgentQueryResponse)
def ask_agent(req: AgentQueryRequest):
    """Ask question to the personal finance agent and return grounded answer with trace."""
    session = get_or_create_session(req.session_id or "default")
    if session.df is None or session.agent is None:
        raise HTTPException(
            status_code=400,
            detail="No statement has been loaded yet. Please upload a statement or load synthetic data first.",
        )

    response = session.agent.run(req.question, chat_history=session.chat_history)

    # Update conversation history
    session.chat_history.append({"role": "user", "content": req.question})
    session.chat_history.append({"role": "assistant", "content": response.answer})

    return response

