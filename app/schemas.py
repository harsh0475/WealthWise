"""Pydantic schemas and data models for WealthWise.
"""
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    """Transaction direction."""
    DEBIT = "debit"
    CREDIT = "credit"


class StandardCategory(str, Enum):
    """Standardized financial category classifications."""
    SALARY = "Salary"
    GROCERIES = "Groceries"
    FOOD_AND_DINING = "Food & Dining"
    HOUSING_AND_RENT = "Housing & Rent"
    UTILITIES = "Utilities"
    SUBSCRIPTIONS = "Subscriptions"
    EMI_AND_LOANS = "EMI & Loans"
    SHOPPING = "Shopping"
    TRAVEL_AND_COMMUTE = "Travel & Commute"
    HEALTHCARE = "Healthcare"
    INVESTMENTS = "Investments"
    TRANSFERS = "Transfers"
    MISCELLANEOUS = "Miscellaneous"
    UNCATEGORIZED = "Uncategorized"


class Transaction(BaseModel):
    """Normalized bank transaction record."""
    date: str = Field(description="Transaction date in YYYY-MM-DD format")
    description: str = Field(description="Redacted, sanitized transaction description")
    amount: float = Field(gt=0, description="Absolute transaction amount")
    type: TransactionType = Field(description="Debit or credit")
    balance: float = Field(description="Account balance after transaction")
    category: Optional[str] = Field(default="Uncategorized", description="Assigned category")
    is_anomaly: bool = Field(default=False, description="Flagged by anomaly detection")
    is_recurring: bool = Field(default=False, description="Flagged as recurring payment/subscription/EMI")


# --- Categorization Models ---

class CategorizedItem(BaseModel):
    """Single transaction category prediction from LLM."""
    id: int = Field(description="Transaction identifier corresponding to input batch")
    category: str = Field(description="Predicted standard category")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: Optional[str] = Field(default=None, description="Short explanation for category selection")


class BatchCategorizationOutput(BaseModel):
    """Structured output expected from LLM for a batch of transactions."""
    predictions: List[CategorizedItem] = Field(default_factory=list)


# --- Tool & Agent Execution Models ---

class ToolCallRecord(BaseModel):
    """Detailed record of a single tool execution within the agent loop."""
    iteration: int
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    latency_ms: float
    status: Literal["success", "error"]
    error_message: Optional[str] = None


class AgentTrace(BaseModel):
    """Execution trace returned with every answer for observability in UI and evals."""
    steps: List[ToolCallRecord] = Field(default_factory=list)
    total_iterations: int = 0
    total_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    security_flags: List[str] = Field(default_factory=list)
    grounding_warnings: List[str] = Field(default_factory=list)


class AgentQueryRequest(BaseModel):
    """User query payload sent to the agent."""
    question: str
    session_id: Optional[str] = "default"


class AgentQueryResponse(BaseModel):
    """Complete response returned by the agent."""
    question: str
    answer: str
    trace: AgentTrace
    grounded: bool = True


# --- Ingestion & Summary Models ---

class UploadSummary(BaseModel):
    """Summary statistics returned after parsing and categorizing a statement."""
    total_transactions: int
    date_range: Dict[str, str]
    total_debit: float
    total_credit: float
    net_savings: float
    top_categories: Dict[str, float]
    recurring_count: int
    anomaly_count: int

