"""Tool registry for registering, discovering, and executing agent tools.

Converts Python functions into OpenAI-compatible tool schemas and handles
safe argument validation before dispatching execution to DataFrame tools.
"""
from typing import Any, Callable, Dict, List, Optional
import inspect
import pandas as pd
from app.tools import finance_tools


class ToolDefinition:
    """Metadata and execution pointer for a registered tool."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., Any],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert tool definition to OpenAI function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry maintaining available tools for the agent loop."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., Any],
    ) -> None:
        """Register a new tool."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Look up tool by name."""
        return self._tools.get(name)

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Return list of all tool schemas formatted for OpenAI chat completions."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute(self, name: str, df: pd.DataFrame, **kwargs) -> Any:
        """Validate tool existence and safely execute against provided DataFrame."""
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' is not registered. Allowed tools: {list(self._tools.keys())}")

        sig = inspect.signature(tool.func)
        bound_args = {}
        for param_name, param in sig.parameters.items():
            if param_name == "df":
                bound_args["df"] = df
            elif param_name in kwargs:
                val = kwargs[param_name]
                # Cast integer types if needed
                if param.annotation == int and isinstance(val, (str, float)):
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                bound_args[param_name] = val

        return tool.func(**bound_args)


def create_default_tool_registry() -> ToolRegistry:
    """Instantiate and populate the default registry with all WealthWise finance tools."""
    registry = ToolRegistry()

    # 1. get_spending_summary
    registry.register(
        name="get_spending_summary",
        description="Get total spending, total income, net savings, and category breakdown for a date range.",
        parameters={
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format (optional)",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format (optional)",
                },
            },
            "required": [],
        },
        func=finance_tools.get_spending_summary,
    )

    # 2. get_top_merchants
    registry.register(
        name="get_top_merchants",
        description="Get the top N spending merchants/recipients by total debit amount.",
        parameters={
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of top merchants to return (default is 5)",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format (optional)",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format (optional)",
                },
            },
            "required": [],
        },
        func=finance_tools.get_top_merchants,
    )

    # 3. get_monthly_trend
    registry.register(
        name="get_monthly_trend",
        description="Get monthly spending totals and month-over-month rate of change, optionally filtered by category.",
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category name (e.g. 'Food & Dining', 'Groceries', 'Utilities')",
                },
            },
            "required": [],
        },
        func=finance_tools.get_monthly_trend,
    )

    # 4. detect_recurring_payments
    registry.register(
        name="detect_recurring_payments",
        description="Detect regular subscriptions, EMIs, and bills based on repeated identical debit amounts and cadence (~monthly or bi-weekly).",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        func=finance_tools.detect_recurring_payments,
    )

    # 5. detect_anomalies
    registry.register(
        name="detect_anomalies",
        description="Detect unusually large debit transactions (statistical IQR outliers) and exact same-day duplicate charges.",
        parameters={
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["iqr"],
                    "description": "Detection method, defaults to 'iqr'",
                },
            },
            "required": [],
        },
        func=finance_tools.detect_anomalies,
    )

    # 6. simulate_budget
    registry.register(
        name="simulate_budget",
        description="Simulate projected monthly and annual savings from reducing spending in a specific category by a given percentage.",
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The category to simulate reduction for (e.g. 'Food & Dining', 'Groceries')",
                },
                "reduction_percent": {
                    "type": "number",
                    "description": "The percentage to reduce spending by (between 1 and 100)",
                },
            },
            "required": ["category", "reduction_percent"],
        },
        func=finance_tools.simulate_budget,
    )

    # 7. compare_periods
    registry.register(
        name="compare_periods",
        description="Compare total spending, income, and percentage difference between two date ranges (period A vs period B).",
        parameters={
            "type": "object",
            "properties": {
                "period_a_start": {"type": "string", "description": "Start date for Period A (YYYY-MM-DD)"},
                "period_a_end": {"type": "string", "description": "End date for Period A (YYYY-MM-DD)"},
                "period_b_start": {"type": "string", "description": "Start date for Period B (YYYY-MM-DD)"},
                "period_b_end": {"type": "string", "description": "End date for Period B (YYYY-MM-DD)"},
            },
            "required": ["period_a_start", "period_a_end", "period_b_start", "period_b_end"],
        },
        func=finance_tools.compare_periods,
    )

    return registry


# Global default instance
default_tool_registry = create_default_tool_registry()

