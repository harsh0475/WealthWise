"""Streamlit frontend dashboard and conversational interface for WealthWise.
"""
import io
import json
import pandas as pd
import streamlit as st
import altair as alt

# Direct imports from app logic for smooth standalone Streamlit execution or API calls
from app.ingestion.parser import parse_statement
from app.ingestion.synthetic import generate_and_save_synthetic_statement
from app.safety.pii import PIIVault
from app.categorization.categorizer import categorize_statement
from app.agent.loop import AgentLoop
from app.tools import finance_tools
from app.config import settings


st.set_page_config(
    page_title="WealthWise | AI Finance Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1E88E5;
    }
    .trace-box {
        background-color: #262730;
        border-radius: 6px;
        padding: 12px;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .grounded-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .warning-badge {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "vault" not in st.session_state:
    st.session_state.vault = PIIVault()
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- Sidebar: Controls & Ingestion ---
with st.sidebar:
    st.title("💰 WealthWise")
    st.caption("Privacy-First Personal Finance Agent")
    st.divider()

    st.subheader("1. Load Bank Statement")
    uploaded_file = st.file_uploader("Upload Statement (CSV or XLSX)", type=["csv", "xlsx"])

    col1, col2 = st.columns(2)
    with col1:
        if uploaded_file is not None:
            if st.button("Process File", use_container_width=True):
                ext = uploaded_file.name.split(".")[-1].lower()
                with st.spinner("Ingesting, redacting PII, and categorizing..."):
                    raw_df = parse_statement(uploaded_file.getvalue(), file_type=ext)
                    # Pipeline order: Redact -> Categorize -> Agent
                    st.session_state.vault = PIIVault()
                    redacted_df = st.session_state.vault.redact_dataframe(raw_df)
                    categorized_df = categorize_statement(redacted_df)

                    st.session_state.df = categorized_df
                    st.session_state.agent = AgentLoop(df=categorized_df, vault=st.session_state.vault)
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.success(f"Loaded {len(categorized_df)} transactions!")
                    st.rerun()

    with col2:
        if st.button("Load Synthetic", use_container_width=True, help="Load 12-month realistic Indian statement with anomalies"):
            with st.spinner("Generating and redacting synthetic statement..."):
                raw_df = generate_and_save_synthetic_statement()
                st.session_state.vault = PIIVault()
                redacted_df = st.session_state.vault.redact_dataframe(raw_df, account_holder_name="Aarav Sharma")
                categorized_df = categorize_statement(redacted_df)

                st.session_state.df = categorized_df
                st.session_state.agent = AgentLoop(df=categorized_df, vault=st.session_state.vault)
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success("Loaded 12-month synthetic statement!")
                st.rerun()

    st.divider()
    st.subheader("LLM Configuration")
    st.info(f"**Provider:** `{settings.llm_provider}`\n\n**Model:** `{settings.llm_model}`\n\n**Temperature:** `{settings.llm_temperature}`")


# --- Main Dashboard ---
if st.session_state.df is None:
    st.info("👈 Upload a bank statement or click **'Load Synthetic'** in the sidebar to begin.")
    st.markdown("""
    ### Why WealthWise?
    - **Zero-Framework Agent Loop**: Pure Python hand-written tool calling agent loop.
    - **Deterministic Analytics**: Pure Pandas functions (IQR outlier detection, subscription cadence analysis, budget simulation).
    - **Privacy-First**: Complete PII redaction and vaulting occurs **before** any data is sent to the LLM.
    - **Prompt Injection Defense**: Untrusted bank narrations wrapped in delimited data blocks with heuristic injection detection.
    - **Measurable Evaluations**: Automated test and eval suite scoring tool selection, numeric grounding, and injection resistance.
    """)
else:
    df = st.session_state.df
    summary = finance_tools.get_spending_summary(df)
    anomalies = finance_tools.detect_anomalies(df)
    recurring = finance_tools.detect_recurring_payments(df)

    # Top KPI Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Debits (Spend)", f"₹{summary['total_spend']:,.2f}")
    with m2:
        st.metric("Total Credits (Income)", f"₹{summary['total_income']:,.2f}")
    with m3:
        st.metric("Net Savings", f"₹{summary['net_savings']:,.2f}")
    with m4:
        st.metric("Recurring Payments", f"{recurring['count']} items", f"₹{recurring['total_estimated_monthly']:,.2f}/mo")
    with m5:
        st.metric("Flagged Anomalies", f"{anomalies['total_flagged']} events")

    # Tabs: Analytics vs Conversational Agent
    tab_chat, tab_analytics, tab_anomalies = st.tabs(["💬 Financial Assistant", "📊 Spending Analytics", "🚨 Flags & Subscriptions"])

    with tab_analytics:
        col_chart1, col_chart2 = st.columns([1, 1])
        with col_chart1:
            st.subheader("Spending by Category")
            if summary["category_breakdown"]:
                cat_df = pd.DataFrame(summary["category_breakdown"])
                chart = alt.Chart(cat_df).mark_bar().encode(
                    x=alt.X("total:Q", title="Total Spend (₹)"),
                    y=alt.Y("category:N", sort="-x", title="Category"),
                    color=alt.Color("category:N", legend=None),
                    tooltip=["category", "total", "count", "percentage"]
                ).properties(height=350)
                st.altair_chart(chart, use_container_width=True)

        with col_chart2:
            st.subheader("Monthly Spending Trend")
            trend_data = finance_tools.get_monthly_trend(df)
            if trend_data["monthly_trend"]:
                trend_df = pd.DataFrame(trend_data["monthly_trend"])
                trend_chart = alt.Chart(trend_df).mark_line(point=True).encode(
                    x=alt.X("month:N", title="Month"),
                    y=alt.Y("total_spend:Q", title="Monthly Spend (₹)"),
                    tooltip=["month", "total_spend", "mom_change_percent"]
                ).properties(height=350)
                st.altair_chart(trend_chart, use_container_width=True)

    with tab_anomalies:
        st.subheader("Statistical Anomalies & Duplicate Charges")
        if anomalies["anomalies"]:
            anom_df = pd.DataFrame(anomalies["anomalies"])
            st.dataframe(anom_df, use_container_width=True)
        else:
            st.success("No anomalies detected.")

        st.divider()
        st.subheader("Detected Recurring Subscriptions & EMIs")
        if recurring["recurring_payments"]:
            rec_df = pd.DataFrame(recurring["recurring_payments"])
            st.dataframe(rec_df, use_container_width=True)
        else:
            st.info("No recurring patterns identified.")

    with tab_chat:
        st.subheader("Ask WealthWise")
        st.caption("Ask questions about spending, recurring bills, anomalies, or simulate what-if budgets.")

        # Render chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "trace" in msg and msg["trace"]:
                    trace = msg["trace"]
                    with st.expander(f"🔍 Agent Execution Trace ({trace['total_iterations']} steps | {trace['total_latency_ms']}ms)"):
                        if msg.get("grounded", True):
                            st.markdown('<span class="grounded-badge">✅ Numerically Grounded</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="warning-badge">⚠️ Grounding Warning</span>', unsafe_allow_html=True)

                        if trace.get("grounding_warnings"):
                            for gw in trace["grounding_warnings"]:
                                st.warning(gw)

                        if trace.get("security_flags"):
                            for sf in trace["security_flags"]:
                                st.info(f"🛡️ Security Note: {sf}")

                        st.write("**Tool Calls:**")
                        for step in trace.get("steps", []):
                            st.markdown(f"- **Step {step['iteration']}**: `{step['tool_name']}` ({step['latency_ms']}ms)")
                            st.json({"arguments": step["arguments"], "result": step["result"]})

        # User Question Input
        if prompt := st.chat_input("E.g., How much did I spend on food in August? What are my recurring bills?"):
            # Add user message to UI
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate Agent Response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing statement with tools..."):
                    agent_resp = st.session_state.agent.run(
                        prompt,
                        chat_history=st.session_state.chat_history
                    )

                    # Restore placeholders for authenticated UI display only
                    display_answer = st.session_state.vault.restore_text(agent_resp.answer)
                    st.markdown(display_answer)

                    # Update history
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": display_answer})

                    trace_dict = agent_resp.trace.model_dump()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": display_answer,
                        "trace": trace_dict,
                        "grounded": agent_resp.grounded,
                    })

                    with st.expander(f"🔍 Agent Execution Trace ({agent_resp.trace.total_iterations} steps | {agent_resp.trace.total_latency_ms}ms)"):
                        if agent_resp.grounded:
                            st.markdown('<span class="grounded-badge">✅ Numerically Grounded</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="warning-badge">⚠️ Grounding Warning</span>', unsafe_allow_html=True)

                        for step in agent_resp.trace.steps:
                            st.markdown(f"- **Step {step.iteration}**: `{step.tool_name}` ({step.latency_ms}ms)")
                            st.json({"arguments": step.arguments, "result": step.result})

