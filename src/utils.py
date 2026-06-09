import streamlit as st

def format_metric(value: int) -> str:
    """Formats large integers with comma separators for clean UI presentation."""
    return f"{value:,}"

def inject_custom_css():
    """Injects specific modular container card styling to override default markdown."""
    st.markdown("""
        <style>
        .metric-card {
            background-color: #1e293b; 
            padding: 24px; 
            border-radius: 12px; 
            border-left: 6px solid #4f46e5; 
            margin-bottom: 16px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }
        .metric-card h4 {
            margin: 0 0 8px 0;
            color: #94a3b8;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-card h2 {
            margin: 0;
            color: #f8fafc;
            font-size: 2rem;
            font-weight: 700;
        }
        .agent-terminal {
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px;
            font-family: 'Courier New', Courier, monospace;
            color: #38bdf8;
            font-size: 0.85rem;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
