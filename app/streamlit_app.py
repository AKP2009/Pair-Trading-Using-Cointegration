"""
Regime-Aware Pair Trading on the Nifty 200 -- project dashboard.

    streamlit run app/streamlit_app.py

Every number shown is read from the research outputs in research/ and data/, or from the
bundle that app/build_data.py exports from the frozen final strategy.
"""

from __future__ import annotations

import streamlit as st

from core import data
from core.style import inject_css

st.set_page_config(page_title="Nifty 200 Pair Trading", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()

pages = {
    "Start here": [
        st.Page("views/overview.py", title="Overview & final result", icon=":material/home:", default=True),
        st.Page("views/how_it_works.py", title="How the model works", icon=":material/account_tree:"),
    ],
    "The model": [
        st.Page("views/data_universe.py", title="Data & point-in-time universe", icon=":material/database:"),
        st.Page("views/regimes.py", title="Market regime engine", icon=":material/partly_cloudy_day:"),
    ],
    "Results": [
        st.Page("views/results.py", title="Final strategy results", icon=":material/trophy:"),
        st.Page("views/all_pairs.py", title="All pairs & trades", icon=":material/join_inner:"),
        st.Page("views/pair_explorer.py", title="Pair explorer", icon=":material/query_stats:"),
        st.Page("views/live.py", title="Live signal monitor", icon=":material/sensors:"),
    ],
    "Research journey": [
        st.Page("views/phase1.py", title="Phase 1 · Classic pairs + ML", icon=":material/history_edu:"),
        st.Page("views/research.py", title="Phase 2 · Nifty 200 experiments", icon=":material/science:"),
    ],
    "Reference": [
        st.Page("views/limitations.py", title="Limitations & reproducibility", icon=":material/fact_check:"),
    ],
}

nav = st.navigation(pages)

with st.sidebar:
    m = data.meta()
    st.caption(f"Data: {m['data_start']} → **{m['data_end']}** · {m['n_stocks_with_prices']} stocks with prices · "
               f"benchmark: Nifty 200 TRI ({'approx.' if 'approx' in m['benchmark_note'] else 'official'})")
    st.caption("All returns are net of NSE stock-futures costs + 5 bps slippage per side per leg.")

nav.run()
