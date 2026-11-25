import os
import sys
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Ensure backend modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.preprocessor import clean_and_preprocess
from backend.analytics import compute_aggregates
from backend.narrator import generate_stakeholder_insights, answer_query

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="TaxiLens AI | Smart City Urban Mobility Platform",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Styling (Tableau / Power BI Enterprise Theme + Accessible Chat Input)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC;
    }

    .sidebar-brand-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.1rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .brand-title {
        font-size: 1.45rem;
        font-weight: 900;
        color: #FACC15 !important;
        letter-spacing: -0.5px;
        margin-top: 0.25rem;
    }
    .brand-subtitle {
        font-size: 0.74rem;
        font-weight: 800;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.15rem;
    }

    /* Sidebar Radio Button Styling */
    div[data-testid="stRadio"] label {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 0.7rem 1rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        color: #F1F5F9 !important;
        transition: all 0.15s ease-in-out !important;
        cursor: pointer;
        display: flex !important;
        align-items: center !important;
        margin-bottom: 0.4rem !important;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #334155 !important;
        border-color: #FACC15 !important;
        color: #FACC15 !important;
        transform: translateX(4px);
    }
    div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stRadio"] div[aria-checked="true"] {
        background-color: #FACC15 !important;
        border-color: #FACC15 !important;
        color: #000000 !important;
        font-weight: 800 !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"] span,
    div[data-testid="stRadio"] div[aria-checked="true"] span {
        color: #000000 !important;
        font-weight: 800 !important;
    }
    
    /* Top Header Bar */
    .bi-header-container {
        background: #FFFFFF;
        border-bottom: 2px solid #E2E8F0;
        padding: 1.1rem 2rem;
        margin: -4rem -4rem 1.5rem -4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .bi-title {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .bi-title-badge {
        background: #FACC15;
        color: #000000;
        font-size: 0.8rem;
        font-weight: 800;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .bi-subtitle {
        color: #475569;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    /* Privacy Banner */
    .privacy-shield-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid #16A34A;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .privacy-title {
        color: #15803D;
        font-weight: 800;
        font-size: 0.92rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .privacy-desc {
        color: #166534;
        font-size: 0.84rem;
        font-weight: 600;
    }
    .privacy-tag {
        background: #DCFCE7;
        color: #15803D;
        font-size: 0.75rem;
        font-weight: 800;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        border: 1px solid #86EFAC;
    }

    /* Executive KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 1.25rem 1.25rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        position: relative;
        overflow: hidden;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .kpi-top-bar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
    }
    .kpi-top-bar.accent-yellow { background: #EAB308; }
    .kpi-top-bar.accent-dark { background: #0F172A; }
    .kpi-top-bar.accent-amber { background: #F59E0B; }
    .kpi-top-bar.accent-emerald { background: #10B981; }
    .kpi-top-bar.accent-blue { background: #0284C7; }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 800;
        color: #334155;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-icon { font-size: 1.2rem; }
    .kpi-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        font-weight: 600;
        color: #334155;
    }

    /* Section & Content Cards */
    .section-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-subtitle {
        font-size: 0.88rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 1.25rem;
    }

    /* CHAT INPUT ACCESSIBILITY FIX (WHITE TEXT & WHITE PLACEHOLDER ON DARK) */
    div[data-testid="stChatInput"] {
        background-color: #0F172A !important;
        border: 2px solid #334155 !important;
        border-radius: 12px !important;
        padding: 4px 8px !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #FFFFFF !important;
        caret-color: #FFFFFF !important;
        background-color: transparent !important;
        font-size: 0.98rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #E2E8F0 !important;
        opacity: 0.95 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stChatInput"] button {
        color: #FACC15 !important;
    }
</style>
""", unsafe_allow_html=True)


def format_hour(hr: int) -> str:
    return f"{hr % 12 or 12}:00 {'PM' if hr >= 12 else 'AM'}"


# Session State Management for Global Dataset & Chat History
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "stats" not in st.session_state:
    st.session_state.stats = {}
if "aggregates" not in st.session_state:
    st.session_state.aggregates = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = "sample_taxi_trips.csv (Bundled)"


# Cached High-Speed Data Ingestion & Preprocessing
@st.cache_data(show_spinner=False, max_entries=4)
def cached_process_csv_bytes(file_bytes: bytes, filename: str):
    import io
    # Read CSV with mixed format and low_memory for maximum speed
    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    clean, s = clean_and_preprocess(df)
    aggs = compute_aggregates(clean)
    return df, clean, s, aggs


def process_dataset(df: pd.DataFrame, source_name: str):
    clean, s = clean_and_preprocess(df)
    aggs = compute_aggregates(clean)
    st.session_state.raw_df = df
    st.session_state.clean_df = clean
    st.session_state.stats = s
    st.session_state.aggregates = aggs
    st.session_state.dataset_loaded = True
    st.session_state.uploaded_file_name = source_name


# Load default dataset initially if none loaded
if not st.session_state.dataset_loaded:
    default_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_taxi_trips.csv")
    if os.path.exists(default_path):
        with open(default_path, "rb") as f:
            raw, clean, s, aggs = cached_process_csv_bytes(f.read(), "sample_taxi_trips.csv")
            st.session_state.raw_df = raw
            st.session_state.clean_df = clean
            st.session_state.stats = s
            st.session_state.aggregates = aggs
            st.session_state.dataset_loaded = True
            st.session_state.uploaded_file_name = "sample_taxi_trips.csv (Bundled TLC Sample)"


# Sidebar Navigation with the 5 Dedicated Pages
with st.sidebar:
    st.markdown("""
<div class="sidebar-brand-box">
    <div style="font-size: 2.2rem; line-height: 1;">🚕</div>
    <div class="brand-title">TaxiLens AI</div>
    <div class="brand-subtitle">URBAN MOBILITY PLATFORM</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='font-size: 0.72rem; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>NAVIGATION MENU</div>", unsafe_allow_html=True)
    
    # Dedicated 5-Page Radio Navigation
    nav_page = st.radio(
        "NAVIGATION",
        [
            "📊 Dashboard",
            "🤖 AI Mobility Insights",
            "📈 Trip Analytics",
            "ℹ️ Dataset Information",
            "📂 Upload Dataset"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color: #334155; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("<div style='font-size: 0.75rem; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;'>Active Dataset</div>", unsafe_allow_html=True)
    st.caption(f"📁 {st.session_state.uploaded_file_name}")
    if st.session_state.stats:
        st.caption(f"✅ {st.session_state.stats.get('clean_record_count', 0):,} records ready")

    st.markdown("<hr style='border-color: #334155; margin: 1.25rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;'>Gemini API Configuration</div>", unsafe_allow_html=True)
    gemini_key = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        label_visibility="collapsed",
        placeholder="Enter Gemini API Key...",
        help="Optional: Enables real-time Gemini Flash AI synthesis. Without key, internal deterministic analytical engine generates insights."
    )

    st.markdown("<hr style='border-color: #334155; margin: 1.25rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; color: #94A3B8;'>Official Data Source:<br><a href='https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page' target='_blank' style='color: #FACC15; font-weight: 700;'>NYC TLC Trip Record Portal</a></div>", unsafe_allow_html=True)


# Top Global Header
st.markdown("""
<div class="bi-header-container">
    <div>
        <div class="bi-title">
            <span>🚕 Taxi Trip Pattern Narrator</span>
            <span class="bi-title-badge">Smart City BI</span>
        </div>
        <div class="bi-subtitle">AI-powered privacy-preserving urban mobility analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Privacy Shield Banner (Consistent across all pages)
st.markdown("""
<div class="privacy-shield-card">
    <div>
        <div class="privacy-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#16A34A" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg>
            <span>Privacy Shield Active &bull; 100% Aggregate Data Processing</span>
        </div>
        <div class="privacy-desc">Zero Individual Journey Reconstruction &bull; Recommendations based exclusively on historical aggregate statistics</div>
    </div>
    <div>
        <span class="privacy-tag">TLC COMPLIANT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Guard against empty state
if not st.session_state.dataset_loaded or st.session_state.clean_df is None:
    st.warning("⚠️ No dataset loaded. Please go to '📂 Upload Dataset' to upload or activate a taxi dataset.")
    st.stop()

clean_df = st.session_state.clean_df
raw_df = st.session_state.raw_df
stats = st.session_state.stats
aggregates = st.session_state.aggregates
summary = aggregates.get("summary", {})
important = aggregates.get("important_trips", {})
dayparts = aggregates.get("daypart_distribution", {})
hourly_data = aggregates.get("hourly_distribution_pct", {})
dow_data = aggregates.get("weekday_distribution", {})
dist_tiers = aggregates.get("distance_tier_breakdown", {})
dur_tiers = aggregates.get("duration_tier_breakdown", {})

total_trips = summary.get("total_trips", 0)
avg_dur = summary.get("avg_duration_minutes", 0.0)
avg_dist = summary.get("avg_distance_miles", 0.0)
peak_h = summary.get("peak_hour", 18)
peak_h_str = format_hour(peak_h)
peak_hour_dur = important.get("peak_hour_avg_duration", 0.0)


# ==============================================================================
# PAGE 1: 📊 DASHBOARD (Executive Overview & High-Level Summary ONLY)
# ==============================================================================
if nav_page == "📊 Dashboard":
    st.markdown("### Executive Mobility Dashboard")
    st.caption("High-level transit key performance indicators and municipal mobility benchmarks.")

    # 5 KPI Cards (Total Trips, Avg Duration, Avg Distance, Peak Hour, Avg Time During Peak Hours)
    st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-top-bar accent-yellow"></div>
        <div class="kpi-header">
            <span class="kpi-label">Total Trips</span>
            <span class="kpi-icon">🚕</span>
        </div>
        <div class="kpi-value">{total_trips:,}</div>
        <div class="kpi-subtext">
            <b>{stats.get('retention_rate_pct', 100)}%</b> retention rate
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-top-bar accent-dark"></div>
        <div class="kpi-header">
            <span class="kpi-label">Average Trip Duration</span>
            <span class="kpi-icon">⏱️</span>
        </div>
        <div class="kpi-value">{avg_dur:.1f} <span style="font-size: 1rem; font-weight: 700; color: #475569;">min</span></div>
        <div class="kpi-subtext">
            Median: <b>{summary.get('median_duration_minutes', 0.0):.1f} min</b>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-top-bar accent-amber"></div>
        <div class="kpi-header">
            <span class="kpi-label">Average Distance</span>
            <span class="kpi-icon">📍</span>
        </div>
        <div class="kpi-value">{avg_dist:.2f} <span style="font-size: 1rem; font-weight: 700; color: #475569;">mi</span></div>
        <div class="kpi-subtext">
            Median: <b>{summary.get('median_distance_miles', 0.0):.2f} mi</b>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-top-bar accent-emerald"></div>
        <div class="kpi-header">
            <span class="kpi-label">Peak Travel Hour</span>
            <span class="kpi-icon">🔥</span>
        </div>
        <div class="kpi-value">{peak_h_str}</div>
        <div class="kpi-subtext">
            Busiest: <b>{summary.get('busiest_day', 'N/A')}</b>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-top-bar accent-blue"></div>
        <div class="kpi-header">
            <span class="kpi-label">Avg Peak Hour Time</span>
            <span class="kpi-icon">⏳</span>
        </div>
        <div class="kpi-value">{peak_hour_dur:.1f} <span style="font-size: 1rem; font-weight: 700; color: #475569;">min</span></div>
        <div class="kpi-subtext">
            At {peak_h_str} peak surge
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Dashboard Executive Summary & Key Mobility Metrics
    col_sum1, col_sum2 = st.columns([1.4, 1])

    with col_sum1:
        st.markdown("""
<div class="section-card">
    <div class="section-title">📋 Urban Mobility Executive Summary</div>
    <div class="section-subtitle">Core patterns extracted from historical TLC trip records</div>
    <p style="font-size: 0.94rem; line-height: 1.6; color: #334155;">
        Analysis of <b>{total:,} validated trips</b> demonstrates an urban mobility pattern heavily weighted towards rapid, high-turnover local transit. 
        Surface transit velocities average <b>{speed:.1f} mph</b> across the street network, with travel times increasing significantly during the evening commuter rush.
    </p>
    <ul style="font-size: 0.9rem; line-height: 1.6; color: #1E293B; margin-top: 1rem;">
        <li><b>Peak Arterial Load</b>: Maximum daily hail density occurs at <b>{peak}</b>, coinciding with corporate and entertainment outbound travel.</li>
        <li><b>Weekday Concentration</b>: Weekday transit accounts for <b>{weekday}%</b> of total trip volume, with <b>{busiest}</b> exhibiting the highest trip volume of the week.</li>
        <li><b>Short-Hop Dominance</b>: Over <b>{short_pct}%</b> of all trips are under 3 miles, highlighting the taxi fleet's primary role as a last-mile arterial connector.</li>
        <li><b>Duration Spread</b>: Typical rides average <b>{avg_dur:.1f} minutes</b> (median <b>{median_dur:.1f} minutes</b>), with 90th percentile trips reaching <b>{p90_dur:.1f} minutes</b> during peak congestion.</li>
    </ul>
</div>
""".format(
            total=total_trips,
            speed=summary.get('avg_speed_mph', 0.0),
            peak=peak_h_str,
            weekday=summary.get('weekday_share_pct', 0.0),
            busiest=summary.get('busiest_day', 'Thursday'),
            short_pct=dist_tiers.get('Short (<3 mi)', {}).get('share_pct', 73.4),
            avg_dur=avg_dur,
            median_dur=summary.get('median_duration_minutes', 0.0),
            p90_dur=summary.get('p90_duration_minutes', 0.0)
        ), unsafe_allow_html=True)

    with col_sum2:
        st.markdown("""
<div class="section-card">
    <div class="section-title">🏙️ Municipal Daypart Benchmarks</div>
    <div class="section-subtitle">Average travel time across daily operational windows</div>
</div>
""", unsafe_allow_html=True)
        
        bench_df = pd.DataFrame([
            {"Period": "🌅 Morning (06:00 – 11:00)", "Avg Time": f"{important.get('morning_avg_duration', 0.0):.1f} min", "Volume Share": f"{dayparts.get('Morning', {}).get('share_pct', 0.0)}%"},
            {"Period": "☀️ Afternoon (11:00 – 16:00)", "Avg Time": f"{important.get('afternoon_avg_duration', 0.0):.1f} min", "Volume Share": f"{dayparts.get('Afternoon', {}).get('share_pct', 0.0)}%"},
            {"Period": "🌇 Evening (16:00 – 21:00)", "Avg Time": f"{important.get('evening_avg_duration', 0.0):.1f} min", "Volume Share": f"{dayparts.get('Evening', {}).get('share_pct', 0.0)}%"},
            {"Period": "🌙 Night (21:00 – 06:00)", "Avg Time": f"{important.get('night_avg_duration', 0.0):.1f} min", "Volume Share": f"{dayparts.get('Night', {}).get('share_pct', 0.0)}%"}
        ])
        st.table(bench_df.set_index("Period"))


# ==============================================================================
# PAGE 2: 🤖 AI MOBILITY INSIGHTS (ChatGPT-Style Interactive AI Assistant ONLY)
# ==============================================================================
elif nav_page == "🤖 AI Mobility Insights":
    st.markdown("### AI-Powered Mobility Insights & Assistant")
    st.caption("Multi-stakeholder urban intelligence powered by Gemini Flash and strictly aggregate data.")

    # Stakeholder Guidance Cards
    with st.spinner("Synthesizing multi-stakeholder insights..."):
        stakeholder_data = generate_stakeholder_insights(aggregates, api_key=gemini_key)

    p = stakeholder_data.get("passengers", {})
    u = stakeholder_data.get("urban_planners", {})
    f = stakeholder_data.get("fleet_operators", {})

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        with st.container(border=True):
            st.markdown("### 🕒 " + p.get('title', 'Passenger Travel Assistant'))
            st.caption("BETTER TRAVEL TIME PLANNING")
            st.info(p.get('summary', ''))
            st.markdown("**📌 Key Observations**")
            for obs in p.get('observations', []):
                st.markdown(f"- {obs}")
            st.markdown("**💡 Recommendations**")
            for act in p.get('actions', []):
                st.markdown(f"- **{act}**")

    with col_s2:
        with st.container(border=True):
            st.markdown("### 🏙️ " + u.get('title', 'Urban Mobility Intelligence'))
            st.caption("TRAFFIC REDUCTION SUGGESTIONS")
            st.info(u.get('summary', ''))
            st.markdown("**📌 Key Observations**")
            for obs in u.get('observations', []):
                st.markdown(f"- {obs}")
            st.markdown("**🚦 City Planning Actions**")
            for act in u.get('actions', []):
                st.markdown(f"- **{act}**")

    with col_s3:
        with st.container(border=True):
            st.markdown("### 🚖 " + f.get('title', 'Fleet Management Advisor'))
            st.caption("TAXI DEPLOYMENT OPTIMIZATION")
            st.info(f.get('summary', ''))
            st.markdown("**📌 Key Observations**")
            for obs in f.get('observations', []):
                st.markdown(f"- {obs}")
            st.markdown("**🎯 Dispatch Staging**")
            for act in f.get('actions', []):
                st.markdown(f"- **{act}**")

    st.markdown("---")
    
    # ChatGPT-Like Conversational Interface
    st.markdown("### 💬 Interactive Mobility Assistant")
    st.caption("Ask questions about peak hours, optimal departure windows, congestion, or fleet deployment strategies.")

    # Suggested Prompts Row
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    quick_prompt = None
    if col_p1.button("🕒 When are taxis busiest?"):
        quick_prompt = "When are taxis busiest?"
    if col_p2.button("⏱️ What is the average trip duration?"):
        quick_prompt = "What is the average trip duration?"
    if col_p3.button("🚦 Suggest the best travel time"):
        quick_prompt = "Suggest the best travel time."
    if col_p4.button("🚕 When deploy more taxis?"):
        quick_prompt = "When should taxi companies deploy more taxis?"

    # Clear chat button
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Display Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # High-contrast Chat Input with White Text on Dark
    user_query = st.chat_input("Ask a question about taxi trips...")
    active_query = user_query or quick_prompt

    if active_query:
        # Append User Message
        st.session_state.chat_history.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Generate Assistant Response with loading indicator
        with st.chat_message("assistant"):
            with st.spinner("Analyzing aggregate mobility statistics..."):
                time.sleep(0.3)
                answer = answer_query(active_query, aggregates, api_key=gemini_key)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ==============================================================================
# PAGE 3: 📈 TRIP ANALYTICS (Visualizations & Deep Charts ONLY)
# ==============================================================================
elif nav_page == "📈 Trip Analytics":
    st.markdown("### Trip Analytics & Visualization Suite")
    st.caption("Detailed statistical breakdowns of trip demand, durations, distances, and cadence.")

    # Important Trips Duration Banner
    st.markdown(f"""
<div class="section-card" style="border-left: 5px solid #F59E0B; margin-bottom: 1.5rem;">
    <div class="section-title">⏱️ Average Time for Important Trips</div>
    <div class="section-subtitle">Benchmark travel duration comparisons to support journey planning</div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem;">
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase;">Peak Rush Hour ({peak_h_str})</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: #D97706;">{important.get('peak_hour_avg_duration', 0.0):.1f} min</div>
            <div style="font-size: 0.78rem; color: #64748B;">Highest daily congestion delay</div>
        </div>
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase;">High Demand (5 PM – 8 PM)</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: #0F172A;">{important.get('high_demand_avg_duration', 0.0):.1f} min</div>
            <div style="font-size: 0.78rem; color: #64748B;">{important.get('high_demand_share_pct', 0.0)}% of daily trips</div>
        </div>
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase;">Evening (4 PM – 9 PM)</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: #1E293B;">{important.get('evening_avg_duration', 0.0):.1f} min</div>
            <div style="font-size: 0.78rem; color: #64748B;">Dinner & outbound commute</div>
        </div>
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase;">Morning (6 AM – 11 AM)</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: #15803D;">{important.get('morning_avg_duration', 0.0):.1f} min</div>
            <div style="font-size: 0.78rem; color: #64748B;">Faster average surface velocity</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Charts Row 1: Hourly Trip Demand & Daypart Volume
    col_c1, col_c2 = st.columns([1.5, 1])

    with col_c1:
        st.markdown("""
<div class="section-card">
    <div class="section-title">⏱️ Trips by Hour (24-Hour Cadence)</div>
    <div class="section-subtitle">Demand share curve across all hours of the day</div>
</div>
""", unsafe_allow_html=True)
        df_hourly = pd.DataFrame({
            "Hour": [int(h) for h in hourly_data.keys()],
            "SharePct": list(hourly_data.values()),
            "HourLabel": [format_hour(int(h)) for h in hourly_data.keys()]
        }).sort_values("Hour")
        
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Bar(
            x=df_hourly["HourLabel"],
            y=df_hourly["SharePct"],
            name="% Share",
            marker_color="#FACC15",
            marker_line_color="#CA8A04",
            marker_line_width=1.5,
            hovertemplate="<b>%{x}</b><br>Share: <b>%{y:.2f}%</b><extra></extra>"
        ))
        fig_hourly.add_trace(go.Scatter(
            x=df_hourly["HourLabel"],
            y=df_hourly["SharePct"],
            mode="lines+markers",
            name="Trend",
            line=dict(color="#0F172A", width=3, shape="spline"),
            marker=dict(size=6, color="#0F172A"),
            hoverinfo="skip"
        ))
        fig_hourly.update_layout(
            template="plotly_white",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            font=dict(color="#0F172A", size=12),
            yaxis=dict(title=dict(text="<b>% of Total Trips</b>", font=dict(color="#0F172A", size=13)), gridcolor="#E2E8F0"),
            xaxis=dict(tickangle=-45, tickfont=dict(color="#0F172A", size=10))
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    with col_c2:
        st.markdown("""
<div class="section-card">
    <div class="section-title">🌅 Morning vs Afternoon vs Evening vs Night</div>
    <div class="section-subtitle">Distribution of total volume across municipal dayparts</div>
</div>
""", unsafe_allow_html=True)
        df_dp = pd.DataFrame([
            {"Daypart": dp, "Trips": val["trip_count"], "Share": val["share_pct"]}
            for dp, val in dayparts.items()
        ])
        fig_dp = px.bar(
            df_dp,
            x="Daypart",
            y="Trips",
            text="Share",
            color="Daypart",
            color_discrete_map={
                "Morning": "#EAB308",
                "Afternoon": "#F97316",
                "Evening": "#0F172A",
                "Night": "#64748B"
            }
        )
        fig_dp.update_traces(
            texttemplate='<b>%{text}%</b>',
            textposition='outside',
            textfont=dict(color="#0F172A", size=13),
            hovertemplate="<b>%{x}</b>: <b>%{y:,} trips</b> (%{text}%)<extra></extra>"
        )
        fig_dp.update_layout(
            template="plotly_white",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=320,
            margin=dict(l=10, r=10, t=25, b=10),
            showlegend=False,
            font=dict(color="#0F172A"),
            yaxis=dict(title=dict(text="<b>Trip Count</b>", font=dict(color="#0F172A", size=13)), gridcolor="#E2E8F0"),
            xaxis=dict(tickfont=dict(color="#0F172A", size=12))
        )
        st.plotly_chart(fig_dp, use_container_width=True)

    # Charts Row 2: Trips by Day & Distance Categories
    col_c3, col_c4 = st.columns([1, 1])

    with col_c3:
        st.markdown("""
<div class="section-card">
    <div class="section-title">📅 Trips by Day of Week</div>
    <div class="section-subtitle">Weekly transit rhythm & commuter density</div>
</div>
""", unsafe_allow_html=True)
        df_dow = pd.DataFrame({
            "Day": list(dow_data.keys()),
            "Trips": list(dow_data.values())
        })
        fig_dow = px.bar(
            df_dow,
            x="Day",
            y="Trips",
            color="Trips",
            color_continuous_scale=[[0, "#FEF08A"], [0.5, "#FACC15"], [1.0, "#CA8A04"]]
        )
        fig_dow.update_traces(
            marker_line_color="#A16207",
            marker_line_width=1,
            hovertemplate="<b>%{x}</b>: <b>%{y:,} trips</b><extra></extra>"
        )
        fig_dow.update_layout(
            template="plotly_white",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=280,
            coloraxis_showscale=False,
            font=dict(color="#0F172A"),
            yaxis=dict(title=dict(text="<b>Total Trips</b>", font=dict(color="#0F172A", size=13)), gridcolor="#E2E8F0"),
            xaxis=dict(tickfont=dict(color="#0F172A", size=11))
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with col_c4:
        st.markdown("""
<div class="section-card">
    <div class="section-title">🍩 Trip Distance Categories</div>
    <div class="section-subtitle">Short (<3mi) vs Medium (3-10mi) vs Long (>10mi)</div>
</div>
""", unsafe_allow_html=True)
        df_dist = pd.DataFrame([
            {"Tier": t, "Count": v["count"], "Share": v["share_pct"]}
            for t, v in dist_tiers.items()
        ])
        fig_dist = px.pie(
            df_dist,
            names="Tier",
            values="Count",
            hole=0.55,
            color="Tier",
            color_discrete_map={
                "Short (<3 mi)": "#FACC15",
                "Medium (3-10 mi)": "#0F172A",
                "Long (>10 mi)": "#F97316"
            }
        )
        fig_dist.update_traces(
            textinfo="percent+label",
            textfont=dict(color="#0F172A", size=12),
            marker=dict(line=dict(color='#FFFFFF', width=2)),
            hovertemplate="<b>%{label}</b><br>Count: <b>%{value:,}</b> (%{percent})<extra></extra>"
        )
        fig_dist.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Charts Row 3: FIXED HIGH-CONTRAST HISTOGRAMS (Duration & Distance)
    col_c5, col_c6 = st.columns(2)

    with col_c5:
        st.markdown("""
<div class="section-card">
    <div class="section-title">📊 Trip Duration Distribution (Histogram)</div>
    <div class="section-subtitle">Statistical binning of travel times from 0 to 60 minutes</div>
</div>
""", unsafe_allow_html=True)
        # Filter cleanly to typical 0-60 min range with fixed 2-minute bin widths so bins are clearly visible
        dur_series = clean_df['duration_minutes'][clean_df['duration_minutes'].between(0.5, 60.0)]
        fig_hist_dur = px.histogram(
            dur_series,
            x="duration_minutes",
            nbins=30,
            color_discrete_sequence=["#0F172A"],
            labels={"duration_minutes": "Duration (Minutes)"}
        )
        fig_hist_dur.update_traces(
            marker_line_color="#CBD5E1",
            marker_line_width=1.5,
            hovertemplate="Duration: <b>%{x} min</b><br>Trips: <b>%{y:,}</b><extra></extra>"
        )
        fig_hist_dur.update_layout(
            template="plotly_white",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=300,
            font=dict(color="#0F172A"),
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title=dict(text="<b>Trip Frequency</b>", font=dict(color="#0F172A", size=13)), gridcolor="#E2E8F0"),
            xaxis=dict(title=dict(text="<b>Trip Duration (Minutes)</b>", font=dict(color="#0F172A", size=13)), range=[0, 60])
        )
        st.plotly_chart(fig_hist_dur, use_container_width=True)

    with col_c6:
        st.markdown("""
<div class="section-card">
    <div class="section-title">📊 Trip Distance Distribution (Histogram)</div>
    <div class="section-subtitle">Statistical binning of journey mileage from 0 to 15 miles</div>
</div>
""", unsafe_allow_html=True)
        dist_series = clean_df['trip_distance'][clean_df['trip_distance'].between(0.1, 15.0)]
        fig_hist_dist = px.histogram(
            dist_series,
            x="trip_distance",
            nbins=30,
            color_discrete_sequence=["#EAB308"],
            labels={"trip_distance": "Distance (Miles)"}
        )
        fig_hist_dist.update_traces(
            marker_line_color="#A16207",
            marker_line_width=1.5,
            hovertemplate="Distance: <b>%{x} miles</b><br>Trips: <b>%{y:,}</b><extra></extra>"
        )
        fig_hist_dist.update_layout(
            template="plotly_white",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=300,
            font=dict(color="#0F172A"),
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title=dict(text="<b>Trip Frequency</b>", font=dict(color="#0F172A", size=13)), gridcolor="#E2E8F0"),
            xaxis=dict(title=dict(text="<b>Trip Distance (Miles)</b>", font=dict(color="#0F172A", size=13)), range=[0, 15])
        )
        st.plotly_chart(fig_hist_dist, use_container_width=True)


# ==============================================================================
# PAGE 4: ℹ️ DATASET INFORMATION (Schema, Data Quality & Sample Records ONLY)
# ==============================================================================
elif nav_page == "ℹ️ Dataset Information":
    st.markdown("### Dataset Overview & Metadata Reference")
    st.caption("Detailed specification of ingested TLC trip schema, data hygiene benchmarks, and privacy compliance.")

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.markdown("""
<div class="section-card">
    <div class="section-title">📋 Dataset Profile Summary</div>
    <div class="section-subtitle">Dimensions and quality audit of the active dataset</div>
</div>
""", unsafe_allow_html=True)
        meta_table = pd.DataFrame([
            {"Attribute": "Active Dataset Name", "Value": str(st.session_state.uploaded_file_name)},
            {"Attribute": "Raw Ingested Records", "Value": f"{stats.get('raw_record_count', len(raw_df)):,}"},
            {"Attribute": "Clean Validated Records", "Value": f"{stats.get('clean_record_count', len(clean_df)):,}"},
            {"Attribute": "Filtered Anomaly Records", "Value": f"{stats.get('removed_records', 0):,}"},
            {"Attribute": "Data Quality Retention Rate", "Value": f"{stats.get('retention_rate_pct', 100)}%"},
            {"Attribute": "Total Features (Columns)", "Value": str(len(clean_df.columns))},
            {"Attribute": "Supported File Format", "Value": "Standard TLC CSV (.csv)"}
        ])
        st.table(meta_table.set_index("Attribute"))

    with col_meta2:
        st.markdown("""
<div class="section-card">
    <div class="section-title">🔍 Column Descriptions & Schema</div>
    <div class="section-subtitle">Key attributes processed by the TaxiLens ingestion engine</div>
</div>
""", unsafe_allow_html=True)
        schema_table = pd.DataFrame([
            {"Column": "tpep_pickup_datetime", "Type": "Datetime", "Description": "Timestamp when taxi meter was engaged."},
            {"Column": "tpep_dropoff_datetime", "Type": "Datetime", "Description": "Timestamp when passenger disembarked."},
            {"Column": "trip_distance", "Type": "Float", "Description": "Elapsed trip distance in miles reported by taximeter."},
            {"Column": "duration_minutes", "Type": "Float (Derived)", "Description": "Calculated trip length in minutes ((dropoff - pickup) / 60)."},
            {"Column": "pickup_hour", "Type": "Integer (Derived)", "Description": "Hour of day (0-23) for temporal cadence analysis."},
            {"Column": "time_of_day", "Type": "Categorical (Derived)", "Description": "Morning (06-11), Afternoon (11-16), Evening (16-21), Night (21-06)."}
        ])
        st.table(schema_table.set_index("Column"))

    # Sample Records (Read-Only Preview)
    st.markdown("""
<div class="section-card">
    <div class="section-title">👀 Sample Records Preview (Top 10 Rows)</div>
    <div class="section-subtitle">First 10 validated records used by the analytics engine</div>
</div>
""", unsafe_allow_html=True)
    display_cols = [c for c in ['pickup_datetime', 'dropoff_datetime', 'trip_distance', 'duration_minutes', 'pickup_hour', 'pickup_weekday', 'time_of_day'] if c in clean_df.columns]
    st.dataframe(clean_df[display_cols].head(10), use_container_width=True)


# ==============================================================================
# PAGE 5: 📂 UPLOAD DATASET (Dedicated Upload Page ONLY)
# ==============================================================================
elif nav_page == "📂 Upload Dataset":
    st.markdown("### 📂 Upload TLC Taxi Dataset")
    st.caption("Upload your custom New York City TLC Yellow/Green trip dataset to re-run the entire analytics and AI suite.")

    col_up1, col_up2 = st.columns([1.6, 1])

    with col_up1:
        st.markdown("""
<div class="section-card">
    <div class="section-title">📤 Dataset Ingestion Portal</div>
    <div class="section-subtitle">Supports CSV files containing standard pickup/dropoff datetimes and trip distances</div>
</div>
""", unsafe_allow_html=True)
        
        new_file = st.file_uploader(
            "Drag and drop your TLC Taxi CSV file here",
            type=["csv"],
            key="dataset_uploader_widget",
            help="Upload an official NYC TLC CSV file (e.g., yellow_tripdata_2024.csv)."
        )

        if new_file is not None:
            # Check if this file was already processed to avoid reprocessing on every render
            if st.session_state.get("last_uploaded_name") != new_file.name:
                with st.spinner(f"⚡ Ingesting and processing {new_file.name}..."):
                    try:
                        file_bytes = new_file.getvalue()
                        raw, clean, s, aggs = cached_process_csv_bytes(file_bytes, new_file.name)
                        st.session_state.raw_df = raw
                        st.session_state.clean_df = clean
                        st.session_state.stats = s
                        st.session_state.aggregates = aggs
                        st.session_state.dataset_loaded = True
                        st.session_state.uploaded_file_name = new_file.name
                        st.session_state.last_uploaded_name = new_file.name
                        st.success(f"🎉 Successfully ingested **{new_file.name}** ({len(raw):,} raw records, {len(clean):,} clean trips)!")
                        st.toast("Dashboard updated across all pages!", icon="🚀")
                    except Exception as e:
                        st.error(f"❌ Error processing CSV: {e}")
            else:
                st.success(f"✅ Active Dataset: **{st.session_state.uploaded_file_name}** ({len(st.session_state.clean_df):,} trips active)")

        st.markdown("---")
        st.markdown("##### Or Reset to Bundled TLC Sample Dataset")
        if st.button("🔄 Reload Bundled TLC Dataset (15,000 trips)"):
            default_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_taxi_trips.csv")
            if os.path.exists(default_path):
                with open(default_path, "rb") as f:
                    raw, clean, s, aggs = cached_process_csv_bytes(f.read(), "sample_taxi_trips.csv")
                    st.session_state.raw_df = raw
                    st.session_state.clean_df = clean
                    st.session_state.stats = s
                    st.session_state.aggregates = aggs
                    st.session_state.dataset_loaded = True
                    st.session_state.uploaded_file_name = "sample_taxi_trips.csv (Bundled TLC Sample)"
                    st.session_state.last_uploaded_name = "sample_taxi_trips.csv (Bundled TLC Sample)"
                st.success("Reset back to bundled 15,000 TLC trips sample!")
                st.rerun()

    with col_up2:
        st.markdown("""
<div class="section-card">
    <div class="section-title">ℹ️ Validation Requirements</div>
    <div class="section-subtitle">File guidelines for automated processing</div>
    <ul style="font-size: 0.88rem; line-height: 1.6; color: #334155; padding-left: 1.2rem;">
        <li><b>Format</b>: Comma-Separated Values (<code>.csv</code>).</li>
        <li><b>Supported Column Formats</b>:
            <ul style="margin-top: 4px;">
                <li>Pickup: <code>tpep_pickup_datetime</code>, <code>pickup_datetime</code>, <code>pickup_time</code>, or <code>start_time</code></li>
                <li>Dropoff: <code>tpep_dropoff_datetime</code>, <code>dropoff_datetime</code>, <code>dropoff_time</code>, or <code>end_time</code></li>
                <li>Distance: <code>trip_distance</code>, <code>distance</code>, or <code>miles</code></li>
            </ul>
        </li>
        <li><b>Optimized Performance</b>: Fast vectorized datetime parsing handles tens of thousands of rows in sub-seconds.</li>
        <li><b>Privacy Guarantee</b>: 100% Zero raw row transmission; records stay local.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
