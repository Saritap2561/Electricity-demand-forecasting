import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# GRIDWISE AI — MULTI-PAGE ELECTRICITY DEMAND DASHBOARD
# Replace your existing app.py with this file.
# Required files in the same folder:
#   AEP_hourly.csv
#   electricity_model.joblib
#   evaluation_results.joblib
# ============================================================

st.set_page_config(
    page_title="GRIDWISE AI | Electricity Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------- THEME ---------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: #101f33;
        --panel-2: #142842;
        --border: #263d59;
        --text: #edf5ff;
        --muted: #9db0c8;
        --cyan: #31c5f4;
        --teal: #24d6b5;
        --orange: #ff9f43;
        --red: #ff637d;
    }

    html, body, [class*="css"] {
        font-family: Inter, "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 80% 0%, rgba(28, 94, 143, .20), transparent 36%),
            linear-gradient(145deg, #07111f 0%, #091626 60%, #0b1728 100%);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(7,17,31,.94); }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stDecoration"] { display: none; }
    .block-container {
        max-width: 1500px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, h4, p, label, .stMarkdown, li { color: var(--text); }
    h1 { letter-spacing: -1px; font-weight: 800; }
    h2, h3 { letter-spacing: -.35px; font-weight: 750; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2e 0%, #10243b 100%);
        border-right: 1px solid #203854;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown { color: #edf5ff; }
    .brand-block {
        padding: 12px 8px 20px 8px;
        border-bottom: 1px solid #263b55;
        margin-bottom: 20px;
    }
    .brand-name {
        font-size: 1.45rem;
        font-weight: 850;
        letter-spacing: .4px;
        color: #f5f9ff;
        line-height: 1.25;
    }
    .brand-bolt { color: #ff9f43; }
    .brand-subtitle {
        color: #91a7c2;
        font-size: .72rem;
        letter-spacing: 1.25px;
        font-weight: 700;
        margin-top: 8px;
    }

    /* Sidebar radio navigation */
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 7px; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 9px 10px;
        transition: all .15s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #172d47;
        border-color: #2d4b6b;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label p {
        color: #dce9f8 !important;
        font-weight: 550;
    }

    /* Number input: readable text and visible +/- controls */
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
    [data-testid="stNumberInput"] input {
        color: #10243b !important;
        -webkit-text-fill-color: #10243b !important;
        background: #f7fbff !important;
        border: 1px solid #b8cce1 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        caret-color: #10243b !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button,
    [data-testid="stNumberInput"] button {
        background: #e2effb !important;
        color: #12304d !important;
        border: 1px solid #a9c4df !important;
        opacity: 1 !important;
        min-width: 2rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button svg,
    [data-testid="stNumberInput"] button svg,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button *,
    [data-testid="stNumberInput"] button * {
        color: #12304d !important;
        fill: #12304d !important;
        stroke: #12304d !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover,
    [data-testid="stNumberInput"] button:hover {
        background: #c7e3fb !important;
    }

    /* Metrics and panels */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(30,41,59,.76), rgba(15,23,42,.78));
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(100,137,171,.28);
        border-radius: 15px;
        padding: 17px 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,.13);
        min-height: 115px;
    }
    [data-testid="stMetricLabel"] {
        color: #a9bdd5 !important;
        font-weight: 650;
        font-size: .84rem;
    }
    [data-testid="stMetricValue"] {
        color: #f1f8ff !important;
        font-weight: 800;
        font-size: 1.75rem;
    }
    [data-testid="stMetricDelta"] { font-size: .82rem; }
    .hero {
        background:
            radial-gradient(circle at 90% 10%, rgba(49,197,244,.22), transparent 32%),
            linear-gradient(125deg, #142d4a 0%, #102039 58%, #12364a 100%);
        border: 1px solid #2a5275;
        border-radius: 22px;
        padding: 30px 32px;
        margin: 12px 0 24px 0;
        box-shadow: 0 15px 35px rgba(0,0,0,.18);
    }
    .hero-kicker {
        color: #68d8fa;
        font-size: .76rem;
        letter-spacing: 2px;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero-title {
        color: #f7fbff;
        font-size: clamp(1.8rem, 3vw, 2.75rem);
        line-height: 1.12;
        letter-spacing: -1.2px;
        font-weight: 850;
        margin-bottom: 12px;
    }
    .hero-copy {
        color: #c2d4e9;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 850px;
    }
    .section-kicker {
        color: #6edafa;
        font-size: .73rem;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin: 20px 0 6px 0;
    }
    .info-card {
        background: linear-gradient(145deg, rgba(30,41,59,.70), rgba(15,23,42,.82));
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(100,137,171,.25);
        transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
        border-radius: 15px;
        padding: 18px 19px;
        min-height: 145px;
        height: 100%;
    }
    .info-card:hover {
        transform: translateY(-3px);
        border-color: rgba(49,197,244,.55);
        box-shadow: 0 12px 28px rgba(0,0,0,.18);
    }
    .info-card .icon { font-size: 1.5rem; margin-bottom: 10px; }
    .info-card .title { color: #f0f7ff; font-size: 1.02rem; font-weight: 750; margin-bottom: 7px; }
    .info-card .copy { color: #aabed4; font-size: .88rem; line-height: 1.55; }
    .status-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(36,214,181,.12);
        border: 1px solid rgba(36,214,181,.38);
        color: #7df1d8;
        font-size: .75rem;
        font-weight: 750;
    }
    .muted-note {
        background: #0e2036;
        border: 1px solid #203d5c;
        border-radius: 12px;
        color: #b6cbe1;
        padding: 13px 16px;
        line-height: 1.6;
    }
    .footer {
        border-top: 1px solid #1e344d;
        margin-top: 35px;
        padding-top: 18px;
        color: #7f95af;
        font-size: .78rem;
    }

    /* Buttons, tabs, tables, alerts */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(100deg, #0c9fc8, #22b9c5);
        color: #041522 !important;
        border: 0;
        border-radius: 10px;
        padding: .62rem 1rem;
        font-weight: 800;
        min-height: 2.7rem;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: linear-gradient(100deg, #31c5f4, #43e0cf);
        box-shadow: 0 6px 18px rgba(49,197,244,.17);
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #29435f;
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stAlert"] { border-radius: 12px; }
    hr { border-color: #233a55; }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 9px;
    }
    @media (max-width: 768px) {
        .block-container { padding: 1rem .8rem 2rem .8rem; }
        .hero { padding: 22px 20px; }
        [data-testid="stMetric"] { padding: 12px; }
    }
    /* Fix tooltip background and text visibility */
    div[data-baseweb="tooltip"],
    div[data-testid="stTooltipContent"] {
         background-color: #111827 !important;
         color: #F8FAFC !important;
         border: 1px solid #38BDF8 !important;
         border-radius: 8px !important;
         opacity: 1 !important;
    }

    /* Make all tooltip text readable */
    div[data-baseweb="tooltip"] *,
    div[data-testid="stTooltipContent"] * {
         color: #F8FAFC !important;
         opacity: 1 !important;
    }

    /* Tooltip arrow */
    div[data-baseweb="tooltip"] > div {
         color: #F8FAFC !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------ DATA LOADING ----------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "AEP_hourly.csv"
MODEL_FILE = BASE_DIR / "electricity_model.joblib"
EVALUATION_FILE = BASE_DIR / "evaluation_results.joblib"

@st.cache_data
def load_data(path_string):
    data = pd.read_csv(path_string)
    required = {"Datetime", "AEP_MW"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("CSV is missing columns: " + ", ".join(sorted(missing)))
    data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")
    data["AEP_MW"] = pd.to_numeric(data["AEP_MW"], errors="coerce")
    data = data.dropna(subset=["Datetime", "AEP_MW"])
    data = data.drop_duplicates(subset=["Datetime"]).sort_values("Datetime")
    return data.reset_index(drop=True)

@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)

@st.cache_data
def load_evaluation(path_string):
    return joblib.load(path_string)

missing = [p.name for p in [DATA_FILE, MODEL_FILE, EVALUATION_FILE] if not p.exists()]
if missing:
    st.error("Required project file(s) not found: " + ", ".join(missing))
    st.markdown(
        "Place `app.py`, `AEP_hourly.csv`, `electricity_model.joblib`, and "
        "`evaluation_results.joblib` in the same project folder, then rerun Streamlit."
    )
    st.stop()

try:
    df = load_data(str(DATA_FILE))
    saved_model = load_model(str(MODEL_FILE))
    evaluation = load_evaluation(str(EVALUATION_FILE))
except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.stop()

if not isinstance(saved_model, dict) or "model" not in saved_model or "features" not in saved_model:
    st.error("The model file must be a dictionary containing 'model' and 'features'.")
    st.stop()

model = saved_model["model"]
features = list(saved_model["features"])

if len(df) < 168:
    st.error("At least 168 hourly records are needed for lag-based forecasting.")
    st.stop()
if not hasattr(model, "predict"):
    st.error("The saved model does not have a predict() method.")
    st.stop()

# ----------------------- FEATURE BUILDER --------------------
def build_feature_row(timestamp, values):
    """Build commonly used calendar, lag, and rolling features."""
    arr = np.asarray(values, dtype=float)
    row = {
        "hour": timestamp.hour,
        "day": timestamp.day,
        "day_of_week": timestamp.dayofweek,
        "weekday": timestamp.dayofweek,
        "month": timestamp.month,
        "day_of_year": timestamp.dayofyear,
        "week_of_year": int(timestamp.isocalendar().week),
        "is_weekend": int(timestamp.dayofweek >= 5),
        "lag_1": arr[-1],
        "lag_2": arr[-2],
        "lag_3": arr[-3],
        "lag_24": arr[-24],
        "lag_48": arr[-48],
        "lag_168": arr[-168],
        "rolling_mean_3": float(np.mean(arr[-3:])),
        "rolling_mean_24": float(np.mean(arr[-24:])),
        "rolling_mean_168": float(np.mean(arr[-168:])),
        "rolling_std_24": float(np.std(arr[-24:])),
        "rolling_std_168": float(np.std(arr[-168:])),
        "rolling_min_24": float(np.min(arr[-24:])),
        "rolling_max_24": float(np.max(arr[-24:])),
    }
    # Support common alternate naming conventions.
    row.update({
        "hour_of_day": timestamp.hour,
        "dayofweek": timestamp.dayofweek,
        "dayofyear": timestamp.dayofyear,
        "lag1": arr[-1],
        "lag24": arr[-24],
        "lag168": arr[-168],
        "rolling_avg_24": float(np.mean(arr[-24:])),
        "rolling_avg_168": float(np.mean(arr[-168:])),
        "rolling_std24": float(np.std(arr[-24:])),
        # Exact feature names used by the improved 11-feature model.
        "Hour": timestamp.hour,
        "Day": timestamp.day,
        "Month": timestamp.month,
        "Day_of_Year": timestamp.dayofyear,
        "Is_Weekend": int(timestamp.dayofweek >= 5),
        "Lag_1": arr[-1],
        "Lag_24": arr[-24],
        "Lag_168": arr[-168],
        "Rolling_Mean_3": float(np.mean(arr[-3:])),
        "Rolling_Mean_24": float(np.mean(arr[-24:])),
        "Rolling_Std_24": float(np.std(arr[-24:])),
    })
    unknown = [name for name in features if name not in row]
    if unknown:
        raise ValueError(
            "The saved model expects feature(s) this app does not currently build: "
            + ", ".join(unknown)
            + ". Please check the feature list saved during model training."
        )
    return pd.DataFrame([[row[name] for name in features]], columns=features)

@st.cache_data
def create_forecast(last_datetime, history_values, feature_names, model_path, horizon=24):
    history_values = list(history_values)
    current_time = pd.Timestamp(last_datetime)
    results = []
    loaded = joblib.load(model_path)
    loaded_model = loaded["model"]
    for _ in range(horizon):
        current_time += pd.Timedelta(hours=1)
        input_row = build_feature_row(current_time, history_values)
        prediction = float(loaded_model.predict(input_row)[0])
        history_values.append(prediction)
        results.append({"Datetime": current_time, "Predicted_MW": prediction})
    return pd.DataFrame(results)

try:
    forecast = create_forecast(
        str(df["Datetime"].iloc[-1]),
        tuple(df["AEP_MW"].astype(float)),
        tuple(features),
        str(MODEL_FILE),
        24,
    )
except Exception as exc:
    st.error(f"Forecast generation failed: {exc}")
    st.info(
        "This often means the feature names saved in the model do not match the "
        "features created by the training script. The error above lists the names to check."
    )
    st.stop()

if forecast.empty:
    st.error("The model returned an empty forecast.")
    st.stop()

# ----------------------- GLOBAL SETTINGS --------------------
historical_threshold = float(df["AEP_MW"].quantile(.90))
with st.sidebar:
    st.markdown(
        """
        <div class="brand-block">
          <div class="brand-name"><span class="brand-bolt">ϟ</span> GRIDWISE AI</div>
          <div class="brand-subtitle">INTELLIGENT ELECTRICITY ANALYTICS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### MAIN MENU")
    pages = [
        "⌂  Home",
        "↗  Electricity Forecasting",
        "⚠  Peak Load Monitoring",
        "⇄  Smart Load-Shifting",
        "◎  AI Model Performance",
        "▤  Historical Data Explorer",
    ]
    page = st.radio("Navigate dashboard", pages, label_visibility="collapsed", key="page_nav")
    st.markdown("---")
    st.markdown("#### FORECAST SETTINGS")
    threshold = st.number_input(
        "Peak-risk threshold (MW)",
        min_value=0.0,
        max_value=max(100000.0, float(df["AEP_MW"].max()) * 2),
        value=historical_threshold,
        step=100.0,
        format="%.0f",
        key="global_threshold",
        help="Default: historical 90th percentile. Change this to explore risk alerts.",
    )
    st.caption("Default threshold is based on the historical 90th percentile.")
    st.markdown("---")
    st.markdown(
        '<div style="color:#8da5bf;font-size:.78rem;line-height:1.6">'
        'MODEL STATUS<br><span style="color:#6fe4ca;font-weight:750">● Loaded successfully</span><br>'
        'Forecast horizon: 24 hours<br>Data: AEP hourly history</div>',
        unsafe_allow_html=True,
    )

forecast["Peak_Risk"] = np.where(forecast["Predicted_MW"] >= threshold, "High", "Normal")
peak_demand = float(forecast["Predicted_MW"].max())
min_demand = float(forecast["Predicted_MW"].min())
average_demand = float(forecast["Predicted_MW"].mean())
peak_idx = int(forecast["Predicted_MW"].idxmax())
peak_time = forecast.loc[peak_idx, "Datetime"]
risk_hours = int((forecast["Predicted_MW"] >= threshold).sum())
latest_actual = float(df["AEP_MW"].iloc[-1])

def section_header(kicker, title, subtitle=None):
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.header(title)
    if subtitle:
        st.caption(subtitle)

def make_forecast_chart(history, future, threshold_value, title="Electricity Demand Forecast"):
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=120)
    fig.patch.set_facecolor("#0c192a")
    ax.set_facecolor("#102238")
    ax.plot(history["Datetime"], history["AEP_MW"], color="#91a7c4", linewidth=2.3, label="Historical demand")
    ax.plot(future["Datetime"], future["Predicted_MW"], color="#31c5f4", linewidth=2.8,
            marker="o", markersize=3.8, label="Predicted demand")
    ax.axhline(threshold_value, color="#ff9f43", linestyle="--", linewidth=1.8, label="Peak-risk threshold")
    ax.set_title(title, color="#f0f7ff", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date and time", color="#b8c9dc", labelpad=10)
    ax.set_ylabel("Demand (MW)", color="#b8c9dc", labelpad=10)
    ax.tick_params(axis="both", colors="#aabdd2", labelsize=9)
    ax.grid(True, color="#6b819a", alpha=.18, linewidth=.8)
    for spine in ax.spines.values():
        spine.set_color("#29415c")
    ax.legend(facecolor="#142842", edgecolor="#29415c", labelcolor="#eaf3ff", fontsize=9)
    ax.margins(x=.03, y=.12)
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout(pad=1.8)
    return fig

def show_forecast_chart():
    hist = df.tail(48).copy()
    fig = make_forecast_chart(hist, forecast, threshold)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)

def show_footer():
    st.markdown(
        '<div class="footer">GRIDWISE AI &nbsp;·&nbsp; Machine-learning-based electricity demand forecasting '
        '&nbsp;·&nbsp; Academic demonstration only</div>',
        unsafe_allow_html=True,
    )

# --------------------------- HOME ---------------------------
if page == "⌂  Home":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">AI-POWERED ENERGY INTELLIGENCE</div>
          <div class="hero-title">Smarter demand.<br>Better decisions.</div>
          <div class="hero-copy">
            Forecast electricity demand, identify high-load periods, and explore
            flexible load-shifting scenarios — all in one interactive analytics workspace.
          </div>
          <div style="margin-top:18px"><span class="status-pill">● MODEL READY</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    section_header("LIVE WORKSPACE", "Your energy overview", "A quick view of the next 24-hour model forecast.")
    metric_row([
        ("Latest recorded demand", f"{latest_actual:,.0f} MW", "Last available observation in the dataset"),
        ("Forecast average", f"{average_demand:,.0f} MW", "Average of the next 24 hourly predictions"),
        ("Forecast peak", f"{peak_demand:,.0f} MW", "Maximum predicted demand over the forecast horizon"),
        ("Peak-risk hours", f"{risk_hours} / 24", "Hours at or above your selected threshold"),
    ])
    st.markdown("")
    left, right = st.columns([1.7, 1])
    with left:
        section_header("DEMAND OUTLOOK", "Upcoming electricity demand")
        show_forecast_chart()
    with right:
        section_header("AT A GLANCE", "Forecast highlights")
        st.metric("Lowest predicted demand", f"{min_demand:,.0f} MW")
        st.metric("Peak expected at", pd.Timestamp(peak_time).strftime("%d %b · %H:%M"))
        st.metric("Configured risk threshold", f"{threshold:,.0f} MW")
        if risk_hours:
            st.warning(f"{risk_hours} hour(s) meet or exceed the peak-risk threshold.")
        else:
            st.success("No forecast hours exceed the configured threshold.")
    section_header("EXPLORE THE PLATFORM", "One dashboard. Multiple insights.")
    cards = st.columns(3)
    card_data = [
        ("🔮", "Electricity Forecasting", "Explore hourly predictions and compare them with recent historical demand."),
        ("🚨", "Peak Load Monitoring", "Identify the highest-load hours and review threshold-based alerts."),
        ("🔁", "Smart Load-Shifting", "Run a what-if simulation to move flexible demand between hours."),
        ("🧠", "AI Model Performance", "Review error metrics, baseline comparison, and model feature importance."),
        ("📚", "Historical Data Explorer", "Filter historical records and inspect daily, monthly, and hourly patterns."),
        ("⚡", "Decision support", "Use forecasts as planning guidance, not as live grid measurements."),
    ]
    for i, (icon, title, copy) in enumerate(card_data):
        with cards[i % 3]:
            st.markdown(
                f'<div class="info-card"><div class="icon">{icon}</div>'
                f'<div class="title">{title}</div><div class="copy">{copy}</div></div>',
                unsafe_allow_html=True,
            )
        if i % 3 == 2 and i < len(card_data) - 1:
            cards = st.columns(3)
    st.markdown("")
    st.markdown(
        '<div class="muted-note">ℹ️ <b>Data note:</b> forecasts are generated from the latest timestamp in the '
        'provided historical dataset. They are model predictions, not real-time measurements from the electricity grid.</div>',
        unsafe_allow_html=True,
    )

# ------------------- ELECTRICITY FORECASTING ----------------
elif page == "↗  Electricity Forecasting":
    section_header("FORECAST LAB", "Electricity forecasting", "Explore the next 24 hours of predicted demand.")
    metric_row([
        ("Forecast horizon", "24 hours", "One hourly prediction for each future hour"),
        ("Average demand", f"{average_demand:,.0f} MW", "Mean forecast demand"),
        ("Maximum demand", f"{peak_demand:,.0f} MW", "Highest predicted value"),
        ("Minimum demand", f"{min_demand:,.0f} MW", "Lowest predicted value"),
    ])
    st.markdown("")
    show_forecast_chart()
    st.markdown("#### Forecast records")
    display = forecast.copy()
    display["Datetime"] = display["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
    display["Predicted_MW"] = display["Predicted_MW"].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True, height=430)
    csv_data = forecast.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download 24-hour forecast (CSV)", data=csv_data,
                       file_name="gridwise_24_hour_forecast.csv", mime="text/csv")

# --------------------- PEAK LOAD MONITORING -----------------
elif page == "⚠  Peak Load Monitoring":
    section_header("RISK & ALERTS", "Peak load monitoring", "Find forecast periods that meet or exceed your selected risk threshold.")
    metric_row([
        ("Risk threshold", f"{threshold:,.0f} MW", "Change this from the sidebar"),
        ("High-risk hours", f"{risk_hours} / 24", "Forecast hours at or above the threshold"),
        ("Peak forecast", f"{peak_demand:,.0f} MW", "Maximum predicted demand"),
        ("Peak time", pd.Timestamp(peak_time).strftime("%d %b · %H:%M"), "Timestamp of the maximum forecast"),
    ])
    st.markdown("")
    show_forecast_chart()
    risky = forecast.loc[forecast["Predicted_MW"] >= threshold].copy()
    if risky.empty:
        st.success("No forecast hours meet or exceed the selected threshold.")
    else:
        st.warning(f"{len(risky)} forecast hour(s) meet or exceed {threshold:,.0f} MW.")
        risky["Datetime"] = risky["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
        risky["Predicted_MW"] = risky["Predicted_MW"].round(2)
        st.dataframe(risky, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download peak-risk alerts (CSV)",
            data=risky.to_csv(index=False).encode("utf-8"),
            file_name="gridwise_peak_risk_alerts.csv",
            mime="text/csv",
        )
    st.markdown("#### Highest-demand hours")
    top_hours = forecast.nlargest(8, "Predicted_MW").sort_values("Predicted_MW", ascending=False).copy()
    top_hours["Datetime"] = top_hours["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
    top_hours["Predicted_MW"] = top_hours["Predicted_MW"].round(2)
    st.dataframe(top_hours, use_container_width=True, hide_index=True)

# ----------------------- LOAD SHIFTING ----------------------
elif page == "⇄  Smart Load-Shifting":
    section_header("WHAT-IF SIMULATOR", "Smart load-shifting", "Explore how shifting flexible demand could change the forecast peak.")
    st.markdown(
        '<div class="muted-note">This is a simplified academic simulation. It does not control real equipment or '
        'guarantee that a customer can move a specific amount of demand.</div>', unsafe_allow_html=True
    )
    st.markdown("")
    source_default = int(forecast["Predicted_MW"].idxmax())
    source_idx = st.selectbox(
        "Peak/source hour",
        options=list(range(len(forecast))),
        index=source_default,
        format_func=lambda i: f"{forecast.iloc[i]['Datetime']:%d %b %Y, %H:%M} · {forecast.iloc[i]['Predicted_MW']:,.0f} MW",
        key="shift_source",
    )
    destination_options = [i for i in range(len(forecast)) if i != source_idx]
    lowest_destination = min(destination_options, key=lambda i: float(forecast.iloc[i]["Predicted_MW"]))
    destination_idx = st.selectbox(
        "Destination hour",
        options=destination_options,
        index=destination_options.index(lowest_destination),
        format_func=lambda i: f"{forecast.iloc[i]['Datetime']:%d %b %Y, %H:%M} · {forecast.iloc[i]['Predicted_MW']:,.0f} MW",
        key="shift_destination",
    )
    max_shift = min(2000.0, max(100.0, float(forecast.iloc[source_idx]["Predicted_MW"])))
    shift = st.slider("Flexible demand to shift (MW)", 0.0, max_shift, min(500.0, max_shift), 50.0)
    simulated = forecast.copy()
    # Limit shifting so the source-hour demand does not become negative.
    shift = min(float(shift), float(simulated.iloc[source_idx]["Predicted_MW"]))
    simulated.loc[source_idx, "Predicted_MW"] -= shift
    simulated.loc[destination_idx, "Predicted_MW"] += shift
    original_peak = float(forecast["Predicted_MW"].max())
    new_peak = float(simulated["Predicted_MW"].max())
    reduction = original_peak - new_peak
    st.markdown("#### Simulation results")
    metric_row([
        ("Original peak", f"{original_peak:,.0f} MW", "Before the simulated shift"),
        ("Simulated peak", f"{new_peak:,.0f} MW", "After the simulated shift"),
        ("Peak change", f"{reduction:+,.0f} MW", "Positive means the maximum forecast decreased"),
    ])
    a, b = st.columns(2)
    with a:
        st.markdown("**Source hour**")
        st.write(f"{forecast.iloc[source_idx]['Datetime']:%Y-%m-%d %H:%M}")
        st.write(f"{forecast.iloc[source_idx]['Predicted_MW']:,.0f} MW → {simulated.iloc[source_idx]['Predicted_MW']:,.0f} MW")
    with b:
        st.markdown("**Destination hour**")
        st.write(f"{forecast.iloc[destination_idx]['Datetime']:%Y-%m-%d %H:%M}")
        st.write(f"{forecast.iloc[destination_idx]['Predicted_MW']:,.0f} MW → {simulated.iloc[destination_idx]['Predicted_MW']:,.0f} MW")
        if simulated.iloc[destination_idx]["Predicted_MW"] >= threshold:
            st.warning("The destination hour is at or above the risk threshold after shifting.")
        else:
            st.success("The destination hour remains below the risk threshold.")
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=120)
    fig.patch.set_facecolor("#0c192a")
    ax.set_facecolor("#102238")
    ax.plot(forecast["Datetime"], forecast["Predicted_MW"], color="#91a7c4", marker="o", markersize=3, linewidth=2.1, label="Original forecast")
    ax.plot(simulated["Datetime"], simulated["Predicted_MW"], color="#31c5f4", marker=".", markersize=4, linestyle="--", linewidth=2.4, label="Simulated forecast")
    ax.axhline(threshold, color="#ff9f43", linestyle=":", linewidth=1.8, label="Risk threshold")
    ax.set_title("Original vs simulated demand", color="#f0f7ff", fontweight="bold", pad=14)
    ax.set_xlabel("Date and time", color="#b8c9dc")
    ax.set_ylabel("Demand (MW)", color="#b8c9dc")
    ax.tick_params(colors="#aabdd2")
    ax.grid(True, color="#6b819a", alpha=.18)
    for spine in ax.spines.values():
        spine.set_color("#29415c")
    ax.legend(facecolor="#142842", edgecolor="#29415c", labelcolor="#eaf3ff")
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ---------------------- MODEL PERFORMANCE -------------------
elif page == "◎  AI Model Performance":
    section_header("MODEL INSIGHTS", "AI model performance", "Understand prediction error and inspect which inputs influence the trained model.")
    if not isinstance(evaluation, dict):
        st.warning("The evaluation file does not contain a dictionary of metrics.")
    else:
        keys = [
            "Random Forest MAE", "Baseline MAE", "Random Forest RMSE",
            "Baseline RMSE", "Recursive 24-Hour MAE", "Recursive 24-Hour RMSE",
            "Training Records", "Testing Records",
        ]
        missing_keys = [k for k in keys if k not in evaluation]
        if missing_keys:
            st.warning("Evaluation file is missing: " + ", ".join(missing_keys))
        else:
            rf_mae = float(evaluation["Random Forest MAE"])
            base_mae = float(evaluation["Baseline MAE"])
            rf_rmse = float(evaluation["Random Forest RMSE"])
            base_rmse = float(evaluation["Baseline RMSE"])
            rec_mae = float(evaluation["Recursive 24-Hour MAE"])
            rec_rmse = float(evaluation["Recursive 24-Hour RMSE"])
            metric_row([
                ("Random Forest MAE", f"{rf_mae:,.2f} MW", "Mean absolute error on held-out test data"),
                ("Baseline MAE", f"{base_mae:,.2f} MW", "Previous-day baseline error"),
                ("Random Forest RMSE", f"{rf_rmse:,.2f} MW", "Penalises larger errors more strongly"),
                ("Baseline RMSE", f"{base_rmse:,.2f} MW", "Previous-day baseline RMSE"),
            ])
            st.markdown("")
            metric_row([
                ("Recursive 24-hour MAE", f"{rec_mae:,.2f} MW", "Error for recursive multi-step forecasting"),
                ("Recursive 24-hour RMSE", f"{rec_rmse:,.2f} MW", "RMSE for recursive 24-hour forecasting"),
                ("Training records", f"{int(evaluation['Training Records']):,}", "Records used to fit the model"),
                ("Testing records", f"{int(evaluation['Testing Records']):,}", "Held-out evaluation records"),
            ])
            mae_gain = (base_mae - rf_mae) / base_mae * 100 if base_mae else 0
            rmse_gain = (base_rmse - rf_rmse) / base_rmse * 100 if base_rmse else 0
            st.markdown("#### Improvement over the previous-day baseline")
            metric_row([
                ("MAE error reduction", f"{mae_gain:+.1f}%", "Positive means Random Forest MAE is lower"),
                ("RMSE error reduction", f"{rmse_gain:+.1f}%", "Positive means Random Forest RMSE is lower"),
            ])
            error_df = pd.DataFrame({
                "Model": ["Random Forest", "Previous-day baseline"],
                "MAE (MW)": [rf_mae, base_mae],
                "RMSE (MW)": [rf_rmse, base_rmse],
            }).set_index("Model")
            st.markdown("#### Error comparison")
            st.bar_chart(error_df)
            st.caption("Lower MAE and RMSE indicate better predictions. Test-set metrics and recursive forecast metrics describe different evaluation tasks.")
    st.markdown("---")
    st.markdown("#### Feature importance")
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        if len(importances) == len(features):
            importance = pd.Series(importances, index=features).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(10, max(4, len(features) * .36)), dpi=120)
            fig.patch.set_facecolor("#0c192a")
            ax.set_facecolor("#102238")
            ax.barh(importance.index, importance.values, color="#31c5f4")
            ax.set_xlabel("Relative importance", color="#b8c9dc")
            ax.tick_params(colors="#aabdd2")
            ax.grid(axis="x", color="#6b819a", alpha=.18)
            for spine in ax.spines.values():
                spine.set_color("#29415c")
            ax.set_title("Model feature importance", color="#f0f7ff", fontweight="bold", pad=12)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Feature importance length does not match the saved feature list.")
    else:
        st.info("This model does not expose feature_importances_.")
    st.markdown(
        '<div class="muted-note">Feature importance describes how the model uses inputs; it does not prove that an input causes a demand peak.</div>',
        unsafe_allow_html=True,
    )

# -------------------- HISTORICAL DATA EXPLORER ---------------
elif page == "▤  Historical Data Explorer":
    section_header("DATA EXPLORER", "Historical electricity demand", "Filter the dataset and explore long-term demand patterns.")
    hist_min = df["Datetime"].min().date()
    hist_max = df["Datetime"].max().date()
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input("Start date", value=hist_min, min_value=hist_min, max_value=hist_max, key="hist_start")
    with date_col2:
        end_date = st.date_input("End date", value=hist_max, min_value=hist_min, max_value=hist_max, key="hist_end")
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        filtered = df.iloc[0:0].copy()
    else:
        mask = (df["Datetime"].dt.date >= start_date) & (df["Datetime"].dt.date <= end_date)
        filtered = df.loc[mask].copy()
    if filtered.empty:
        st.info("No records were found in the selected date range.")
    else:
        metric_row([
            ("Records in selection", f"{len(filtered):,}", "Hourly observations in the chosen period"),
            ("Average demand", f"{filtered['AEP_MW'].mean():,.0f} MW", "Mean observed demand"),
            ("Maximum demand", f"{filtered['AEP_MW'].max():,.0f} MW", "Maximum observed demand"),
            ("Minimum demand", f"{filtered['AEP_MW'].min():,.0f} MW", "Minimum observed demand"),
        ])
        st.markdown("")
        fig, ax = plt.subplots(figsize=(12, 4.8), dpi=120)
        fig.patch.set_facecolor("#0c192a")
        ax.set_facecolor("#102238")
        ax.plot(filtered["Datetime"], filtered["AEP_MW"], color="#31c5f4", linewidth=1.5)
        ax.set_title("Historical demand over selected period", color="#f0f7ff", fontweight="bold", pad=14)
        ax.set_xlabel("Date", color="#b8c9dc")
        ax.set_ylabel("Demand (MW)", color="#b8c9dc")
        ax.tick_params(colors="#aabdd2")
        ax.grid(True, color="#6b819a", alpha=.18)
        for spine in ax.spines.values():
            spine.set_color("#29415c")
        fig.autofmt_xdate(rotation=25, ha="right")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("#### Average demand by hour of day")
        hourly = filtered.assign(Hour=filtered["Datetime"].dt.hour).groupby("Hour", as_index=False)["AEP_MW"].mean()
        st.bar_chart(hourly.set_index("Hour"))
        st.markdown("#### Historical records")
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=430)
        st.download_button(
            "⬇ Download filtered historical data (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="gridwise_historical_data.csv",
            mime="text/csv",
        )

show_footer()
