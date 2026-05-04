import streamlit as st
import pandas as pd
import numpy as np
from ultralytics import YOLO
from PIL import Image
import datetime
import cv2
import os

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Microclimate Monitor",
    layout="wide",
    page_icon="🌍"
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700&display=swap');

/* Root palette */
:root {
    --bg:       #0b0f14;
    --surface:  #131920;
    --border:   #1f2d3d;
    --accent:   #00e5a0;
    --warn:     #f5a623;
    --danger:   #e8394d;
    --muted:    #5a6a7a;
    --text:     #d4dde6;
    --mono:     'Share Tech Mono', monospace;
    --sans:     'Barlow Condensed', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* Header */
.main-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.75rem;
    margin-bottom: 1.5rem;
}
.main-header h1 {
    font-family: var(--sans);
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0;
}
.main-header p {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    margin: 0.2rem 0 0 0;
    letter-spacing: 0.12em;
}

/* Section label */
.section-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: var(--muted);
    text-transform: uppercase;
    border-left: 2px solid var(--accent);
    padding-left: 0.5rem;
    margin-bottom: 0.75rem;
}

/* Stat cards */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.9rem 1rem;
}
.stat-card .label {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    color: var(--muted);
    text-transform: uppercase;
}
.stat-card .value {
    font-family: var(--mono);
    font-size: 1.5rem;
    color: var(--text);
    margin-top: 0.2rem;
}
.stat-card .unit {
    font-size: 0.7rem;
    color: var(--muted);
}
.stat-card .delta {
    font-size: 0.68rem;
    color: var(--muted);
    margin-top: 0.1rem;
}

/* Anomaly score bar */
.score-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
}
.score-title {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.score-number {
    font-family: var(--mono);
    font-size: 2.8rem;
    line-height: 1;
}
.score-bar-bg {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    margin-top: 0.6rem;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 0.4rem 1.1rem;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 0.5rem;
}
.badge-ok      { background: #0d2e20; color: #00e5a0; border: 1px solid #00e5a0; }
.badge-warn    { background: #2e1f06; color: #f5a623; border: 1px solid #f5a623; }
.badge-danger  { background: #2a0a0e; color: #e8394d; border: 1px solid #e8394d; }

/* Alert box */
.alert-box {
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    line-height: 1.6;
    margin-top: 0.75rem;
}
.alert-ok     { background: #0d2e20; border-left: 3px solid #00e5a0; }
.alert-warn   { background: #2e1f06; border-left: 3px solid #f5a623; }
.alert-danger { background: #2a0a0e; border-left: 3px solid #e8394d; }

/* Table */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 0.73rem;
}
.styled-table th {
    background: var(--border);
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.4rem 0.7rem;
    text-align: left;
}
.styled-table td {
    padding: 0.35rem 0.7rem;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
.styled-table tr:last-child td { border-bottom: none; }

/* Uploader override */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 6px !important;
}

/* Metric overrides */
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    color: var(--accent) !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div > div {
    background-color: var(--accent) !important;
}

/* Button */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 0.4rem 1rem !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌍 Urban Microclimate Monitor</h1>
    <p>AI-POWERED ANOMALY DETECTION · SENSOR FUSION SYSTEM · v2.1</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

@st.cache_data
def load_sensor_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.dropna(subset=["PM2.5", "NO2"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_sub_index_pm25(pm25: float) -> float:
    """
    India CPCB AQI sub-index for PM2.5 (24-hr avg).
    Breakpoints: Good 0-30 | Satisfactory 31-60 | Moderate 61-90 |
                 Poor 91-120 | Very Poor 121-250 | Severe >250
    Returns sub-index on 0-500 scale.
    """
    breakpoints = [
        (0, 30,   0,   50),
        (31, 60,  51,  100),
        (61, 90,  101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (250, 500, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo
    return 500.0


def compute_sub_index_no2(no2: float) -> float:
    """
    India CPCB AQI sub-index for NO2 (µg/m³, 24-hr avg).
    Breakpoints: Good 0-40 | Satisfactory 41-80 | Moderate 81-180 |
                 Poor 181-280 | Very Poor 281-400 | Severe >400
    """
    breakpoints = [
        (0, 40,   0,   50),
        (41, 80,  51,  100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (400, 800, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= no2 <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (no2 - c_lo) + i_lo
    return 500.0


def compute_anomaly_index(pm25: float, no2: float, vision_conf: float) -> dict:
    """
    Fuses sensor AQI sub-indices with vision-model confidence.

    Formula (CPCB-grounded):
      AQI_sensor = max(SI_PM25, SI_NO2)          ← official CPCB rule
      AQI_norm   = AQI_sensor / 500              ← normalise to [0, 1]
      anomaly    = 0.65 × AQI_norm + 0.35 × conf ← weighted fusion

    Weights: sensor data carries more authority (0.65) while vision
    model detection boosts the score (0.35) when visual evidence exists.
    """
    si_pm25 = compute_sub_index_pm25(pm25)
    si_no2  = compute_sub_index_no2(no2)
    aqi_sensor = max(si_pm25, si_no2)
    aqi_norm   = min(aqi_sensor / 500.0, 1.0)
    anomaly    = 0.65 * aqi_norm + 0.35 * vision_conf
    return {
        "si_pm25":    round(si_pm25, 1),
        "si_no2":     round(si_no2, 1),
        "aqi_sensor": round(aqi_sensor, 1),
        "aqi_norm":   round(aqi_norm, 4),
        "anomaly":    round(anomaly, 4),
    }


def status_info(anomaly: float) -> dict:
    if anomaly >= 0.75:
        return {"label": "SEVERE ANOMALY", "cls": "danger",
                "msg": "🚨 Critical microclimate conditions detected. Elevated particulate and visual hazard signals are simultaneous. Immediate intervention recommended."}
    elif anomaly >= 0.55:
        return {"label": "HIGH RISK", "cls": "danger",
                "msg": "⚠️ High anomaly level. Sensor readings exceed safe thresholds and visual detection confirms atmospheric hazard. Limit outdoor exposure."}
    elif anomaly >= 0.35:
        return {"label": "MODERATE RISK", "cls": "warn",
                "msg": "🟡 Moderate anomaly. Pollutant levels are elevated. Sensitive groups should avoid prolonged outdoor activity."}
    else:
        return {"label": "NORMAL", "cls": "ok",
                "msg": "✅ Environmental conditions within acceptable limits. No significant anomaly detected."}


def aqi_bucket_color(bucket: str) -> str:
    mapping = {
        "Good": "#00e5a0", "Satisfactory": "#7ecf72",
        "Moderate": "#f5c842", "Poor": "#f5a623",
        "Very Poor": "#e8394d", "Severe": "#9b1c2e",
        "Unknown": "#5a6a7a",
    }
    return mapping.get(bucket, "#5a6a7a")


# ──────────────────────────────────────────────
# LAYOUT: two columns
# ──────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

# ── LEFT: image + detection ──────────────────
with col_left:
    st.markdown('<div class="section-label">01 — Image Input & Visual Detection</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Urban Scene Image", type=["jpg", "png", "jpeg"],
                                     label_visibility="collapsed")

    vision_conf = 0.0  # default when no image

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scene", use_container_width=True)

        output_folder = "outputs"
        os.makedirs(output_folder, exist_ok=True)

        with st.spinner("Running visual detection models…"):
            model1 = YOLO('../models/smog-detection-4/runs/detect/train4/weights/best.pt')
            model2 = YOLO('../models/fire-and-smoke-detection-yolov8/weights/best.pt')

            results1 = model1(image, conf=0.15, verbose=False)
            results2 = model2(image, conf=0.30, verbose=False)

        # Show smog detection
        det1 = results1[0].plot()
        det1 = cv2.cvtColor(det1, cv2.COLOR_BGR2RGB)
        st.image(det1, caption="Smog / Haze Detection", use_container_width=True)

        # Show fire/smoke detection
        det2 = results2[0].plot()
        det2 = cv2.cvtColor(det2, cv2.COLOR_BGR2RGB)
        st.image(det2, caption="Fire / Smoke Detection", use_container_width=True)

        # Fuse confidence from both models
        confs = []
        for res in [results1[0], results2[0]]:
            if res.boxes is not None and len(res.boxes) > 0:
                confs.append(float(res.boxes.conf.max()))
        vision_conf = max(confs) if confs else 0.0

        st.markdown(f"""
        <div style="font-family:var(--mono);font-size:0.7rem;color:var(--muted);margin-top:0.4rem;">
            VISION CONFIDENCE &nbsp;·&nbsp;
            <span style="color:var(--accent);">{vision_conf:.3f}</span>
            &nbsp; (max across both models)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:var(--surface);border:1px dashed var(--border);
                    border-radius:6px;padding:2rem;text-align:center;
                    font-family:var(--mono);font-size:0.7rem;color:var(--muted);">
            NO IMAGE UPLOADED<br>
            <span style="font-size:0.62rem;">Anomaly score will use sensor data only (vision weight set to 0)</span>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT: sensor data + anomaly score ───────
with col_right:

    # ── Sensor data loader ───────────────────
    st.markdown('<div class="section-label">02 — Sensor Data</div>', unsafe_allow_html=True)

    csv_path = st.text_input("Path to CSV file", value="data/clean/air_quality_clean.csv",
                             label_visibility="collapsed",
                             placeholder="sensor_data.csv")

    try:
        sensor_df = load_sensor_data(csv_path)

        # City filter
        cities = sorted(sensor_df["City"].dropna().unique().tolist())
        selected_city = st.selectbox("City", cities, label_visibility="collapsed")

        city_df = sensor_df[sensor_df["City"] == selected_city].copy()
        latest = city_df.iloc[-1]

        pm25  = float(latest["PM2.5"])
        no2   = float(latest["NO2"])
        pm10  = float(latest.get("PM10", 0) or 0)
        aqi   = latest.get("AQI", None)
        bucket = str(latest.get("AQI_Bucket", "Unknown"))
        date_str = str(latest["Date"])[:10]

        b_color = aqi_bucket_color(bucket)

        # Stat cards
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="label">PM2.5</div>
                <div class="value">{pm25:.1f}<span class="unit"> µg/m³</span></div>
                <div class="delta">Safe &lt; 60</div>
            </div>
            <div class="stat-card">
                <div class="label">PM10</div>
                <div class="value">{pm10:.1f}<span class="unit"> µg/m³</span></div>
                <div class="delta">Safe &lt; 100</div>
            </div>
            <div class="stat-card">
                <div class="label">NO₂</div>
                <div class="value">{no2:.1f}<span class="unit"> µg/m³</span></div>
                <div class="delta">Safe &lt; 40</div>
            </div>
            <div class="stat-card">
                <div class="label">AQI</div>
                <div class="value">{int(aqi) if pd.notna(aqi) else "–"}</div>
                <div class="delta" style="color:{b_color};">{bucket}</div>
            </div>
        </div>
        <div style="font-family:var(--mono);font-size:0.6rem;color:var(--muted);margin-bottom:1rem;">
            LATEST RECORD · {date_str} · {selected_city}
        </div>
        """, unsafe_allow_html=True)

        # ── Anomaly Score ────────────────────
        st.markdown('<div class="section-label">03 — Anomaly Index (Sensor × Vision Fusion)</div>', unsafe_allow_html=True)

        scores = compute_anomaly_index(pm25, no2, vision_conf)
        anomaly = scores["anomaly"]
        status  = status_info(anomaly)
        bar_pct = int(anomaly * 100)
        bar_col = {"ok": "#00e5a0", "warn": "#f5a623", "danger": "#e8394d"}[status["cls"]]

        st.markdown(f"""
        <div class="score-wrap">
            <div class="score-title">COMPOSITE ANOMALY INDEX — CPCB SUB-INDEX FUSION</div>
            <div class="score-number" style="color:{bar_col};">{anomaly:.4f}</div>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{bar_pct}%;background:{bar_col};"></div>
            </div>
            <div style="display:flex;justify-content:space-between;
                        font-family:var(--mono);font-size:0.58rem;color:var(--muted);margin-top:0.3rem;">
                <span>0.00 — NORMAL</span><span>0.50 — MODERATE</span><span>1.00 — SEVERE</span>
            </div>
            <div style="margin-top:0.8rem;">
                <span class="status-badge badge-{status['cls']}">{status['label']}</span>
            </div>
        </div>

        <div class="alert-box alert-{status['cls']}">
            {status['msg']}
        </div>
        """, unsafe_allow_html=True)

        # Sub-index breakdown
        st.markdown('<div class="section-label" style="margin-top:1rem;">04 — Sub-Index Breakdown</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <table class="styled-table">
            <thead>
                <tr><th>Component</th><th>Raw Value</th><th>Sub-Index (0–500)</th><th>Weight</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td>PM2.5 (CPCB)</td>
                    <td>{pm25:.1f} µg/m³</td>
                    <td>{scores['si_pm25']}</td>
                    <td>sensor 65%</td>
                </tr>
                <tr>
                    <td>NO₂ (CPCB)</td>
                    <td>{no2:.1f} µg/m³</td>
                    <td>{scores['si_no2']}</td>
                    <td>sensor 65%</td>
                </tr>
                <tr>
                    <td>AQI Sensor (max SI)</td>
                    <td colspan="2">{scores['aqi_sensor']} → norm {scores['aqi_norm']:.4f}</td>
                    <td>0.65 × norm</td>
                </tr>
                <tr>
                    <td>Vision Confidence</td>
                    <td colspan="2">{vision_conf:.4f}</td>
                    <td>0.35 × conf</td>
                </tr>
                <tr>
                    <td><strong>Anomaly Index</strong></td>
                    <td colspan="2">
                        0.65 × {scores['aqi_norm']:.4f} + 0.35 × {vision_conf:.4f}
                    </td>
                    <td><strong>{anomaly:.4f}</strong></td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        # ── Recent trend table ───────────────
        st.markdown('<div class="section-label" style="margin-top:1rem;">05 — Recent Readings ({selected_city})</div>'.replace("{selected_city}", selected_city), unsafe_allow_html=True)

        recent = city_df.tail(8)[["Date","PM2.5","NO2","AQI","AQI_Bucket"]].copy()
        recent["Date"] = recent["Date"].astype(str).str[:10]
        recent = recent.reset_index(drop=True)

        rows_html = ""
        for _, row in recent.iterrows():
            bc = aqi_bucket_color(str(row["AQI_Bucket"]))
            rows_html += f"""
            <tr>
                <td>{row['Date']}</td>
                <td>{row['PM2.5']:.1f}</td>
                <td>{row['NO2']:.1f}</td>
                <td>{int(row['AQI']) if pd.notna(row['AQI']) else '–'}</td>
                <td style="color:{bc};">{row['AQI_Bucket']}</td>
            </tr>"""

        st.markdown(f"""
        <table class="styled-table">
            <thead>
                <tr><th>Date</th><th>PM2.5</th><th>NO₂</th><th>AQI</th><th>Category</th></tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error(f"CSV not found at `{csv_path}`. Please check the path.")
    except Exception as e:
        st.error(f"Error loading data: {e}")


# ──────────────────────────────────────────────
# FEEDBACK (full width below)
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">06 — Alert Reliability Feedback</div>', unsafe_allow_html=True)

fb_col1, fb_col2 = st.columns([2, 1])
with fb_col1:
    rating = st.slider("How reliable is this alert? (1 = unreliable · 5 = accurate)", 1, 5, 3)
with fb_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Submit Feedback"):
        try:
            anomaly_val = scores["anomaly"] if "scores" in dir() else 0.0
        except Exception:
            anomaly_val = 0.0
        feedback_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "rating": rating,
            "anomaly_index": anomaly_val,
            "vision_conf": vision_conf,
        }
        feedback_df = pd.DataFrame([feedback_data])
        header = not os.path.exists("feedback.csv")
        feedback_df.to_csv("feedback.csv", mode="a", header=header, index=False)
        st.success("Feedback logged ✓")

st.markdown("""
<div style="font-family:var(--mono);font-size:0.58rem;color:var(--muted);
            text-align:center;margin-top:2rem;padding-top:1rem;
            border-top:1px solid var(--border);">
    ANOMALY FORMULA · CPCB AQI SUB-INDEX (PM2.5 + NO₂) × 0.65 + VISION CONF × 0.35
    &nbsp;·&nbsp; SUB-INDEX BREAKPOINTS PER CPCB NATIONAL AQI 2014 NOTIFICATION
</div>
""", unsafe_allow_html=True)