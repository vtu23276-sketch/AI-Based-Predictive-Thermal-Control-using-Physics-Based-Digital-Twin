"""
Neonatal Incubator — AI Predictive Thermal Control Dashboard
============================================================
Real-time Streamlit simulation comparing Traditional (Bang-Bang) vs
AI Predictive (Linear Regression) control across 5 live disturbances.

Run:  streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from twin.digital_twin import DigitalTwin
from twin.traditional_ctrl import TraditionalController
from twin.ai_ctrl import AIPredictiveController
from disturbances.events import DISTURBANCES, DISTURBANCE_MAP
from analysis.metrics import compute_metrics, improvement_pct
from configs import settings

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Neonatal Thermal Control",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #070c1a !important;
    color: #dde4f0 !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1529 0%, #101c34 100%) !important;
    border-right: 1px solid #1a2d4e !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color: #8ba4c7 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #60a5fa !important; }
.block-container { padding-top: 1rem !important; }
.mcard {
    background: linear-gradient(135deg, #0f1e38 0%, #152340 100%);
    border: 1px solid #1e3557;
    border-radius: 14px;
    padding: 14px 16px;
    text-align: center;
    height: 100%;
    margin-bottom: 4px;
}
.mcard .lbl {
    font-size: 0.68rem; font-weight: 600;
    color: #6b8ab0; text-transform: uppercase;
    letter-spacing: 0.09em; margin-bottom: 5px;
}
.mcard .val { font-size: 1.45rem; font-weight: 700; line-height: 1.1; }
.mcard .sub { font-size: 0.68rem; color: #5a7299; margin-top: 3px; }
.stButton > button {
    background: linear-gradient(135deg, #131f38, #1a2a48) !important;
    color: #c8d8f0 !important;
    border: 1px solid #203452 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 10px 6px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover:not(:disabled) {
    border-color: #4a82c4 !important;
    box-shadow: 0 4px 18px rgba(74,130,196,0.35) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:disabled { opacity: 0.4 !important; }
.sec-hdr {
    font-size: 0.78rem; font-weight: 600; color: #4a82c4;
    text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 1px solid #1a2d4e; padding-bottom: 6px;
    margin: 10px 0 8px 0;
}
.hero { padding: 4px 0 14px 0; }
.hero h1 {
    font-size: 1.9rem; font-weight: 800; margin: 0; line-height: 1.2;
    background: linear-gradient(120deg, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { font-size: 0.85rem; color: #5a7299; margin: 4px 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SETPOINT       = settings.SETPOINT
T_INITIAL      = settings.T_INITIAL
T_AMBIENT      = settings.T_AMBIENT
DISPLAY_WINDOW = settings.DISPLAY_WINDOW
STEPS_PER_FRAME = settings.STEPS_PER_FRAME
FRAME_DELAY    = settings.FRAME_DELAY

# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────
def _reset_sim():
    sp = st.session_state.get("setpoint_slider", SETPOINT)
    st.session_state.trad_twin = DigitalTwin(T_initial=T_INITIAL, T_ambient=T_AMBIENT, T_setpoint=sp)
    st.session_state.ai_twin   = DigitalTwin(T_initial=T_INITIAL, T_ambient=T_AMBIENT, T_setpoint=sp)
    st.session_state.trad_ctrl = TraditionalController(setpoint=sp, dead_band=settings.TRAD_DEAD_BAND)
    st.session_state.ai_ctrl   = AIPredictiveController(setpoint=sp, window=settings.AI_WINDOW, horizon=settings.AI_HORIZON)
    st.session_state.times      = [0]
    st.session_state.trad_temps = [T_INITIAL]
    st.session_state.ai_temps   = [T_INITIAL]
    st.session_state.trad_duty  = [0.0]
    st.session_state.ai_duty    = [0.0]
    st.session_state.active_dist      = None
    st.session_state.dist_remaining   = 0
    st.session_state.dist_events      = []
    st.session_state.running          = False
    st.session_state.elapsed          = 0
    st.session_state.last_report      = None

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    _reset_sim()

# ─────────────────────────────────────────────────────────────────────────────
# Simulation Logic
# ─────────────────────────────────────────────────────────────────────────────
def _advance():
    trad_twin = st.session_state.trad_twin
    ai_twin   = st.session_state.ai_twin

    for _ in range(STEPS_PER_FRAME):
        st.session_state.elapsed += 1
        t = st.session_state.elapsed

        if st.session_state.dist_remaining > 0:
            dist = DISTURBANCE_MAP[st.session_state.active_dist]
            dist.apply_to_twin(trad_twin)
            dist.apply_to_twin(ai_twin)
            st.session_state.dist_remaining -= 1
            if st.session_state.dist_remaining == 0:
                trad_twin.clear_disturbance()
                ai_twin.clear_disturbance()
                ev = st.session_state.dist_events
                if ev:
                    last = ev[-1]
                    ev[-1] = (last[0], t, last[2], last[3], last[4])
                    # Generate post-disturbance report
                    t0, t1 = last[0], t
                    trad_data = st.session_state.trad_temps[t0:t1]
                    ai_data = st.session_state.ai_temps[t0:t1]
                    m_trad = compute_metrics(trad_data, SETPOINT)
                    m_ai = compute_metrics(ai_data, SETPOINT)
                    st.session_state.last_report = {
                        "name": last[2],
                        "trad_mse": m_trad["MSE"],
                        "ai_mse": m_ai["MSE"],
                        "improvement": improvement_pct(m_trad["MSE"], m_ai["MSE"])
                    }
                st.session_state.active_dist = None

        trad_duty = st.session_state.trad_ctrl.compute(trad_twin.T)
        ai_duty   = st.session_state.ai_ctrl.compute(ai_twin.T)
        trad_T    = trad_twin.step(trad_duty)
        ai_T      = ai_twin.step(ai_duty)

        st.session_state.times.append(t)
        st.session_state.trad_temps.append(trad_T)
        st.session_state.ai_temps.append(ai_T)
        st.session_state.trad_duty.append(trad_duty)
        st.session_state.ai_duty.append(ai_duty)

def _trigger(dist_id: str):
    if st.session_state.active_dist is not None:
        return
    dist = DISTURBANCE_MAP[dist_id]
    st.session_state.active_dist    = dist_id
    st.session_state.dist_remaining = dist.duration
    t0 = st.session_state.elapsed
    st.session_state.dist_events.append(
        (t0, t0 + dist.duration, dist.name, dist.color, dist.icon)
    )

# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
def _build_live_chart() -> go.Figure:
    times = st.session_state.times
    trad  = st.session_state.trad_temps
    ai    = st.session_state.ai_temps
    n = len(times)
    s = max(0, n - DISPLAY_WINDOW)
    t_w, tr_w, ai_w = times[s:], trad[s:], ai[s:]

    fig = go.Figure()

    for (t0, t1, name, color, icon) in st.session_state.dist_events:
        if t_w:
            x0 = max(t0, t_w[0])
            x1 = min(t1, t_w[-1])
            if x0 < x1:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.12, line_width=0)
                mid = (x0 + x1) / 2
                fig.add_annotation(x=mid, y=38.6, text=f"<b>{icon} {name}</b>",
                    showarrow=False, font=dict(size=11, color="white"),
                    bgcolor=color, bordercolor="white",
                    borderwidth=1, borderpad=5)

    fig.add_hrect(y0=SETPOINT - 0.5, y1=SETPOINT + 0.5,
                  fillcolor="rgba(255,255,255,0.03)", line_width=0)
    fig.add_trace(go.Scatter(x=t_w, y=[SETPOINT]*len(t_w), name="Setpoint (37.0°C)",
        line=dict(color="#94a3b8", width=1.5, dash="dot"), mode="lines", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t_w, y=tr_w, name="🔴 Traditional (Bang-Bang)",
        line=dict(color="#f97316", width=2), mode="lines"))
    fig.add_trace(go.Scatter(x=t_w, y=ai_w, name="🔵 AI Predictive",
        line=dict(color="#22d3ee", width=2.5), mode="lines"))

    # Use a stable window for the X-axis to prevent jumping
    x_range = [max(0, n - DISPLAY_WINDOW), max(DISPLAY_WINDOW, n)]
    
    fig.update_layout(
        plot_bgcolor="#070c1a", paper_bgcolor="#0d1830",
        font=dict(family="Inter", color="#8ba4c7", size=12),
        xaxis=dict(title="Time (s)", gridcolor="#142038", color="#6b8ab0",
                   showgrid=True, zeroline=False, range=x_range,
                   fixedrange=True), # Prevent user from zooming/panning to keep it stable
        yaxis=dict(title="Temperature (°C)", gridcolor="#142038", color="#6b8ab0",
                   showgrid=True, zeroline=False, range=[34.5, 39.0],
                   fixedrange=True),
        legend=dict(bgcolor="rgba(13,24,48,0.9)", bordercolor="#1e3557", borderwidth=1,
                    font=dict(size=12), x=0.01, y=0.99, xanchor="left", yanchor="top"),
        margin=dict(l=60, r=20, t=20, b=60),
        height=450, hovermode="x unified",
        transition_duration=0, # Crucial: zero transitions for stable movement
        uirevision='constant' # Keeps the UI state (like zoom/pan) stable across reruns
    )
    return fig

def _build_comparison_chart() -> go.Figure:
    trad_m = compute_metrics(st.session_state.trad_temps, SETPOINT)
    ai_m   = compute_metrics(st.session_state.ai_temps,   SETPOINT)
    labels = ["MSE", "RMSE", "MAE", "Max Dev (°C)"]
    trad_v = [trad_m.get("MSE",0), trad_m.get("RMSE",0),
              trad_m.get("MAE",0), trad_m.get("Max Deviation (°C)",0)]
    ai_v   = [ai_m.get("MSE",0),  ai_m.get("RMSE",0),
              ai_m.get("MAE",0),  ai_m.get("Max Deviation (°C)",0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Traditional", x=labels, y=trad_v, marker_color="#f97316"))
    fig.add_trace(go.Bar(name="AI Predictive", x=labels, y=ai_v,  marker_color="#22d3ee"))
    fig.update_layout(
        barmode="group", plot_bgcolor="#070c1a", paper_bgcolor="#0d1830",
        font=dict(family="Inter", color="#8ba4c7"),
        xaxis=dict(gridcolor="#142038", color="#6b8ab0"),
        yaxis=dict(title="Error Value", gridcolor="#142038", color="#6b8ab0"),
        legend=dict(bgcolor="rgba(13,24,48,0.9)", bordercolor="#1e3557", borderwidth=1),
        margin=dict(l=60, r=20, t=16, b=60), height=300,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Incubator Control")
    st.markdown('<div class="sec-hdr">Simulation</div>', unsafe_allow_html=True)
    st.slider("🌡️ Target Temp (°C)", 36.0, 38.5, SETPOINT, 0.1,
              key="setpoint_slider", disabled=st.session_state.running)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Start", disabled=st.session_state.running, type="primary"):
            st.session_state.running = True
            st.rerun()
    with c2:
        if st.button("⏹ Stop", disabled=not st.session_state.running):
            st.session_state.running = False
            st.rerun()
    if st.button("🔄 Reset", use_container_width=True):
        _reset_sim()
        st.rerun()

    st.markdown('<div class="sec-hdr">Controller Info</div>', unsafe_allow_html=True)
    st.markdown("""
**🔴 Traditional**
Bang-bang thermostat — reacts *after* temperature deviates.
Dead band: **±0.3 °C**

**🔵 AI Predictive**
Sliding-window linear regression predicts **10 s ahead**.
Adjusts heating *proactively* before deviation occurs.
""")
    st.markdown('<div class="sec-hdr">Why AI Predictive?</div>', unsafe_allow_html=True)
    st.markdown("""
    **Proactive vs Reactive**
    Traditional bang-bang control only reacts *after* the temperature has already dropped. 
    
    **AI Advantage:**
    The Linear Regression model predicts the future trend. It can activate the heater **before** the disturbance causes a significant drop, maintaining a much tighter stability window which is critical for neonatal health.
    """)

    st.markdown('<div class="sec-hdr">Session Stats</div>', unsafe_allow_html=True)
    st.markdown(f"⏱ Elapsed: **{st.session_state.elapsed} s**")
    st.markdown(f"📊 Data points: **{len(st.session_state.times)}**")
    if st.session_state.dist_events:
        st.markdown('<div class="sec-hdr">Disturbance Log</div>', unsafe_allow_html=True)
        for (t0, t1, name, color, icon) in st.session_state.dist_events:
            st.markdown(f"<span style='color:{color}'>{icon} **{name}**</span> t={t0}–{t1}s",
                        unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main Panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏥 AI Predictive Thermal Control</h1>
  <p>Physics-Based Digital Twin · Neonatal Incubator Simulation · Real-Time Comparison</p>
</div>
""", unsafe_allow_html=True)

# Metric cards
def _mcard(label, val, sub="", color="#60a5fa"):
    return (f'<div class="mcard"><div class="lbl">{label}</div>'
            f'<div class="val" style="color:{color}">{val}</div>'
            f'<div class="sub">{sub}</div></div>')

trad_t   = st.session_state.trad_temps[-1]
ai_t     = st.session_state.ai_temps[-1]
trad_dev = trad_t - SETPOINT
ai_dev   = ai_t   - SETPOINT
trad_mse = compute_metrics(st.session_state.trad_temps, SETPOINT).get("MSE", 0.0)
ai_mse   = compute_metrics(st.session_state.ai_temps,   SETPOINT).get("MSE", 0.0)
impr     = improvement_pct(trad_mse, ai_mse)

status_c = "#34d399" if st.session_state.running else "#4b6080"
status_l = "🟢 RUNNING" if st.session_state.running else "⚫ STOPPED"
trad_c   = "#ef4444" if abs(trad_dev)>0.5 else "#fbbf24" if abs(trad_dev)>0.2 else "#34d399"
ai_c     = "#ef4444" if abs(ai_dev)>0.5   else "#fbbf24" if abs(ai_dev)>0.2   else "#22d3ee"
impr_c   = "#34d399" if impr>0 else "#f97316"

dist_l, dist_c, dist_s = "None", "#4b6080", "Click below to trigger"
if st.session_state.active_dist:
    d = DISTURBANCE_MAP[st.session_state.active_dist]
    dist_l, dist_c = f"{d.icon} {d.name}", d.color
    dist_s = f"{st.session_state.dist_remaining}s remaining"

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.markdown(_mcard("Status",        status_l, f"{st.session_state.elapsed}s", status_c), unsafe_allow_html=True)
mc2.markdown(_mcard("Traditional",   f"{trad_t:.2f}°C", f"Dev: {trad_dev:+.3f}°C", trad_c), unsafe_allow_html=True)
mc3.markdown(_mcard("AI Predictive", f"{ai_t:.2f}°C",   f"Dev: {ai_dev:+.3f}°C",  ai_c),   unsafe_allow_html=True)
mc4.markdown(_mcard("AI Improvement", f"{impr:+.1f}%",  "vs Traditional MSE", impr_c),       unsafe_allow_html=True)
mc5.markdown(_mcard("Disturbance",   dist_l, dist_s, dist_c),                                 unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Live chart
chart_ph = st.empty()
chart_ph.plotly_chart(_build_live_chart(), use_container_width=True)

# Heater duty bars
dc1, dc2 = st.columns(2)
trad_duty_now = st.session_state.trad_duty[-1]
ai_duty_now   = st.session_state.ai_duty[-1]
dc1.progress(trad_duty_now, text=f"🔴 Traditional heater duty: {trad_duty_now*100:.0f}%")
dc2.progress(ai_duty_now,   text=f"🔵 AI heater duty: {ai_duty_now*100:.0f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Disturbance Panel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🌪️ Disturbance Controls")
st.caption("Trigger disturbances manually — both controllers are affected simultaneously. Observe how AI reacts faster.")

db_cols = st.columns(5)
for i, dist in enumerate(DISTURBANCES):
    with db_cols[i]:
        disabled = (not st.session_state.running or
                    st.session_state.active_dist is not None)
        if st.button(f"{dist.icon}  {dist.name}\n({dist.duration}s)",
                     key=f"btn_{dist.id}",
                     disabled=disabled,
                     help=dist.description):
            _trigger(dist.id)
            st.rerun()

if st.session_state.active_dist:
    d = DISTURBANCE_MAP[st.session_state.active_dist]
    pct = 1.0 - st.session_state.dist_remaining / d.duration
    st.progress(pct, text=f"{d.icon} {d.name} active: {d.description}")
elif st.session_state.last_report:
    rep = st.session_state.last_report
    st.markdown(f"""
    <div style="background: rgba(34, 211, 238, 0.1); border: 1px solid #22d3ee; border-radius: 10px; padding: 15px; margin-top: 10px;">
        <h4 style="margin-top: 0; color: #22d3ee;">📊 Post-Disturbance Report: {rep['name']}</h4>
        <p style="margin-bottom: 5px;">During the last disturbance, the <b>AI Predictive</b> model outperformed <b>Traditional</b> control by:</p>
        <div style="display: flex; gap: 20px;">
            <div><span style="color: #8ba4c7; font-size: 0.8rem;">Traditional MSE</span><br><b style="color: #f97316;">{rep['trad_mse']:.6f}</b></div>
            <div><span style="color: #8ba4c7; font-size: 0.8rem;">AI Predictive MSE</span><br><b style="color: #22d3ee;">{rep['ai_mse']:.6f}</b></div>
            <div><span style="color: #8ba4c7; font-size: 0.8rem;">Improvement</span><br><b style="color: #34d399;">{rep['improvement']:+.1f}%</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Performance Analysis
# ─────────────────────────────────────────────────────────────────────────────
if len(st.session_state.times) > 10:
    st.markdown("---")
    st.markdown("### 📊 Performance Analysis")
    pcol1, pcol2 = st.columns(2)

    with pcol1:
        st.markdown("**Error Comparison Chart**")
        st.plotly_chart(_build_comparison_chart(), use_container_width=True)

    with pcol2:
        st.markdown("**Metrics Summary**")
        trad_m = compute_metrics(st.session_state.trad_temps, SETPOINT)
        ai_m   = compute_metrics(st.session_state.ai_temps,   SETPOINT)
        rows = []
        for key in ["MSE","RMSE","MAE","Max Deviation (°C)","Stability Score"]:
            tv = trad_m.get(key, 0.0)
            av = ai_m.get(key, 0.0)
            winner = ("🔵 AI" if av >= tv else "🔴 Trad") if key=="Stability Score" \
                else ("🔵 AI" if av <= tv else "🔴 Trad")
            rows.append({"Metric": key, "Traditional": f"{tv:.4f}",
                         "AI Predictive": f"{av:.4f}", "Better": winner})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if impr > 0:
            st.success(f"✅ AI is **{impr:.1f}% better** than Traditional (lower MSE).")
        elif impr < 0:
            st.warning(f"⚠️ More simulation time needed — Traditional currently has lower MSE.")
        else:
            st.info("Simulation warming up — trigger disturbances to see differentiation.")

# ─────────────────────────────────────────────────────────────────────────────
# Real-Time Loop
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.running:
    _advance()
    time.sleep(FRAME_DELAY)
    st.rerun()
