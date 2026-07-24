"""
AI Visual Quality Inspector - Enterprise Dashboard
====================================================
Modernized Streamlit UI with sidebar navigation, card-based
stats grid, defect history log, and settings panel.
All pure Streamlit + custom CSS - no external JS frameworks.
"""

import streamlit as st
import cv2
import time
import datetime
import csv
import io
import os
from ultralytics import YOLO

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Visual Quality Inspector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# CUSTOM CSS - Slate Industrial Dark Theme
# -----------------------------------------------------------------------
st.markdown("""
<style>
    /* -- Import premium fonts -- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* -- Root dark background -- */
    .stApp {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* -- Hide default Streamlit header/toolbar/footer -- */
    header[data-testid="stHeader"] {
        background-color: #0F172A !important;
    }
    #MainMenu, footer, header .stDeployButton { display: none !important; }

    /* ============================================================
       SIDEBAR - Nuclear override for ALL text visibility
       ============================================================ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
        border-right: 1px solid #334155 !important;
    }
    /* Force ALL text inside sidebar to be white */
    section[data-testid="stSidebar"] * {
        color: #F1F5F9 !important;
    }
    /* Radio option labels - crisp white, bold */
    section[data-testid="stSidebar"] .stRadio label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #38BDF8 !important;
    }
    /* Radio label container ("Navigation") - hide since collapsed */
    section[data-testid="stSidebar"] .stRadio > label {
        color: #64748B !important;
    }
    /* Sidebar divider */
    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    /* -- Header Bar -- */
    .app-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .app-header .title-group { flex: 1; }
    .app-header .title {
        font-size: 24px;
        font-weight: 800;
        color: #38BDF8;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header .subtitle {
        font-size: 13px;
        color: #94A3B8;
        margin-top: 3px;
    }
    .app-header .sys-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 600;
        color: #94A3B8;
    }
    .app-header .dot {
        width: 9px; height: 9px;
        border-radius: 50%;
        display: inline-block;
    }
    .app-header .dot-green {
        background: #10B981;
        box-shadow: 0 0 8px rgba(16,185,129,0.5);
    }
    .app-header .dot-amber {
        background: #F59E0B;
        box-shadow: 0 0 8px rgba(245,158,11,0.4);
    }

    /* -- Stat Card -- */
    .stat-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 22px;
        transition: border-color 0.2s ease;
    }
    .stat-card:hover {
        border-color: #475569;
    }
    .stat-card .card-label {
        font-size: 11px;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .stat-card .card-value {
        font-size: 28px;
        font-weight: 800;
        color: #F1F5F9;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
    }
    .stat-card .card-sub {
        font-size: 12px;
        color: #64748B;
        margin-top: 4px;
    }

    /* -- Color Variants for card values -- */
    .val-cyan    { color: #38BDF8 !important; }
    .val-green   { color: #10B981 !important; }
    .val-red     { color: #EF4444 !important; }
    .val-amber   { color: #F59E0B !important; }
    .val-purple  { color: #A78BFA !important; }

    /* -- PASS / FAIL / IDLE Badges -- */
    .badge-pass {
        background: linear-gradient(135deg, #064E3B 0%, #065F46 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 28px 16px;
        text-align: center;
        box-shadow: 0 0 30px rgba(16,185,129,0.15);
    }
    .badge-pass .badge-label {
        font-size: 48px;
        font-weight: 900;
        color: #34D399;
        letter-spacing: 4px;
    }
    .badge-pass .badge-sub {
        font-size: 13px;
        color: #6EE7B7;
        margin-top: 6px;
    }

    .badge-fail {
        background: linear-gradient(135deg, #450A0A 0%, #7F1D1D 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 28px 16px;
        text-align: center;
        box-shadow: 0 0 30px rgba(239,68,68,0.2);
        animation: pulse-fail 1.2s ease-in-out infinite;
    }
    .badge-fail .badge-label {
        font-size: 48px;
        font-weight: 900;
        color: #FCA5A5;
        letter-spacing: 4px;
    }
    .badge-fail .badge-sub {
        font-size: 13px;
        color: #FCA5A5;
        margin-top: 6px;
    }
    @keyframes pulse-fail {
        0%, 100% { box-shadow: 0 0 25px rgba(239,68,68,0.15); }
        50%      { box-shadow: 0 0 45px rgba(239,68,68,0.3);  }
    }

    .badge-idle {
        background: #1E293B;
        border: 2px solid #475569;
        border-radius: 14px;
        padding: 28px 16px;
        text-align: center;
    }
    .badge-idle .badge-label {
        font-size: 32px;
        font-weight: 800;
        color: #64748B;
        letter-spacing: 2px;
    }
    .badge-idle .badge-sub {
        font-size: 13px;
        color: #475569;
        margin-top: 6px;
    }

    /* -- Metric inline cards (FPS / Latency row) -- */
    .metric-row {
        display: flex;
        gap: 12px;
        margin-top: 16px;
    }
    .metric-mini {
        flex: 1;
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }
    .metric-mini .m-label {
        font-size: 10px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-mini .m-value {
        font-size: 22px;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #F1F5F9;
        margin-top: 2px;
    }

    /* -- Section heading -- */
    .section-heading {
        font-size: 13px;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding-bottom: 8px;
        border-bottom: 1px solid #334155;
        margin-bottom: 16px;
        margin-top: 8px;
    }

    /* -- History Table -- */
    .history-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 13px;
    }
    .history-table thead th {
        background: #1E293B;
        color: #38BDF8;
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid #334155;
    }
    .history-table thead th:first-child { border-radius: 8px 0 0 0; }
    .history-table thead th:last-child  { border-radius: 0 8px 0 0; }
    .history-table tbody td {
        padding: 9px 14px;
        border-bottom: 1px solid rgba(51,65,85,0.5);
        color: #E2E8F0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
    }
    .history-table tbody tr:hover {
        background: rgba(56,189,248,0.04);
    }

    /* -- Defect type pills -- */
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
    }
    .pill-crack   { background: rgba(167,139,250,0.15); color: #C4B5FD; border: 1px solid rgba(167,139,250,0.3); }
    .pill-scratch { background: rgba(245,158,11,0.15);  color: #FCD34D; border: 1px solid rgba(245,158,11,0.3);  }
    .pill-rust    { background: rgba(239,68,68,0.15);   color: #FCA5A5; border: 1px solid rgba(239,68,68,0.3);   }
    .pill-dent    { background: rgba(6,182,212,0.15);   color: #67E8F9; border: 1px solid rgba(6,182,212,0.3);   }
    .pill-other   { background: rgba(100,116,139,0.15); color: #94A3B8; border: 1px solid rgba(100,116,139,0.3); }

    /* -- Empty state -- */
    .empty-state {
        text-align: center;
        padding: 40px 20px;
        color: #475569;
    }
    .empty-state .icon { font-size: 2.5rem; margin-bottom: 8px; }
    .empty-state .msg  { font-size: 14px; }

    /* -- Settings group -- */
    .settings-group {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .settings-group h4 {
        color: #38BDF8;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* -- Streamlit metric overrides -- */
    div[data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* -- Primary button (Start Inspection) -- */
    .stButton > button[kind="primary"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 10px 20px !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0369A1 !important;
    }
    .stButton > button[kind="primary"]:active {
        background-color: #075985 !important;
    }

    /* -- Secondary / Default buttons -- */
    .stButton > button {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
    }
    .stButton > button:hover {
        background: #334155 !important;
        border-color: #475569 !important;
    }

    /* -- Download button -- */
    .stDownloadButton button {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
    }
    .stDownloadButton button:hover {
        background: #334155 !important;
        border-color: #475569 !important;
    }

    /* -- Checkbox override for dark theme -- */
    .stCheckbox label span {
        color: #F1F5F9 !important;
        font-weight: 600 !important;
    }

    /* -- Slider labels -- */
    .stSlider label, .stNumberInput label {
        color: #CBD5E1 !important;
    }

    /* -- General dividers -- */
    hr { border-color: #334155 !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------------------------
defaults = {
    "defect_history": [],
    "total_frames": 0,
    "total_defects": 0,
    "pass_count": 0,
    "fail_count": 0,
    "last_run_time": None,
    "last_run_result": None,
    "is_inspecting": False,
    # Settings
    "conf_threshold": 0.45,
    "camera_index": 0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -----------------------------------------------------------------------
# MODEL LOADER
# -----------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    """Load YOLO model with multi-path search + fallback."""
    search_paths = [
        "best.pt",
        "runs/detect/defect_inspector_model/weights/best.pt",
        "runs/detect/runs/detect/defect_inspector_model/weights/best.pt",
        "runs/detect/train/weights/best.pt",
    ]
    for path in search_paths:
        if os.path.exists(path):
            try:
                return YOLO(path), path, True
            except Exception:
                continue
    # Fallback
    try:
        return YOLO("yolov8n.pt"), "yolov8n.pt", False
    except Exception as e:
        return None, f"Error: {e}", False

model, model_path, is_custom_model = load_yolo_model()

# -----------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------

DEFECT_COLORS_BGR = {
    "crack":   (250, 139, 167),
    "scratch": (11, 158, 245),
    "rust":    (68, 68, 239),
    "dent":    (212, 182, 6),
}

def defect_pill(name):
    """Return an HTML pill span for a defect type."""
    n = name.lower()
    cls_map = {"crack": "pill-crack", "scratch": "pill-scratch",
               "rust": "pill-rust", "dent": "pill-dent"}
    pill_cls = cls_map.get(n, "pill-other")
    return f'<span class="pill {pill_cls}">{n.upper()}</span>'

def bbox_color(name):
    """Return BGR color for drawing bounding boxes."""
    return DEFECT_COLORS_BGR.get(name.lower(), (180, 180, 180))

def draw_boxes(frame, boxes, names):
    """Draw styled bounding boxes on a frame."""
    annotated = frame.copy()
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = bbox_color(label)

        # Label background
        text = f"{label.upper()} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)),
                      (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 4, max(th + 4, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Box + corner accents
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cl = min(14, (x2 - x1) // 4, (y2 - y1) // 4)
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),
                                (x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(annotated, (cx, cy), (cx + cl * dx, cy), color, 3)
            cv2.line(annotated, (cx, cy), (cx, cy + cl * dy), color, 3)
    return annotated

def generate_csv(history):
    """Generate CSV bytes from defect history."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf,
                            fieldnames=["frame", "timestamp", "type", "confidence"])
    writer.writeheader()
    for row in history:
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


# -----------------------------------------------------------------------
# HEADER BAR
# -----------------------------------------------------------------------
model_dot = "dot-green" if is_custom_model else "dot-amber"
model_label = "Custom Model" if is_custom_model else "Base Model"

st.markdown(f"""
<div class="app-header">
    <div class="title-group">
        <div class="title">🛡️ AI Visual Quality Inspector</div>
        <div class="subtitle">Edge AI Defect Detection System &bull; Real-Time Quality Control</div>
    </div>
    <div class="sys-status">
        <span class="dot {model_dot}"></span> {model_label} &nbsp;|&nbsp;
        <span style="color:#E2E8F0; font-family:'JetBrains Mono',monospace; font-size:12px;">
            {os.path.basename(model_path)}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 16px 0;">
        <span style="color:#38BDF8; font-size:18px; font-weight:800; letter-spacing:-0.02em;">
            🛡️ Inspector
        </span>
        <span style="color:#94A3B8; font-size:12px; margin-left:4px;">v2.0</span>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊  Dashboard", "🔍  Inspection Run", "📋  History", "⚙️  Settings"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick stats in sidebar
    total = st.session_state.pass_count + st.session_state.fail_count
    pass_rate = (st.session_state.pass_count / total * 100) if total > 0 else 0
    st.markdown(f"""
    <div style="font-size:11px; color:#38BDF8; text-transform:uppercase; font-weight:700;
                letter-spacing:0.08em; margin-bottom:8px;">
        Session Summary
    </div>
    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
        <span style="color:#CBD5E1; font-size:13px;">Frames</span>
        <span style="color:#FFFFFF; font-weight:700; font-family:'JetBrains Mono',monospace;
              font-size:13px;">
            {st.session_state.total_frames}
        </span>
    </div>
    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
        <span style="color:#CBD5E1; font-size:13px;">Defects</span>
        <span style="color:#EF4444; font-weight:700; font-family:'JetBrains Mono',monospace;
              font-size:13px;">
            {st.session_state.total_defects}
        </span>
    </div>
    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
        <span style="color:#CBD5E1; font-size:13px;">Pass Rate</span>
        <span style="color:#10B981; font-weight:700; font-family:'JetBrains Mono',monospace;
              font-size:13px;">
            {pass_rate:.1f}%
        </span>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# PAGE: DASHBOARD
# =====================================================================
if page == "📊  Dashboard":

    # -- Stat Cards Row --
    c1, c2, c3, c4 = st.columns(4, gap="medium")

    last_run_display = st.session_state.last_run_time or "---"
    last_result = st.session_state.last_run_result

    with c1:
        result_color = ""
        if last_result == "PASS":
            result_color = "val-green"
        elif last_result == "FAIL":
            result_color = "val-red"
        st.markdown(f"""
        <div class="stat-card">
            <div class="card-label">Last Inspection</div>
            <div class="card-value {result_color}">{last_result or "---"}</div>
            <div class="card-sub">{last_run_display}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="card-label">Camera Status</div>
            <div class="card-value val-cyan">Ready</div>
            <div class="card-sub">Index {st.session_state.camera_index}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        m_color = "val-green" if is_custom_model else "val-amber"
        m_text = "Trained" if is_custom_model else "Base"
        st.markdown(f"""
        <div class="stat-card">
            <div class="card-label">Model Status</div>
            <div class="card-value {m_color}">{m_text}</div>
            <div class="card-sub">{os.path.basename(model_path)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        d_color = "val-red" if st.session_state.total_defects > 0 else "val-green"
        st.markdown(f"""
        <div class="stat-card">
            <div class="card-label">Total Defects</div>
            <div class="card-value {d_color}">{st.session_state.total_defects}</div>
            <div class="card-sub">{st.session_state.total_frames} frames analyzed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # -- Quick Actions --
    act_col1, act_col2, act_col3 = st.columns([2, 1, 1], gap="medium")

    with act_col1:
        st.markdown('<div class="section-heading">Quick Actions</div>',
                    unsafe_allow_html=True)
        st.info("To start inspection: select **🔍 Inspection Run** in the sidebar, "
                "then tick the **Start Live Inspection Stream** checkbox.")

    with act_col2:
        st.markdown('<div class="section-heading">Export</div>',
                    unsafe_allow_html=True)
        if st.session_state.defect_history:
            csv_data = generate_csv(st.session_state.defect_history)
            st.download_button(
                "Download CSV Report",
                data=csv_data,
                file_name=f"inspection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("No Data Yet", disabled=True, use_container_width=True)

    with act_col3:
        st.markdown('<div class="section-heading">Session</div>',
                    unsafe_allow_html=True)
        if st.button("Clear History", use_container_width=True):
            for k in ["defect_history", "total_frames", "total_defects",
                       "pass_count", "fail_count", "last_run_time", "last_run_result"]:
                st.session_state[k] = defaults[k]
            st.rerun()

    # -- Recent Defects Preview --
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Recent Detections</div>',
                unsafe_allow_html=True)

    recent = st.session_state.defect_history[-8:]
    if recent:
        rows = ""
        for entry in reversed(recent):
            pill = defect_pill(entry["type"])
            conf_pct = entry["confidence"] * 100
            rows += f"""
            <tr>
                <td>{entry['frame']}</td>
                <td>{entry['timestamp']}</td>
                <td>{pill}</td>
                <td>{conf_pct:.1f}%</td>
            </tr>"""
        st.markdown(f"""
        <table class="history-table">
            <thead><tr><th>Frame</th><th>Time</th><th>Type</th><th>Conf</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📋</div>
            <div class="msg">No detections yet. Run an inspection to see results here.</div>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# PAGE: INSPECTION RUN
# =====================================================================
elif page == "🔍  Inspection Run":

    if model is None:
        st.error("Failed to load YOLO model. Check installation and model files.")
        st.stop()

    # -- Controls row --
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        conf_thresh = st.slider("Confidence Threshold", 0.10, 1.00,
                                st.session_state.conf_threshold, 0.05,
                                key="insp_conf")
        st.session_state.conf_threshold = conf_thresh
    with ctrl2:
        cam_idx = st.number_input("Camera Index", 0, 10,
                                  st.session_state.camera_index, key="insp_cam")
        st.session_state.camera_index = cam_idx
    with ctrl3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_inspection = st.checkbox("Start Live Inspection Stream",
                                     value=False, key="insp_run")

    st.markdown("---")

    # -- Main layout: feed + status panel --
    col_feed, col_panel = st.columns([5, 2], gap="medium")

    with col_feed:
        st.markdown('<div class="section-heading">Live Camera Feed</div>',
                    unsafe_allow_html=True)
        frame_placeholder = st.empty()

    with col_panel:
        badge_placeholder = st.empty()
        metrics_placeholder = st.empty()
        st.markdown('<div class="section-heading" style="margin-top:16px;">'
                    'Detection Log</div>', unsafe_allow_html=True)
        log_placeholder = st.empty()

    # Show idle state by default
    badge_placeholder.markdown("""
    <div class="badge-idle">
        <div class="badge-label">IDLE</div>
        <div class="badge-sub">Toggle the checkbox above to start</div>
    </div>
    """, unsafe_allow_html=True)

    metrics_placeholder.markdown("""
    <div class="metric-row">
        <div class="metric-mini">
            <div class="m-label">FPS</div>
            <div class="m-value">---</div>
        </div>
        <div class="metric-mini">
            <div class="m-label">Latency</div>
            <div class="m-value">---</div>
        </div>
        <div class="metric-mini">
            <div class="m-label">Frames</div>
            <div class="m-value">0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    log_placeholder.markdown("""
    <div class="empty-state">
        <div class="icon">📷</div>
        <div class="msg">Waiting for inspection stream...</div>
    </div>
    """, unsafe_allow_html=True)

    # -- Live Inspection Loop (while-loop with st.empty placeholders) --
    if run_inspection:
        cap = cv2.VideoCapture(int(cam_idx))
        if not cap.isOpened():
            st.error(f"Cannot open camera at index {cam_idx}")
            st.info(
                "**Troubleshooting:**\n"
                "- Ensure webcam is connected and not used by another app\n"
                "- Try a different Camera Index\n"
                "- If using DroidCam, check the phone app is running"
            )
            st.stop()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_count = 0
        fps_counter = 0
        fps_start = time.time()
        current_fps = 0.0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost camera feed. Reconnecting...")
                    time.sleep(1)
                    cap.release()
                    cap = cv2.VideoCapture(int(cam_idx))
                    if not cap.isOpened():
                        st.error("Camera reconnection failed.")
                        break
                    continue

                frame_count += 1
                fps_counter += 1
                st.session_state.total_frames += 1

                # FPS calculation (every second)
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    current_fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_start = time.time()

                # Inference
                t0 = time.time()
                results = model.predict(frame, conf=conf_thresh, verbose=False)[0]
                latency_ms = (time.time() - t0) * 1000

                # Collect detections
                defects = []
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    label = model.names[cls_id]
                    defects.append({"type": label, "confidence": conf_val})

                # Update history
                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                for d in defects:
                    st.session_state.defect_history.append({
                        "frame": st.session_state.total_frames,
                        "timestamp": now_str,
                        "type": d["type"],
                        "confidence": round(d["confidence"], 4),
                    })
                    st.session_state.total_defects += 1

                is_pass = len(defects) == 0
                if is_pass:
                    st.session_state.pass_count += 1
                else:
                    st.session_state.fail_count += 1

                st.session_state.last_run_time = now_str
                st.session_state.last_run_result = "PASS" if is_pass else "FAIL"

                # Draw boxes & render frame
                annotated = draw_boxes(frame, results.boxes, model.names)
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB",
                                        use_container_width=True)

                # Update badge
                if is_pass:
                    badge_placeholder.markdown("""
                    <div class="badge-pass">
                        <div class="badge-label">PASS</div>
                        <div class="badge-sub">Surface clear &mdash; no defects</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    badge_placeholder.markdown("""
                    <div class="badge-fail">
                        <div class="badge-label">FAIL</div>
                        <div class="badge-sub">Defect detected &mdash; review required</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Update metrics
                fps_color = ("#10B981" if current_fps > 10
                             else ("#F59E0B" if current_fps > 3
                                   else "#EF4444"))
                lat_color = ("#10B981" if latency_ms < 100
                             else ("#F59E0B" if latency_ms < 250
                                   else "#EF4444"))
                metrics_placeholder.markdown(f"""
                <div class="metric-row">
                    <div class="metric-mini">
                        <div class="m-label">FPS</div>
                        <div class="m-value" style="color:{fps_color}">
                            {current_fps:.1f}</div>
                    </div>
                    <div class="metric-mini">
                        <div class="m-label">Latency</div>
                        <div class="m-value" style="color:{lat_color}">
                            {latency_ms:.0f}<span style="font-size:12px;color:#64748B">
                            ms</span></div>
                    </div>
                    <div class="metric-mini">
                        <div class="m-label">Frames</div>
                        <div class="m-value">{st.session_state.total_frames}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Update log
                if defects:
                    log_lines = "".join(
                        f"<div style='margin-bottom:4px;'>"
                        f"{defect_pill(d['type'])} "
                        f"<span style='color:#CBD5E1; font-family:JetBrains Mono,"
                        f"monospace; font-size:12px;'>"
                        f"{d['confidence']*100:.1f}%</span></div>"
                        for d in defects
                    )
                    log_placeholder.markdown(f"""
                    <div style="background:#1E293B; border:1px solid #334155;
                                border-radius:10px; padding:14px 16px;">
                        <div style="font-size:11px; color:#EF4444; font-weight:700;
                                    text-transform:uppercase;
                                    letter-spacing:0.06em; margin-bottom:8px;">
                            Defects Detected
                        </div>
                        {log_lines}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    log_placeholder.markdown("""
                    <div style="background:#1E293B; border:1px solid #334155;
                                border-radius:10px; padding:14px 16px;">
                        <div style="font-size:11px; color:#10B981; font-weight:700;
                                    text-transform:uppercase;
                                    letter-spacing:0.06em; margin-bottom:4px;">
                            Surface Clear
                        </div>
                        <div style="color:#94A3B8; font-size:13px;">
                            No defects detected in current frame.</div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Runtime error: {e}")
        finally:
            if cap:
                cap.release()


# =====================================================================
# PAGE: HISTORY
# =====================================================================
elif page == "📋  History":

    st.markdown('<div class="section-heading">Inspection History</div>',
                unsafe_allow_html=True)

    history = st.session_state.defect_history

    # Summary row
    h1, h2, h3, h4 = st.columns(4, gap="medium")
    with h1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="card-label">Total Detections</div>
            <div class="card-value val-red">{len(history)}</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="card-label">Pass Count</div>
            <div class="card-value val-green">{st.session_state.pass_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="card-label">Fail Count</div>
            <div class="card-value val-red">{st.session_state.fail_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with h4:
        total = st.session_state.pass_count + st.session_state.fail_count
        pr = (st.session_state.pass_count / total * 100) if total > 0 else 0
        pr_color = "val-green" if pr > 90 else ("val-amber" if pr > 70 else "val-red")
        st.markdown(f"""
        <div class="stat-card">
            <div class="card-label">Pass Rate</div>
            <div class="card-value {pr_color}">{pr:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Export
    if history:
        csv_bytes = generate_csv(history)
        st.download_button(
            "Download Full Report (.csv)",
            data=csv_bytes,
            file_name=f"defect_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Full history table
    if history:
        rows = ""
        for entry in reversed(history[-100:]):
            pill = defect_pill(entry["type"])
            conf = entry["confidence"] * 100
            sev = "🔴" if conf > 70 else "🟡"
            rows += f"""
            <tr>
                <td>{entry['frame']}</td>
                <td>{entry['timestamp']}</td>
                <td>{pill}</td>
                <td>{conf:.1f}%</td>
                <td>{sev}</td>
            </tr>"""

        st.markdown(f"""
        <table class="history-table">
            <thead>
                <tr><th>Frame</th><th>Time</th><th>Defect Type</th>
                    <th>Confidence</th><th>Sev</th></tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📋</div>
            <div class="msg">No inspection history yet.
                Run an inspection to populate this table.</div>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# PAGE: SETTINGS
# =====================================================================
elif page == "⚙️  Settings":

    st.markdown('<div class="section-heading">System Configuration</div>',
                unsafe_allow_html=True)

    s1, s2 = st.columns(2, gap="large")

    with s1:
        st.markdown("""
        <div class="settings-group">
            <h4>🎯 Detection Settings</h4>
        </div>
        """, unsafe_allow_html=True)

        new_conf = st.slider(
            "Default Confidence Threshold",
            0.10, 1.00, st.session_state.conf_threshold, 0.05,
            help="Minimum confidence score to flag a region as defective.",
        )
        st.session_state.conf_threshold = new_conf

        new_cam = st.number_input(
            "Default Camera Index",
            min_value=0, max_value=10,
            value=st.session_state.camera_index,
            help="0 = default webcam, 1+ = external cameras or DroidCam.",
        )
        st.session_state.camera_index = new_cam

    with s2:
        st.markdown("""
        <div class="settings-group">
            <h4>📦 Model Information</h4>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#1E293B; border:1px solid #334155;
                    border-radius:10px; padding:16px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#94A3B8; font-size:13px;">Model File</span>
                <span style="color:#F1F5F9; font-family:'JetBrains Mono',monospace;
                      font-size:12px;">
                    {os.path.basename(model_path)}
                </span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#94A3B8; font-size:13px;">Model Type</span>
                <span style="color:{'#10B981' if is_custom_model else '#F59E0B'};
                      font-weight:700; font-size:13px;">
                    {'Custom Trained' if is_custom_model else 'Base (Untrained)'}
                </span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#94A3B8; font-size:13px;">Classes</span>
                <span style="color:#F1F5F9; font-size:12px;">
                    {', '.join(model.names.values()) if model else 'N/A'}
                </span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#94A3B8; font-size:13px;">Full Path</span>
                <span style="color:#64748B; font-family:'JetBrains Mono',monospace;
                      font-size:11px; max-width:200px; text-align:right;
                      word-break:break-all;">
                    {model_path}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="settings-group">
            <h4>🛠️ Session Controls</h4>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Reset All Session Data", use_container_width=True):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()

        st.caption("This clears all defect history and counters for the current session.")
