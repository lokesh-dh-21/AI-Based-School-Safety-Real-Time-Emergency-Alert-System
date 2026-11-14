import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="SafeCampus | AI Safety Command",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "temperature_c", "smoke_level", "co_ppm", "gas_level",
    "sound_db", "crowd_count", "motion_index",
    "flame_detected", "event_duration_min",
]
LABELS = {0: "Safe", 1: "Normal Alert", 2: "High-Priority Emergency"}
COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}
DATA_URL = (
    "https://github.com/YBIFoundation/Internship/raw/main/"
    "SchoolSafetyEmergencyDataset.csv"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1220; color: #e5e7eb; }
[data-testid="stSidebar"] { background: #101a2b; border-right: 1px solid #263449; }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1500px; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f8fafc; }
div[data-testid="stMetric"] {
 background: linear-gradient(145deg,#152238,#101a2b);
 border: 1px solid #263b56; padding: 17px 18px; border-radius: 15px;
}
div[data-testid="stMetricLabel"] { color: #9fb0c7; }
div[data-testid="stMetricValue"] { color: #f8fafc; }
.panel {
 background: linear-gradient(145deg,#142137,#101a2b);
 border: 1px solid #263b56; border-radius: 16px;
 padding: 18px 20px; margin-bottom: 12px;
}
.muted { color: #9fb0c7; font-size: 0.88rem; }
.status {
 display:inline-block; padding:7px 12px; border-radius:999px;
 font-weight:700; font-size:0.82rem; letter-spacing:.02em;
}
div.stButton > button {
 border-radius: 10px; border: 1px solid #3b82f6;
 background: #2563eb; color: white; font-weight: 700;
 min-height: 2.7rem;
}
div.stButton > button:hover { background:#1d4ed8; border-color:#60a5fa; color:white; }
hr { border-color: #263449; }
[data-testid="stDataFrame"] { border: 1px solid #263449; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading simulated school-safety dataset...")
def load_dataset():
    return pd.read_csv(DATA_URL)

@st.cache_resource(show_spinner="Training AI classification model...")
def train_ai():
    data = load_dataset()
    X = data[FEATURES]
    y = data["severity"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    return data, model, accuracy, matrix, y_test, predictions

def alert_text(code):
    if code == 0:
        return "CLEAR / SAFE — No active emergency detected."
    if code == 1:
        return "NORMAL ALERT — Please inspect the monitored location."
    return "HIGH-PRIORITY EMERGENCY — Immediate human response required."

def metric_card(label, value, note=""):
    st.metric(label, value, help=note or None)

# Sidebar
with st.sidebar:
    st.markdown("## 🛡️ SafeCampus")
    st.caption("AI SAFETY COMMAND CENTER")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Live Monitoring", "Alert Management", "Analytics", "Project Info"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**SYSTEM STATUS**")
    st.markdown('<span class="status" style="background:#123b2a;color:#4ade80;">● MODEL DEMO READY</span>', unsafe_allow_html=True)
    st.caption("Dataset: simulated academic data")
    st.caption("Mode: local demonstration")
    st.divider()
    st.caption("SafeCampus AI · Student Project Prototype")

st.markdown("# SafeCampus <span style='color:#60a5fa'>/</span> Safety Intelligence", unsafe_allow_html=True)
st.markdown('<p class="muted">AI-assisted school safety monitoring · Multiclass Logistic Regression</p>', unsafe_allow_html=True)

try:
    df, model, accuracy, cm, y_test, predictions = train_ai()
except Exception as exc:
    st.error("Unable to load the training dataset or train the model.")
    st.code(str(exc))
    st.info("Check your internet connection and try restarting the Streamlit app.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# Use the latest recorded reading; otherwise show a neutral demo state.
if st.session_state.history:
    latest = st.session_state.history[-1]
    current_status = latest["severity_code"]
else:
    current_status = 0

if page == "Overview":
    st.markdown("### Command overview")
    st.caption("High-level summary of the model dataset and this browser session.")
    total = len(df)
    safe_count = int((df.severity == 0).sum())
    normal_count = int((df.severity == 1).sum())
    emergency_count = int((df.severity == 2).sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Dataset records", f"{total:,}", "Records in the supplied simulated dataset")
    with c2: metric_card("Safe records", f"{safe_count:,}")
    with c3: metric_card("Normal alerts", f"{normal_count:,}")
    with c4: metric_card("Emergency records", f"{emergency_count:,}")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Incident severity distribution")
        counts = df["severity_label"].value_counts().reindex(
            ["Safe", "Normal Alert", "High-Priority Emergency"]
        ).fillna(0).reset_index()
        counts.columns = ["Severity", "Records"]
        fig = px.bar(counts, x="Severity", y="Records", color="Severity",
                     color_discrete_map={"Safe":"#22c55e","Normal Alert":"#f59e0b",
                                         "High-Priority Emergency":"#ef4444"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                          margin=dict(l=10,r=10,t=15,b=10), height=330)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Model performance")
        st.metric("Test accuracy", f"{accuracy:.1%}")
        st.caption("80/20 stratified split · 240 test records")
        st.progress(float(accuracy))
        st.markdown("**Model:** Multiclass Logistic Regression")
        st.markdown("**Preprocessing:** StandardScaler")
        st.markdown("**Class handling:** Balanced class weights")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### Latest session activity")
    if st.session_state.history:
        show = pd.DataFrame(st.session_state.history).drop(columns=["severity_code"])
        st.dataframe(show.tail(5), use_container_width=True, hide_index=True)
    else:
        st.info("No new readings have been submitted in this session. Open Live Monitoring to test a sensor reading.")

elif page == "Live Monitoring":
    st.markdown("### Live sensor monitoring")
    st.caption("Adjust the simulated sensor values and ask the trained model to classify the reading.")
    with st.form("sensor_form"):
        a, b, c = st.columns(3)
        with a:
            temperature = st.slider("Temperature (°C)", 10.0, 80.0, 28.0, 0.5)
            smoke = st.slider("Smoke level", 0.0, 100.0, 10.0, 0.5)
            co = st.slider("Carbon monoxide (ppm)", 0.0, 100.0, 5.0, 0.5)
        with b:
            gas = st.slider("Gas level", 0.0, 100.0, 15.0, 0.5)
            sound = st.slider("Sound level (dB)", 20.0, 120.0, 55.0, 0.5)
            crowd = st.slider("Crowd count", 0, 100, 20)
        with c:
            motion = st.slider("Motion index", 0.0, 100.0, 20.0, 0.5)
            flame = st.selectbox("Flame detected", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            duration = st.slider("Event duration (minutes)", 0.0, 30.0, 1.0, 0.5)
        location = st.text_input("Monitoring location", "Main Building - Ground Floor")
        submitted = st.form_submit_button("Run AI safety assessment", use_container_width=True)

    if submitted:
        reading = pd.DataFrame([{
            "temperature_c": temperature, "smoke_level": smoke, "co_ppm": co,
            "gas_level": gas, "sound_db": sound, "crowd_count": crowd,
            "motion_index": motion, "flame_detected": flame,
            "event_duration_min": duration,
        }], columns=FEATURES)
        code = int(model.predict(reading)[0])
        probs = model.predict_proba(reading)[0]
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": location,
            "temperature_c": temperature,
            "smoke_level": smoke,
            "co_ppm": co,
            "gas_level": gas,
            "sound_db": sound,
            "crowd_count": crowd,
            "motion_index": motion,
            "flame_detected": flame,
            "event_duration_min": duration,
            "predicted_status": LABELS[code],
            "system_action": alert_text(code),
            "severity_code": code,
        }
        st.session_state.history.append(record)
        st.session_state.latest_probabilities = probs.tolist()
        st.session_state.latest_code = code

    if "latest_code" in st.session_state:
        code = st.session_state.latest_code
        color = COLORS[code]
        st.markdown("---")
        st.markdown("#### AI assessment result")
        st.markdown(
            f'<div class="panel"><span class="status" style="background:{color}22;color:{color};">'
            f'● {LABELS[code].upper()}</span><h3 style="margin-top:16px">{alert_text(code)}</h3>'
            '<p class="muted">Model output for the submitted simulated reading. Verify any concern through established school safety procedures and trained personnel.</p></div>',
            unsafe_allow_html=True,
        )
        pcols = st.columns(3)
        for i, (class_id, label) in enumerate(LABELS.items()):
            with pcols[i]:
                st.metric(label, f"{st.session_state.latest_probabilities[i]*100:.1f}%")
    else:
        st.info("Set the sensor values and click **Run AI safety assessment** to generate a prediction.")

elif page == "Alert Management":
    st.markdown("### Alert management")
    st.caption("Alerts generated during this app session.")
    if not st.session_state.history:
        st.info("No session alerts yet. Generate a reading in Live Monitoring.")
    else:
        hist = pd.DataFrame(st.session_state.history).drop(columns=["severity_code"])
        f = st.multiselect("Filter by status", options=hist["predicted_status"].unique(),
                           default=list(hist["predicted_status"].unique()))
        filtered = hist[hist["predicted_status"].isin(f)].sort_values("timestamp", ascending=False)
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download session alert history (CSV)",
            data=hist.to_csv(index=False).encode("utf-8"),
            file_name="alert_history.csv",
            mime="text/csv",
            use_container_width=False,
        )
        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()

elif page == "Analytics":
    st.markdown("### Model analytics")
    st.caption("Evaluation metrics are calculated from the held-out test set.")
    st.metric("Test accuracy", f"{accuracy:.2%}")
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Predicted Safe","Predicted Normal","Predicted Emergency"],
        y=["Actual Safe","Actual Normal","Actual Emergency"],
        colorscale="Blues", text=cm, texttemplate="%{text}", showscale=True
    ))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=430,
                      margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Rows are actual classes; columns are model predictions.")
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        counts = hist["predicted_status"].value_counts().reset_index()
        counts.columns = ["Predicted status", "Count"]
        fig2 = px.pie(counts, names="Predicted status", values="Count", hole=0.58,
                      color="Predicted status",
                      color_discrete_map={"Safe":"#22c55e","Normal Alert":"#f59e0b",
                                          "High-Priority Emergency":"#ef4444"})
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           height=330, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Session prediction analytics will appear after you submit readings.")

else:
    st.markdown("### Project information")
    st.markdown("""
    <div class="panel">
    <h3>AI-Based School Safety & Real-Time Emergency Alert System</h3>
    <p class="muted">Academic prototype · Multiclass Logistic Regression</p>
    <p>This application demonstrates how simulated sensor readings can be classified into three severity levels:</p>
    <ul>
      <li><b style="color:#22c55e">Safe:</b> no active alert in the prototype.</li>
      <li><b style="color:#f59e0b">Normal Alert:</b> location inspection suggested.</li>
      <li><b style="color:#ef4444">High-Priority Emergency:</b> urgent human response suggested.</li>
    </ul>
    <p>Inputs include temperature, smoke, CO, gas, sound, crowd count, motion, flame detection, and event duration.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("#### Technology stack")
    st.write("Python · Streamlit · Pandas · Scikit-learn · Plotly")
    st.warning(
        "Safety limitation: This project uses a simulated dataset and user-entered demo readings. "
        "It is not connected to physical sensors, does not send real notifications, and is not "
        "validated or certified for real-world emergency response. Human oversight and qualified "
        "safety procedures are essential."
    )

st.divider()
st.caption("SafeCampus AI · Educational demonstration only · Not for real emergency use")
