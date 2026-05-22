import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Machine Failure Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
    font-weight: 500;
}

/* TITLE */
.title {
    text-align: center;
    color: white;
    font-size: 40px;
    font-weight: bold;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    color: #CCCCCC;
    font-size: 18px;
    margin-bottom: 20px;
}

/* BUTTON */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    background-color: #FF4B4B;
    color: white;
    font-size: 18px;
    border: none;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #ff2e2e;
}

/* METRICS BASE STYLE */
div[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #222;
    padding: 15px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #B0B0B0 !important;
}

/* PROGRESS BAR */
.stProgress > div > div > div > div {
    background-color: #00FFAA;
}

/* ALERT BOX */
.stAlert {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>⚙️ AI Machine Failure Prediction Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-Time Industrial Equipment Monitoring using LightGBM</div>", unsafe_allow_html=True)

st.write("")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Machine Inputs")

machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])

air_temp = st.sidebar.slider("Air Temperature (K)", 290.0, 320.0, 298.0)
process_temp = st.sidebar.slider("Process Temperature (K)", 300.0, 340.0, 308.0)
rpm = st.sidebar.slider("Rotational Speed (RPM)", 1000.0, 3000.0, 1500.0)
torque = st.sidebar.slider("Torque (Nm)", 0.0, 100.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0.0, 300.0, 10.0)

type_M = 1 if machine_type == "M" else 0
type_L = 1 if machine_type == "L" else 0

# ---------------- PREDICTION ----------------
if st.sidebar.button("🔮 Predict Failure"):

    with st.spinner("Predicting Machine Failure..."):

        input_data = np.array([[
            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear,
            type_M,
            type_L
        ]])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

    st.toast("✅ Prediction Completed")

    # ---------------- KPI CARDS ----------------
    col1, col2, col3 = st.columns(3)

    # FIXED NON-MOVING PERCENTAGE CARD
    with col1:
        risk_color = "#00FFAA" if probability < 0.5 else "#F59E0B" if probability < 0.8 else "#FF4B4B"

        st.markdown(f"""
        <div style="
            background-color:#111827;
            padding:15px;
            border-radius:12px;
            border:1px solid #222;
            text-align:center;
        ">
            <div style="color:#B0B0B0; font-size:14px;">
                Failure Probability
            </div>
            <div style="
                color:{risk_color};
                font-size:28px;
                font-weight:bold;
                font-family:monospace;
                letter-spacing:1px;
            ">
                {probability*100:06.2f} %
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("Machine RPM", f"{rpm}")

    with col3:
        st.metric("Torque", f"{torque} Nm")

    st.divider()

    # ---------------- GAUGE ----------------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Failure Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#FF4B4B"},
            'steps': [
                {'range': [0, 40], 'color': "#1F2937"},
                {'range': [40, 70], 'color': "#374151"},
                {'range': [70, 100], 'color': "#FF4B4B"}
            ],
        }
    ))

    gauge.update_layout(
        height=400,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white")
    )

    col4, col5 = st.columns([2, 1])

    with col4:
        st.plotly_chart(gauge, use_container_width=True)

    with col5:

        st.subheader("📊 System Status")

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Failure")
        else:
            st.success("✅ Machine Operating Normally")

        st.progress(int(probability * 100))

        st.write("### 📝 Machine Summary")
        st.info(f"""
        - Machine Type: {machine_type}
        - Tool Wear: {tool_wear} min
        - RPM: {rpm}
        - Torque: {torque} Nm
        """)

    st.divider()

    # ---------------- BAR CHART ----------------
    chart_data = pd.DataFrame({
        "Feature": ["Air Temp", "Process Temp", "RPM", "Torque", "Tool Wear"],
        "Value": [air_temp, process_temp, rpm, torque, tool_wear]
    })

    fig = px.bar(
        chart_data,
        x="Feature",
        y="Value",
        text="Value",
        title="📈 Machine Parameters Overview"
    )

    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white"),
        xaxis=dict(color="white", gridcolor="#333333"),
        yaxis=dict(color="white", gridcolor="#333333")
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.caption("Built with Streamlit + LightGBM + Plotly")
