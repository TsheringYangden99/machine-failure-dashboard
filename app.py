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

# ---------------- SAFE MODEL LOADING ----------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except Exception as e:
        st.error("❌ Failed to load model.pkl")
        st.exception(e)
        return None

model = load_model()

if model is None:
    st.stop()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main { background-color: #0E1117; }

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    background-color: #FF4B4B;
    color: white;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #ff2e2e;
}

.title {
    text-align: center;
    color: white;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #B0B0B0;
    font-size: 18px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>⚙️ AI Machine Failure Prediction Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-Time Industrial Equipment Monitoring</div>", unsafe_allow_html=True)

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("⚙️ Machine Inputs")

machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])

air_temp = st.sidebar.slider("Air Temperature (K)", 290.0, 320.0, 298.0)
process_temp = st.sidebar.slider("Process Temperature (K)", 300.0, 340.0, 308.0)
rpm = st.sidebar.slider("Rotational Speed (RPM)", 1000.0, 3000.0, 1500.0)
torque = st.sidebar.slider("Torque (Nm)", 0.0, 100.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0.0, 300.0, 10.0)

# Encoding
type_M = 1 if machine_type == "M" else 0
type_L = 1 if machine_type == "L" else 0

# ---------------- PREDICTION ----------------
if st.sidebar.button("🔮 Predict Failure"):

    input_data = np.array([[
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear,
        type_M,
        type_L
    ]])

    try:
        with st.spinner("Predicting Machine Failure..."):
            prediction = model.predict(input_data)

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_data)[0][1]
            else:
                probability = 0.0

        st.toast("✅ Prediction Completed")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)
        st.stop()

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Failure Probability", f"{probability*100:.2f}%")
    col2.metric("Machine RPM", rpm)
    col3.metric("Torque", f"{torque} Nm")

    st.divider()

    # ---------------- GAUGE ----------------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Failure Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'bar': {'color': "red"}
        }
    ))

    gauge.update_layout(
        height=400,
        paper_bgcolor="#0E1117",
        font={'color': "white"}
    )

    col4, col5 = st.columns([2, 1])

    with col4:
        st.plotly_chart(gauge, use_container_width=True)

    with col5:
        if prediction[0] == 1:
            st.error("⚠️ High Risk of Failure")
        else:
            st.success("✅ Machine Operating Normally")

        st.progress(min(int(probability * 100), 100))

        st.info(f"""
        Machine Type: {machine_type}
        Tool Wear: {tool_wear}
        RPM: {rpm}
        Torque: {torque}
        """)

    # ---------------- CHART ----------------
    chart_data = pd.DataFrame({
        "Feature": ["Air Temp", "Process Temp", "RPM", "Torque", "Tool Wear"],
        "Value": [air_temp, process_temp, rpm, torque, tool_wear]
    })

    fig = px.bar(chart_data, x="Feature", y="Value", text="Value")

    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.caption("Built with Streamlit + Machine Learning + Plotly")