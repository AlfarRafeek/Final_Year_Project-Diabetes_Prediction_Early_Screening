import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# -------- App config --------
st.set_page_config(page_title="Diabetes Screening & Awareness", layout="centered")

MODEL_PATH = "best_pipe.pkl"
THRESHOLD = 0.40

FEATURES = [
    "Age",
    "Gender",
    "Hight(cm)",
    "Weight(kg)",
    "Do you have a family history of diabetes?",
    "How often do you exercise per week?",
    "How would you describe your diet?",
    "Have you ever been diagnosed with high blood pressure or cholesterol?",
    "Do you experience frequent urination?",
    "Do you often feel unusually thirsty?",
    "Have you noticed unexplained weight loss or gain?",
    "Do you feel unusually fatigued or tired?",
    "Do you have blurred vision or slow-healing wounds?",
    "Occupation",
    "Average sleep hours per night",
    "Waist circumference (cm)",
    "Systolic BP (mmHg)",
    "Diastolic BP (mmHg)",
    "BMI (kg/m²)",
]

AGE_OPTS = ["Below 20", "20 - 29", "30 - 39", "40 - 49", "50 - 59", "60 and above"]
HEIGHT_OPTS = ["Below 150 cm", "150 - 159 cm", "160 - 169 cm", "170 - 179 cm", "180 cm and above"]
WEIGHT_OPTS = ["Below 50 kg", "50 - 59 kg", "60 - 69 kg", "70 - 79 kg", "80 - 89 kg", "90 kg and above"]

YES_NO = ["No", "Yes"]
GENDER_OPTS = ["Male", "Female", "Other"]
EXERCISE_OPTS = ["Never", "1–2 days/week", "3–5 days/week", "Almost daily"]
DIET_OPTS = ["Healthy", "Moderate", "Unhealthy"]
SLEEP_OPTS = ["<6 hours", "6–7 hours", "7–8 hours", "8+ hours"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def init_state():
    st.session_state.setdefault("name", "")
    st.session_state.setdefault("last_prob", None)
    st.session_state.setdefault("last_label", "N/A")
    st.session_state.setdefault("last_input", None)
    st.session_state.setdefault("tips", [])


def build_input_form():
    """Collects user input and returns dict with exactly the same keys as training features."""
    row = {}
    for col in FEATURES:
        if col == "Age":
            row[col] = st.selectbox(col, AGE_OPTS)
        elif col == "Hight(cm)":
            row[col] = st.selectbox(col, HEIGHT_OPTS)
        elif col == "Weight(kg)":
            row[col] = st.selectbox(col, WEIGHT_OPTS)
        elif col == "Gender":
            row[col] = st.selectbox(col, GENDER_OPTS)
        elif col == "How often do you exercise per week?":
            row[col] = st.selectbox(col, EXERCISE_OPTS)
        elif col == "How would you describe your diet?":
            row[col] = st.selectbox(col, DIET_OPTS)
        elif col == "Average sleep hours per night":
            row[col] = st.selectbox(col, SLEEP_OPTS)
        elif col in ["Waist circumference (cm)", "BMI (kg/m²)"]:
            row[col] = st.number_input(col, value=0.0, step=0.1)
        elif col in ["Systolic BP (mmHg)", "Diastolic BP (mmHg)"]:
            row[col] = st.number_input(col, value=0.0, step=1.0)
        else:
            row[col] = st.selectbox(col, YES_NO)
    return row


def explain_risk(row):
    """Simple explanation using common screening thresholds + symptoms."""
    reasons = []

    bmi = row.get("BMI (kg/m²)")
    if isinstance(bmi, (int, float)) and bmi > 25:
        reasons.append("BMI is above the healthy range (> 25 kg/m²).")

    waist = row.get("Waist circumference (cm)")
    if isinstance(waist, (int, float)) and waist > 90:
        reasons.append("Waist circumference suggests central obesity (high belly fat).")

    sbp = row.get("Systolic BP (mmHg)")
    dbp = row.get("Diastolic BP (mmHg)")
    if isinstance(sbp, (int, float)) and sbp >= 130:
        reasons.append("Systolic blood pressure is elevated (≥ 130).")
    if isinstance(dbp, (int, float)) and dbp >= 85:
        reasons.append("Diastolic blood pressure is elevated (≥ 85).")

    if row.get("How often do you exercise per week?") in ["Never", "1–2 days/week"]:
        reasons.append("Low physical activity can increase diabetes risk.")

    if row.get("Do you have a family history of diabetes?") == "Yes":
        reasons.append("Family history increases the chance of developing diabetes.")

    if row.get("Do you often feel unusually thirsty?") == "Yes":
        reasons.append("Thirst can be an early warning symptom.")
    if row.get("Do you experience frequent urination?") == "Yes":
        reasons.append("Frequent urination can be an early warning symptom.")

    if not reasons:
        reasons.append("No strong risk factors detected from the entered values.")

    return reasons


def make_pdf(name, prob, label, tips):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Type 2 Diabetes – Awareness Leaflet")
    y -= 24

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Screening summary")
    y -= 16

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Name: {name if name else 'N/A'}")
    y -= 14
    c.drawString(50, y, f"Risk category: {label}")
    y -= 14
    c.drawString(50, y, f"Estimated probability: {prob:.2f}" if prob is not None else "Estimated probability: N/A")
    y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Personal tips (Sri Lanka-friendly)")
    y -= 16

    c.setFont("Helvetica", 10)
    for t in (tips or [])[:12]:
        if y < 80:
            c.showPage()
            y = h - 60
            c.setFont("Helvetica", 10)
        c.drawString(50, y, f"• {t}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Disclaimer: Educational screening tool only. Not a medical diagnosis.")
    c.save()

    pdf = buf.getvalue()
    buf.close()
    return pdf


# -------- Main UI --------
init_state()

st.title(":blood: Diabetes Risk Screening + Awareness (Sri Lanka)")
st.caption("Educational screening prototype (not a diagnosis).")

try:
    model = load_model()
except Exception as e:
    st.error("Could not load the trained model file (best_pipe.pkl).")
    st.code(str(e))
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Risk Prediction", "Why risk is high?", "Awareness Quiz", "Download Leaflet (PDF)"])

with tab1:
    st.subheader("Risk Prediction")
    st.session_state["name"] = st.text_input("Name (optional)", value=st.session_state["name"])

    st.write("Enter patient details:")
    row = build_input_form()

    if st.button("Predict risk"):
        st.session_state["last_input"] = row

        X = pd.DataFrame([row])
        prob = float(model.predict_proba(X)[:, 1][0])

        st.session_state["last_prob"] = prob
        if prob >= THRESHOLD:
            st.session_state["last_label"] = "Higher Risk"
            st.error(f"Estimated probability: {prob:.2f} → Higher Risk (threshold={THRESHOLD})")
        else:
            st.session_state["last_label"] = "Lower Risk"
            st.success(f"Estimated probability: {prob:.2f} → Lower Risk (threshold={THRESHOLD})")

        st.info("For medical confirmation, consider FBS / HbA1c tests and consult a doctor.")

with tab2:
    st.subheader("Why risk is high?")
    if st.session_state["last_input"] is None:
        st.info("Run a prediction first to see the explanation.")
    else:
        for r in explain_risk(st.session_state["last_input"]):
            st.write("•", r)

with tab3:
    st.subheader("Awareness Quiz (10 questions)")
    st.write("This quiz gives general lifestyle guidance (not a diagnosis).")

    q1 = st.radio("1) Exercise (≥30 mins)?", ["Rarely", "1–2 days/week", "3–5 days/week", "Almost daily"])
    q2 = st.radio("2) Sugary drinks per day?", ["0", "1", "2", "3 or more"])
    q3 = st.radio("3) Rice portion?", ["Small", "Medium", "Large"])
    q4 = st.radio("4) Family history?", ["No", "Yes"])
    q5 = st.radio("5) Sleep?", ["<6 hours", "6–7 hours", "7–8 hours", "8+ hours"])
    q6 = st.radio("6) Fried foods?", ["Rarely", "1–2 times/week", "3–4 times/week", "Almost daily"])
    q7 = st.radio("7) Fruits & vegetables daily?", ["<2 portions", "2–3 portions", "4–5 portions", "More than 5"])
    q8 = st.radio("8) Blood sugar check?", ["Never", "Only when sick", "Once a year", "Regularly"])
    q9 = st.radio("9) Stress level?", ["Low", "Moderate", "High"])
    q10 = st.radio("10) Sitting time?", ["<4 hours", "4–6 hours", "6–8 hours", "More than 8 hours"])

    if st.button("Get my tips"):
        tips = []

        if q1 in ["Rarely", "1–2 days/week"]:
            tips.append("Increase activity: 30 minutes walking on most days.")
        if q2 in ["2", "3 or more"]:
            tips.append("Reduce sweet tea/soft drinks gradually; drink more water.")
        if q3 == "Large":
            tips.append("Reduce rice portion; add vegetables (gotukola, mukunuwenna, beans).")
        if q4 == "Yes":
            tips.append("Family history increases risk: do regular screening (FBS/HbA1c).")
        if q5 == "<6 hours":
            tips.append("Try to sleep 7–8 hours to support healthy metabolism.")
        if q6 in ["3–4 times/week", "Almost daily"]:
            tips.append("Limit fried foods; try boiled/steamed/grilled meals.")
        if q7 in ["<2 portions", "2–3 portions"]:
            tips.append("Add more vegetables and fruit daily (aim for 4–5 portions).")
        if q8 in ["Never", "Only when sick"]:
            tips.append("Check blood sugar occasionally, especially if you have risk factors.")
        if q9 == "High":
            tips.append("Manage stress (walk, breathing, relaxation, prayer).")
        if q10 in ["6–8 hours", "More than 8 hours"]:
            tips.append("Break long sitting: stand/walk 5 mins each hour.")

        tips.append("Maintain healthy waist circumference and blood pressure.")
        st.session_state["tips"] = tips

        st.markdown("### Your tips")
        for t in tips:
            st.write("•", t)

with tab4:
    st.subheader("Download Leaflet (PDF)")
    if st.button("Generate PDF"):
        pdf = make_pdf(
            st.session_state["name"],
            st.session_state["last_prob"],
            st.session_state["last_label"],
            st.session_state["tips"],
        )
        st.download_button(
            ":arrow_down: Download PDF leaflet",
            data=pdf,
            file_name="diabetes_awareness_leaflet.pdf",
            mime="application/pdf",
        )

st.divider()
st.caption(":warning: Disclaimer: Educational screening tool only. Not a medical diagnosis.")
