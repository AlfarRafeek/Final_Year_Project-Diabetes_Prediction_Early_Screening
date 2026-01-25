import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================
# SETTINGS
# =========================
st.set_page_config(page_title="Diabetes Screening & Awareness", layout="centered")

MODEL_PATH = "best_pipe.pkl"
THRESHOLD = 0.4

st.title("🩺 Diabetes Risk Screening + Awareness (Sri Lanka)")
st.caption("Educational screening prototype (not a diagnosis).")


# =========================
# ✅ YOUR REAL FEATURE COLUMNS (from your Excel)
# =========================
FEATURE_COLUMNS = [
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


# =========================
# Load pipeline (ONLY ONCE)
# =========================
@st.cache_resource
def load_pipe():
    return joblib.load(MODEL_PATH)

try:
    pipe = load_pipe()
except Exception as e:
    st.error("❌ Could not load best_pipe.pkl")
    st.info("Make sure best_pipe.pkl is uploaded to GitHub in the same folder as streamlit_app.py")
    st.code(str(e))
    st.stop()


# =========================
# Feature importance
# =========================
def feature_importance_df(pipeline, top_k=12):
    if not hasattr(pipeline, "named_steps"):
        return None

    model = pipeline.named_steps.get("model")
    prep = pipeline.named_steps.get("preprocess")

    if model is None or prep is None:
        return None

    try:
        names = prep.get_feature_names_out()
    except Exception:
        names = None

    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_.ravel())
        if names is None:
            names = [f"feature_{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
        return df.head(top_k)

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        if names is None:
            names = [f"feature_{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
        return df.head(top_k)

    return None


# =========================
# PDF leaflet
# =========================
def make_leaflet_pdf(name, risk_prob, risk_label, tips_list):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Type 2 Diabetes – Awareness Leaflet")
    y -= 25

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Summary")
    y -= 18

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Name: {name if name else 'N/A'}")
    y -= 16
    c.drawString(50, y, f"Screening Risk: {risk_label}")
    y -= 16
    if risk_prob is not None:
        c.drawString(50, y, f"Estimated probability: {risk_prob:.2f}")
    else:
        c.drawString(50, y, "Estimated probability: N/A")
    y -= 24

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Personal Tips (Sri Lanka-friendly)")
    y -= 18

    c.setFont("Helvetica", 10)
    tips_list = tips_list or []
    for tip in tips_list[:12]:
        if y < 80:
            c.showPage()
            y = h - 60
            c.setFont("Helvetica", 10)
        c.drawString(50, y, f"• {tip}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Disclaimer: Educational screening tool only. Not a medical diagnosis.")
    c.save()

    pdf = buf.getvalue()
    buf.close()
    return pdf


# =========================
# Session state
# =========================
if "last_prob" not in st.session_state:
    st.session_state.last_prob = None
if "last_label" not in st.session_state:
    st.session_state.last_label = "N/A"
if "last_tips" not in st.session_state:
    st.session_state.last_tips = []
if "last_name" not in st.session_state:
    st.session_state.last_name = ""



def explain_risk(user_row):
    reasons = []

    # BMI
    bmi = user_row.get("BMI (kg/m²)")
    if bmi is not None and bmi > 25:
        reasons.append("BMI is above the healthy range (>25 kg/m²)")

    # Waist circumference
    waist = user_row.get("Waist circumference (cm)")
    if waist is not None and waist > 90:
        reasons.append("Waist circumference suggests central obesity")

    # Blood pressure
    sbp = user_row.get("Systolic BP (mmHg)")
    dbp = user_row.get("Diastolic BP (mmHg)")
    if sbp is not None and sbp >= 130:
        reasons.append("Systolic blood pressure is elevated")
    if dbp is not None and dbp >= 85:
        reasons.append("Diastolic blood pressure is elevated")

    # Lifestyle factors
    if user_row.get("How often do you exercise per week?") in ["Never", "1–2 days/week"]:
        reasons.append("Low level of physical activity")

    if user_row.get("Do you have a family history of diabetes?") == "Yes":
        reasons.append("Family history of diabetes increases risk")

    if user_row.get("Do you often feel unusually thirsty?") == "Yes":
        reasons.append("Presence of classic diabetes-related symptoms")

    if user_row.get("Do you experience frequent urination?") == "Yes":
        reasons.append("Frequent urination is a common early symptom")

    if not reasons:
        reasons.append("No strong individual risk factors detected from inputs")

    return reasons


# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Risk Prediction", "Why risk is high?", "Awareness Quiz", "Download Leaflet (PDF)"]
)


# =========================
# TAB 1: Prediction
# =========================

with tab1:
    st.subheader("Risk Prediction")
    st.write("Enter details. This is a **screening estimate (not a diagnosis)**.")

    name = st.text_input("Name (optional)", value=st.session_state.last_name)

    st.markdown("### Patient inputs")

    user_row = {}

    # IMPORTANT: Age/Height/Weight are categorical ranges in your data → use selectbox
    age_opts = ["Below 20", "20 - 29", "30 - 39", "40 - 49", "50 - 59", "60 and above"]
    height_opts = ["Below 150 cm", "150 - 159 cm", "160 - 169 cm", "170 - 179 cm", "180 cm and above"]
    weight_opts = ["Below 50 kg", "50 - 59 kg", "60 - 69 kg", "70 - 79 kg", "80 - 89 kg", "90 kg and above"]

    yes_no = ["No", "Yes"]
    gender_opts = ["Male", "Female", "Other"]
    exercise_opts = ["Never", "1–2 days/week", "3–5 days/week", "Almost daily"]
    diet_opts = ["Healthy", "Moderate", "Unhealthy"]
    sleep_opts = ["<6 hours", "6–7 hours", "7–8 hours", "8+ hours"]

    for col in FEATURE_COLUMNS:
        if col == "Age":
            user_row[col] = st.selectbox(col, age_opts)
        elif col == "Hight(cm)":
            user_row[col] = st.selectbox(col, height_opts)
        elif col == "Weight(kg)":
            user_row[col] = st.selectbox(col, weight_opts)

        elif col == "Gender":
            user_row[col] = st.selectbox(col, gender_opts)

        elif col == "How often do you exercise per week?":
            user_row[col] = st.selectbox(col, exercise_opts)

        elif col == "How would you describe your diet?":
            user_row[col] = st.selectbox(col, diet_opts)

        elif col == "Average sleep hours per night":
            user_row[col] = st.selectbox(col, sleep_opts)

        elif col in ["Waist circumference (cm)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)", "BMI (kg/m²)"]:
            user_row[col] = st.number_input(col, value=0.0)

        else:
            user_row[col] = st.selectbox(col, yes_no)

    if st.button("Predict"):
        st.session_state.last_name = name
        st.session_state.last_input = user_row
        try:
            X_input = pd.DataFrame([user_row])
            prob = float(pipe.predict_proba(X_input)[:, 1][0])

            st.session_state.last_prob = prob

            if prob >= THRESHOLD:
                st.session_state.last_label = "Higher Risk"
                st.error(f"Estimated probability: {prob:.2f} → Higher Risk (threshold={THRESHOLD})")
            else:
                st.session_state.last_label = "Lower Risk"
                st.success(f"Estimated probability: {prob:.2f} → Lower Risk (threshold={THRESHOLD})")

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))
            st.warning("This usually happens if the deployed best_pipe.pkl was trained with different column names.")


# =========================
# TAB 2: Feature importance
# =========================
with tab2:
    st.subheader("Why risk is high?")
    st.write("Explanation based on your inputs and established health guidelines.")

    if "last_input" not in st.session_state:
        st.info("Please run a risk prediction first.")
    else:
        reasons = explain_risk(st.session_state.last_input)

        for r in reasons:
            st.write("•", r)

        st.caption(
            "Explanation is based on known clinical risk factors and model findings "
            "reported during evaluation. This improves transparency for users."
        )


# =========================
# TAB 3: Awareness quiz
# =========================
with tab3:
    st.subheader("Awareness Quiz (10 questions)")
    st.write(
        "Answer honestly. You’ll receive personalised diabetes-prevention tips "
        "based on lifestyle and health awareness."
    )

    # -------------------------
    # Questions
    # -------------------------
    q1 = st.radio("1) How often do you exercise (≥30 mins)?",
                  ["Rarely", "1–2 days/week", "3–5 days/week", "Almost daily"])

    q2 = st.radio("2) Sweet tea / sugary drinks per day?",
                  ["0", "1", "2", "3 or more"])

    q3 = st.radio("3) Your usual rice portion?",
                  ["Small", "Medium", "Large"])

    q4 = st.radio("4) Family history of diabetes?",
                  ["No", "Yes"])

    q5 = st.radio("5) Average sleep per night?",
                  ["<6 hours", "6–7 hours", "7–8 hours", "8+ hours"])

    q6 = st.radio("6) How often do you eat fried foods?",
                  ["Rarely", "1–2 times/week", "3–4 times/week", "Almost daily"])

    q7 = st.radio("7) Fruits & vegetables intake per day?",
                  ["<2 portions", "2–3 portions", "4–5 portions", "More than 5"])

    q8 = st.radio("8) How often do you check blood sugar?",
                  ["Never", "Only when sick", "Once a year", "Regularly"])

    q9 = st.radio("9) How would you describe your stress level?",
                  ["Low", "Moderate", "High"])

    q10 = st.radio("10) Sitting time per day?",
                   ["<4 hours", "4–6 hours", "6–8 hours", "More than 8 hours"])

    # -------------------------
    # Generate tips
    # -------------------------
    if st.button("Get my awareness tips"):
        tips = []

        if q1 in ["Rarely", "1–2 days/week"]:
            tips.append("Increase physical activity: walking, cycling, or home exercises at least 30 mins/day.")

        if q2 in ["2", "3 or more"]:
            tips.append("Reduce sugary drinks; switch to plain tea or water.")

        if q3 == "Large":
            tips.append("Reduce rice portion; increase vegetables like gotukola, mukunuwenna, beans, cabbage.")

        if q4 == "Yes":
            tips.append("Family history increases risk—do regular screening (FBS / HbA1c).")

        if q5 == "<6 hours":
            tips.append("Improve sleep duration to 7–8 hours to reduce insulin resistance.")

        if q6 in ["3–4 times/week", "Almost daily"]:
            tips.append("Limit fried foods; choose boiled, steamed, or grilled options.")

        if q7 in ["<2 portions", "2–3 portions"]:
            tips.append("Increase fruits and vegetables to at least 4–5 portions daily.")

        if q8 in ["Never", "Only when sick"]:
            tips.append("Check blood sugar periodically, especially if risk factors are present.")

        if q9 == "High":
            tips.append("Manage stress through relaxation, prayer, breathing exercises, or walking.")

        if q10 in ["6–8 hours", "More than 8 hours"]:
            tips.append("Reduce prolonged sitting—stand or walk for 5 minutes every hour.")

        tips.append("Maintain a healthy waist circumference and blood pressure.")
        tips.append("Avoid smoking and excessive alcohol consumption.")

        st.session_state.last_tips = tips

        st.markdown("### Your Personal Awareness Tips")
        for t in tips:
            st.write("•", t)



# =========================
# TAB 4: PDF leaflet
# =========================
with tab4:
    st.subheader("Download Leaflet (PDF)")

    leaflet_name = st.text_input("Name on leaflet", value=st.session_state.last_name)

    if st.button("Generate PDF"):
        pdf_bytes = make_leaflet_pdf(
            name=leaflet_name,
            risk_prob=st.session_state.last_prob,
            risk_label=st.session_state.last_label,
            tips_list=st.session_state.last_tips
        )

        st.download_button(
            label="⬇️ Download PDF leaflet",
            data=pdf_bytes,
            file_name="diabetes_awareness_leaflet.pdf",
            mime="application/pdf"
        )

st.divider()
st.caption("⚠️ Disclaimer: Educational screening tool only. Not a medical diagnosis.")
