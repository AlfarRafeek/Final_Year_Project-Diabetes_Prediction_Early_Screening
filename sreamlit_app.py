import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime

# --- PDF (leaflet) ---
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


st.set_page_config(page_title="Diabetes Screening & Awareness", layout="centered")

# =========================
# Helpers
# =========================
@st.cache_resource
def load_model():
    with open("best_pipe.pkl", "rb") as f:
        return pickle.load(f)

def make_leaflet_pdf(name, risk_prob, risk_label, tips_list):
    """Create a simple one-page PDF leaflet and return bytes."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 60
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
    c.drawString(50, y, f"Screening Risk (tool estimate): {risk_label}")
    y -= 16
    c.drawString(50, y, f"Estimated probability: {risk_prob:.2f}" if risk_prob is not None else "Estimated probability: N/A")
    y -= 24

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Quick Awareness")
    y -= 18
    c.setFont("Helvetica", 10)
    lines = [
        "Type 2 diabetes happens when the body cannot use insulin properly.",
        "High blood sugar over time can damage the heart, kidneys, eyes and nerves.",
        "Early screening helps prevent complications."
    ]
    for line in lines:
        c.drawString(50, y, f"- {line}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Personal Tips (Sri Lanka-friendly)")
    y -= 18
    c.setFont("Helvetica", 10)
    for tip in tips_list[:10]:
        if y < 80:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)
        c.drawString(50, y, f"• {tip}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "When to check blood sugar (FBS / HbA1c)")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(50, y, "If you have frequent urination, thirst, fatigue, blurred vision or slow-healing wounds, visit a clinic.")
    y -= 30

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Disclaimer: Educational screening tool only. Not a medical diagnosis.")
    c.save()

    pdf = buffer.getvalue()
    buffer.close()
    return pdf


def extract_feature_importance(pipe, top_k=12):
    """
    Returns a DataFrame with top feature importances (works for:
    - LogisticRegression: absolute coefficients
    - Tree models: feature_importances_
    If it can’t compute, returns None.
    """
    if not hasattr(pipe, "named_steps"):
        return None

    # common names in your pipeline: ("prep"/"preprocess") + ("model")
    prep = pipe.named_steps.get("prep") or pipe.named_steps.get("preprocess")
    model = pipe.named_steps.get("model")

    if prep is None or model is None:
        return None

    try:
        names = prep.get_feature_names_out()
    except Exception:
        # fallback if names not available
        names = None

    # Logistic Regression
    if hasattr(model, "coef_"):
        coefs = model.coef_.ravel()
        imp = np.abs(coefs)
        if names is None:
            names = [f"feature_{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
        return df.head(top_k)

    # Tree models
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        if names is None:
            names = [f"feature_{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
        return df.head(top_k)

    return None


# =========================
# Load model
# =========================
pipe = load_model()

st.title("🩺 Diabetes Risk Screening + Awareness (Sri Lanka)")

tab1, tab2, tab3, tab4 = st.tabs(["Risk Prediction", "Why risk is high?", "Awareness Quiz", "Download Leaflet (PDF)"])

# We'll store last prediction in session state
if "last_prob" not in st.session_state:
    st.session_state.last_prob = None
if "last_label" not in st.session_state:
    st.session_state.last_label = "N/A"
if "last_tips" not in st.session_state:
    st.session_state.last_tips = []


# =========================
# TAB 1: Prediction
# =========================
with tab1:
    st.header("Risk Prediction")

    st.write("Enter details. This is a **screening estimate** (not a diagnosis).")

    # --- Keep inputs simple (customize to your dataset columns later if needed) ---
    name = st.text_input("Name (optional)")

    # Basic numeric inputs (you can add more)
    age = st.number_input("Age", 18, 100, 40)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    sbp = st.number_input("Systolic BP (mmHg)", 80, 220, 120)
    dbp = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)

    # Your tuned threshold from results
    threshold = 0.4

    # IMPORTANT:
    # This basic input works only if your model was trained on exactly these 4 columns.
    # If your model uses many columns, you should build a full input form matching training columns.
    # For now, we try simple usage and catch errors.
    if st.button("Predict"):
        try:
            X_input = pd.DataFrame({
                "Age": age,
                "BMI (kg/m²)": bmi,
                "Systolic BP (mmHg)": sbp,
                "Diastolic BP (mmHg)": dbp
            }, index=[0]) # Add index=[0] to create a DataFrame with one row

            prob = float(pipe.predict_proba(X_input)[:, 1][0])
            st.session_state.last_prob = prob

            if prob >= threshold:
                st.session_state.last_label = "Higher Risk"
                st.error(f"Estimated probability: {prob:.2f} → Higher Risk (threshold={threshold})")
            else:
                st.session_state.last_label = "Lower Risk"
                st.success(f"Estimated probability: {prob:.2f} → Lower Risk (threshold={threshold})")

            st.info(
                "If you have symptoms (frequent urination, thirst, fatigue, blurred vision, slow healing), "
                "please visit a clinic and check FBS/HbA1c."
            )
        except Exception as e:
            st.warning(
                "Your trained model likely expects more input columns (not only Age/BMI/BP). "
                "If you want, send me your final training feature list and I’ll update the app form to match exactly."
            )
            st.code(str(e))


# =========================
# TAB 2: Why risk is high? (Feature importance)
# =========================
with tab2:
    st.header("Why risk is high? (Feature Importance)")

    st.write("This graph helps explain which features influence the model most.")

    imp_df = extract_feature_importance(pipe, top_k=12)

    if imp_df is None or imp_df.empty:
        st.warning("Feature importance not available for this saved model/pipeline.")
    else:
        # Plot top features
        fig = plt.figure()
        plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
        plt.title("Top Features (importance)")
        plt.xlabel("Importance (abs coef or tree importance)")
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "Note: For Logistic Regression, importance = |coefficient| after preprocessing. "
            "For tree models, importance = built-in feature importance."
        )


# =========================
# TAB 3: Awareness Quiz (5 questions + tips)
# =========================
with tab3:
    st.header("Awareness Quiz (5 Questions)")
    st.write("Answer quickly — you’ll get personalised tips (Sri Lanka-friendly).")

    q1 = st.radio("1) How often do you exercise (at least 30 mins)?", ["Rarely", "1–2 days/week", "3–5 days/week", "Almost daily"])
    q2 = st.radio("2) How many sugary drinks / sweet tea per day?", ["0", "1", "2", "3 or more"])
    q3 = st.radio("3) Your usual rice portion at lunch/dinner?", ["Small", "Medium", "Large"])
    q4 = st.radio("4) Do you have close family history of diabetes?", ["No", "Yes"])
    q5 = st.radio("5) How is your sleep most nights?", ["<6 hours", "6–7 hours", "7–8 hours", "8+ hours"])

    if st.button("Get my tips"):
        tips = []

        # Exercise tips
        if q1 in ["Rarely", "1–2 days/week"]:
            tips.append("Try a 30-minute walk after dinner (even 5 days/week helps insulin sensitivity).")
            tips.append("If you’re busy, do 10 mins walk × 3 times/day.")

        # Sugary drink tips
        if q2 in ["1", "2", "3 or more"]:
            tips.append("Reduce sweet tea / soft drinks. Try plain tea, or reduce sugar gradually (half → quarter → none).")
            tips.append("Replace sugary drinks with water, king coconut (without added sugar), or unsweetened herbal drinks.")

        # Rice tips (Sri Lanka)
        if q3 == "Large":
            tips.append("Reduce rice portion slightly and increase vegetables (gotukola, mukunuwenna, beans, cabbage).")
            tips.append("Try replacing part of white rice with brown/red rice or kurakkan occasionally (portion control still matters).")

        # Family history
        if q4 == "Yes":
            tips.append("Because of family history, do regular screening (FBS/HbA1c) and keep weight/waist in healthy range.")

        # Sleep
        if q5 == "<6 hours":
            tips.append("Aim for 7–8 hours sleep. Poor sleep increases insulin resistance and cravings.")

        # Always useful tips
        tips.append("Add protein and fibre to meals: dhal, chickpeas, eggs, fish, chicken, leafy salads.")
        tips.append("Snack idea: roasted gram (kadala), nuts (small portion), yogurt without sugar, fruit (portion control).")
        tips.append("If you smoke, reducing/stopping helps blood vessels and metabolic health.")
        tips.append("If BP is high, reduce salt + processed foods and follow medical advice.")

        st.session_state.last_tips = tips

        st.subheader("Your Tips")
        for t in tips:
            st.write("•", t)

        st.info("This quiz is for awareness. For diagnosis, please check blood sugar with a healthcare professional.")


# =========================
# TAB 4: Download leaflet PDF
# =========================
with tab4:
    st.header("Download PDF Leaflet")

    st.write("This will generate a one-page PDF you can share for awareness.")

    leaflet_name = st.text_input("Name to show on leaflet (optional)", key="leaflet_name")

    prob = st.session_state.last_prob
    label = st.session_state.last_label
    tips = st.session_state.last_tips

    if st.button("Generate PDF"):
        pdf_bytes = make_leaflet_pdf(leaflet_name, prob, label, tips)
        st.download_button(
            label="⬇️ Download Leaflet PDF",
            data=pdf_bytes,
            file_name="diabetes_awareness_leaflet.pdf",
            mime="application/pdf"
        )

    st.caption("Tip: Run a prediction and quiz first so the leaflet contains personalised info.")
