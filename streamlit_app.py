import warnings
warnings.filterwarnings("ignore")

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from fpdf import FPDF


# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(page_title="Diabetes Risk Analytics Dashboard (Sri Lanka)", layout="wide")
st.title("📊 Diabetes Risk Analytics Dashboard (Sri Lanka)")
st.caption("Prototype for early screening + analytics (Educational tool — not a diagnosis).")


# -----------------------------
# CONSTANTS (your dataset columns)
# -----------------------------
TARGET_COL = "Have you ever been diagnosed with diabetes?"
LEAKAGE_COL = "Are you currently taking any medications for diabetes or related conditions?"
DROP_COLS_ALWAYS = ["Timestamp"]  # safe to drop

RANDOM_STATE = 42
THRESHOLD = 0.40  # you can tune later


# -----------------------------
# HELPERS
# -----------------------------
def load_excel(file_like) -> pd.DataFrame:
    df0 = pd.read_excel(file_like)
    df0.columns = [c.strip() for c in df0.columns]
    return df0


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # drop timestamp if present
    for c in DROP_COLS_ALWAYS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # clean target
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df = df[df[TARGET_COL].isin(["Yes", "No"])].copy()
    df[TARGET_COL] = df[TARGET_COL].map({"No": 0, "Yes": 1}).astype(int)

    # numeric conversions
    for col in ["Waist circumference (cm)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)", "BMI (kg/m²)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return pre


def metric_row(y_true, proba, pred):
    return {
        "ROC-AUC": roc_auc_score(y_true, proba),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }


def make_pdf_leaflet(name, prob, label, tips):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Type 2 Diabetes - Awareness Leaflet (Sri Lanka)", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, f"Name: {name if name else 'N/A'}")
    pdf.multi_cell(0, 7, f"Screening result: {label}")
    if prob is not None:
        pdf.multi_cell(0, 7, f"Estimated probability: {prob:.2f}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Quick Awareness", ln=True)
    pdf.set_font("Helvetica", "", 10)
    bullets = [
        "Type 2 diabetes happens when the body becomes resistant to insulin.",
        "High blood sugar over time can damage heart, kidneys, eyes and nerves.",
        "Early screening + lifestyle changes can reduce complications."
    ]
    for b in bullets:
        pdf.multi_cell(0, 6, f"- {b}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Personal Tips (Sri Lanka-friendly)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    tips = tips or []
    for t in tips[:14]:
        pdf.multi_cell(0, 6, f"• {t}")

    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 6, "Disclaimer: Educational screening only. Not a medical diagnosis.")

    out = io.BytesIO()
    pdf.output(out)
    return out.getvalue()


def explain_risk_rules(user_row: dict):
    reasons = []

    # waist / BMI / BP rules (simple clinical-style explanations)
    bmi = user_row.get("BMI (kg/m²)")
    if bmi is not None and isinstance(bmi, (int, float)) and bmi > 25:
        reasons.append("BMI is above the healthy range (>25 kg/m²).")

    waist = user_row.get("Waist circumference (cm)")
    if waist is not None and isinstance(waist, (int, float)) and waist > 90:
        reasons.append("Waist circumference is high (central obesity risk).")

    sbp = user_row.get("Systolic BP (mmHg)")
    dbp = user_row.get("Diastolic BP (mmHg)")
    if sbp is not None and isinstance(sbp, (int, float)) and sbp >= 130:
        reasons.append("Systolic BP is elevated (>=130 mmHg).")
    if dbp is not None and isinstance(dbp, (int, float)) and dbp >= 85:
        reasons.append("Diastolic BP is elevated (>=85 mmHg).")

    if user_row.get("Do you have a family history of diabetes?") == "Yes":
        reasons.append("Family history increases risk.")

    if user_row.get("How often do you exercise per week?") in ["Never"]:
        reasons.append("No weekly exercise reported (low physical activity).")

    # symptoms
    for sym in [
        "Do you experience frequent urination?",
        "Do you often feel unusually thirsty?",
        "Do you feel unusually fatigued or tired?",
        "Do you have blurred vision or slow-healing wounds?",
    ]:
        if user_row.get(sym) == "Yes":
            reasons.append(f"Reported symptom: {sym}")

    if not reasons:
        reasons.append("No strong risk signals detected from the entered inputs.")

    return reasons


# -----------------------------
# SIDEBAR: DATA INPUT
# -----------------------------
st.sidebar.header("1) Data Source")
uploaded = st.sidebar.file_uploader("Upload your Excel dataset (.xlsx)", type=["xlsx"])

use_sample = st.sidebar.checkbox("Use bundled sample (only if you committed the Excel to repo)", value=False)
sample_path = "Diabetes Risk Survey (Responses) (1).xlsx"  # optional if you commit dataset (usually you won't)

df_raw = None

if uploaded is not None:
    df_raw = load_excel(uploaded)
elif use_sample:
    try:
        df_raw = load_excel(sample_path)
    except Exception:
        st.sidebar.error("Sample Excel not found in repo. Upload the dataset instead.")

if df_raw is None:
    st.info("Upload your dataset on the left sidebar to start.")
    st.stop()

df = clean_dataset(df_raw)

st.sidebar.success(f"Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")


# -----------------------------
# OPTIONAL: DROP LEAKAGE
# -----------------------------
st.sidebar.header("2) Leakage Control")
drop_leakage = st.sidebar.checkbox("Drop leakage column (recommended)", value=True)
drop_symptoms = st.sidebar.checkbox("Drop symptom questions (harder early-screening)", value=False)

symptom_cols = [
    "Do you experience frequent urination?",
    "Do you often feel unusually thirsty?",
    "Have you noticed unexplained weight loss or gain?",
    "Do you feel unusually fatigued or tired?",
    "Do you have blurred vision or slow-healing wounds?",
]

drop_cols = []
if drop_leakage and LEAKAGE_COL in df.columns:
    drop_cols.append(LEAKAGE_COL)
if drop_symptoms:
    drop_cols += [c for c in symptom_cols if c in df.columns]

df_model = df.drop(columns=drop_cols, errors="ignore")

# build X/y
X = df_model.drop(columns=[TARGET_COL])
y = df_model[TARGET_COL].astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)


# -----------------------------
# DASHBOARD TABS
# -----------------------------
tab_overview, tab_eda, tab_models, tab_explain, tab_awareness = st.tabs(
    ["Overview", "EDA (Visual Analytics)", "Model Training & Comparison", "Significant Factors", "Awareness + Leaflet"]
)


# -----------------------------
# TAB: OVERVIEW
# -----------------------------
with tab_overview:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)

    counts = y.value_counts().to_dict()
    c1.metric("Total responses", df_model.shape[0])
    c2.metric("Non-diabetes (0)", int(counts.get(0, 0)))
    c3.metric("Diabetes (1)", int(counts.get(1, 0)))

    st.write("**Features used (after drops):**")
    st.code("\n".join(list(X.columns)), language="text")

    st.write("**Dropped columns (leakage / optional):**")
    st.code(", ".join(drop_cols) if drop_cols else "None", language="text")

    st.write("Preview:")
    st.dataframe(df_model.head(10), use_container_width=True)


# -----------------------------
# TAB: EDA
# -----------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    # Class distribution
    st.markdown("### Class Distribution")
    fig = plt.figure()
    cls = y.value_counts().sort_index()
    plt.bar(["0 (No)", "1 (Yes)"], [cls.get(0, 0), cls.get(1, 0)])
    plt.title("Class Distribution")
    plt.ylabel("Count")
    st.pyplot(fig)

    # Missing values
    st.markdown("### Missing Values (Top Columns)")
    miss = df_model.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0].head(15)
    if len(miss) == 0:
        st.success("No missing values detected.")
    else:
        fig2 = plt.figure()
        plt.bar(miss.index.astype(str), miss.values)
        plt.xticks(rotation=70, ha="right")
        plt.title("Top Missing Columns")
        plt.tight_layout()
        st.pyplot(fig2)

    # Numeric distributions
    st.markdown("### Numeric Distributions")
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    pick = st.multiselect("Choose numeric columns to plot:", num_cols, default=num_cols[:3])
    for col in pick:
        fig3 = plt.figure()
        plt.hist(df_model[col].dropna(), bins=20)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        st.pyplot(fig3)

    st.markdown("### Simple Correlation (Numeric only)")
    if len(num_cols) >= 2:
        corr = df_model[num_cols].corr(numeric_only=True)
        fig4 = plt.figure()
        plt.imshow(corr.values)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.colorbar()
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig4)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")


# -----------------------------
# TAB: MODELS
# -----------------------------
with tab_models:
    st.subheader("Model Training & Comparison (SMOTE + Pipelines)")

    st.write(
        "This section trains several classifiers using a leakage-safe pipeline:\n"
        "**Preprocess → SMOTE (train only) → Model**.\n\n"
        "Tip: Don’t report only accuracy. Use ROC-AUC, Recall, and F1 as well."
    )

    # models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=600, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    pre = build_preprocessor(X_train)
    smote = SMOTE(sampling_strategy=0.9, k_neighbors=3, random_state=RANDOM_STATE)

    def make_pipe(m):
        return ImbPipeline([
            ("preprocess", pre),
            ("smote", smote),
            ("model", m)
        ])

    if "trained" not in st.session_state:
        st.session_state.trained = False
    if "best_name" not in st.session_state:
        st.session_state.best_name = None
    if "best_pipe" not in st.session_state:
        st.session_state.best_pipe = None
    if "model_table" not in st.session_state:
        st.session_state.model_table = None

    if st.button("🚀 Train & Compare Models"):
        rows = []
        fitted = {}

        for name, m in models.items():
            pipe = make_pipe(m)
            pipe.fit(X_train, y_train)

            proba = pipe.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            rows.append({"Model": name, **metric_row(y_test, proba, pred)})
            fitted[name] = pipe

        table = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
        st.session_state.model_table = table

        best_name = table.loc[0, "Model"]
        st.session_state.best_name = best_name
        st.session_state.best_pipe = fitted[best_name]
        st.session_state.trained = True

    if st.session_state.model_table is not None:
        st.markdown("### Results on Test Split (25%)")
        st.dataframe(st.session_state.model_table.style.format({
            "ROC-AUC": "{:.3f}",
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
        }), use_container_width=True)

        st.success(f"Selected (best ROC-AUC): **{st.session_state.best_name}**")

        # Confusion matrix for best
        bp = st.session_state.best_pipe
        proba = bp.predict_proba(X_test)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        cm = confusion_matrix(y_test, pred)

        st.markdown(f"### Confusion Matrix (threshold = {THRESHOLD:.2f})")
        fig = plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xticks([0, 1], ["No (0)", "Yes (1)"])
        plt.yticks([0, 1], ["No (0)", "Yes (1)"])
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        plt.colorbar()
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()
    st.markdown("### Individual Risk Prediction (uses selected best model)")
    if not st.session_state.trained:
        st.info("Train the models first.")
    else:
        # Build input form from dataset categories
        user_row = {}
        for col in X.columns:
            if col in ["Waist circumference (cm)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)", "BMI (kg/m²)"]:
                user_row[col] = st.number_input(col, value=float(df_model[col].median(skipna=True) if col in df_model.columns else 0.0))
            else:
                options = sorted(df_model[col].dropna().astype(str).unique().tolist())
                # keep small lists reasonable
                if len(options) == 0:
                    user_row[col] = st.text_input(col, "")
                else:
                    user_row[col] = st.selectbox(col, options)

        if st.button("Predict risk"):
            X_input = pd.DataFrame([user_row])
            prob = float(st.session_state.best_pipe.predict_proba(X_input)[:, 1][0])
            st.session_state.last_prob = prob
            st.session_state.last_input = user_row

            if prob >= THRESHOLD:
                st.error(f"Estimated probability: **{prob:.2f}** → Higher Risk (threshold={THRESHOLD:.2f})")
            else:
                st.success(f"Estimated probability: **{prob:.2f}** → Lower Risk (threshold={THRESHOLD:.2f})")


# -----------------------------
# TAB: SIGNIFICANT FACTORS
# -----------------------------
with tab_explain:
    st.subheader("Significant Predictive Factors (PPS-aligned)")

    if not st.session_state.get("trained", False):
        st.info("Train models first (Model Training tab).")
    else:
        bp = st.session_state.best_pipe

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### Global importance (Permutation on test set)")
            st.caption("This shows which features reduce ROC-AUC most when shuffled (model-agnostic).")

            # permutation importance on pipeline (works, but can be a bit slow)
            perm = permutation_importance(
                bp, X_test, y_test,
                n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc"
            )

            # feature names after preprocessing
            try:
                names = bp.named_steps["preprocess"].get_feature_names_out()
            except Exception:
                names = np.array([f"feature_{i}" for i in range(len(perm.importances_mean))])

            imp = pd.DataFrame({
                "Feature": names,
                "Importance": perm.importances_mean
            }).sort_values("Importance", ascending=False).head(15)

            fig = plt.figure()
            plt.barh(imp["Feature"][::-1], imp["Importance"][::-1])
            plt.title("Top 15 Permutation Importances")
            plt.xlabel("ROC-AUC drop")
            plt.tight_layout()
            st.pyplot(fig)

            st.dataframe(imp, use_container_width=True)

        with colB:
            st.markdown("### User-friendly explanation (rule-based)")
            st.caption("If a user predicted risk, this explains *why* in simple clinical language.")

            if "last_input" not in st.session_state:
                st.info("Run a prediction first to get personalised reasons.")
            else:
                reasons = explain_risk_rules(st.session_state.last_input)
                for r in reasons:
                    st.write("•", r)

        st.divider()
        st.markdown("### Simple model-level explanation (Logistic Regression coefficients)")
        st.caption("Helpful for report discussion: interpretable direction & strength (after preprocessing).")

        # Train a quick LR baseline for interpretability
        pre = build_preprocessor(X_train)
        smote = SMOTE(sampling_strategy=0.9, k_neighbors=3, random_state=RANDOM_STATE)

        lr_pipe = ImbPipeline([
            ("preprocess", pre),
            ("smote", smote),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE))
        ])
        lr_pipe.fit(X_train, y_train)

        coef = lr_pipe.named_steps["model"].coef_.ravel()
        try:
            feat_names = lr_pipe.named_steps["preprocess"].get_feature_names_out()
        except Exception:
            feat_names = np.array([f"feature_{i}" for i in range(len(coef))])

        coef_df = pd.DataFrame({"Feature": feat_names, "AbsCoef": np.abs(coef), "Coef": coef})
        coef_df = coef_df.sort_values("AbsCoef", ascending=False).head(15)

        fig = plt.figure()
        plt.barh(coef_df["Feature"][::-1], coef_df["Coef"][::-1])
        plt.title("Top 15 Logistic Regression Coefficients")
        plt.xlabel("Coefficient (direction matters)")
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(coef_df, use_container_width=True)


# -----------------------------
# TAB: AWARENESS + PDF
# -----------------------------
with tab_awareness:
    st.subheader("Awareness Quiz + PDF Leaflet")
    st.write("Answer the questions. You’ll get Sri Lanka-friendly tips and can download a leaflet.")

    name = st.text_input("Name (optional)", value=st.session_state.get("last_name", ""))
    st.session_state["last_name"] = name

    q1 = st.radio("1) Exercise at least 30 minutes?", ["Rarely", "1–2 days/week", "3–5 days/week", "Almost daily"])
    q2 = st.radio("2) Sweet tea / sugary drinks per day?", ["0", "1", "2", "3 or more"])
    q3 = st.radio("3) Rice portion size usually?", ["Small", "Medium", "Large"])
    q4 = st.radio("4) Family history of diabetes?", ["No", "Yes"])
    q5 = st.radio("5) Sleep most nights?", ["<6 hours", "6–7 hours", "7–8 hours", "8+ hours"])
    q6 = st.radio("6) Fried foods frequency?", ["Rarely", "1–2/week", "3–4/week", "Almost daily"])
    q7 = st.radio("7) Fruits & vegetables per day?", ["<2 portions", "2–3", "4–5", "More than 5"])
    q8 = st.radio("8) Alcohol consumption?", ["No", "Occasionally", "Weekly", "Daily"])
    q9 = st.radio("9) Smoking?", ["No", "Yes"])
    q10 = st.radio("10) Stress level?", ["Low", "Moderate", "High"])
    q11 = st.radio("11) Do you know HbA1c test?", ["No", "Yes"])
    q12 = st.radio("12) Have you ever checked blood sugar?", ["Never", "Once a year", "Sometimes", "Regularly"])

    if st.button("Get my tips"):
        tips = []

        if q1 in ["Rarely", "1–2 days/week"]:
            tips.append("Try a 30-minute walk after dinner at least 5 days/week.")
            tips.append("If busy: 10 minutes walking × 3 times/day.")

        if q2 in ["2", "3 or more"]:
            tips.append("Reduce sweet tea/soft drinks gradually (half sugar → quarter → none).")
            tips.append("Replace with water or unsweetened tea.")

        if q3 == "Large":
            tips.append("Reduce rice portion; increase vegetables (gotukola, mukunuwenna, beans, cabbage).")

        if q4 == "Yes":
            tips.append("Family history increases risk: consider regular screening (FBS/HbA1c).")

        if q5 == "<6 hours":
            tips.append("Aim for 7–8 hours sleep — poor sleep can worsen insulin resistance.")

        if q6 in ["3–4/week", "Almost daily"]:
            tips.append("Limit fried foods; prefer boiled/steamed/grilled meals.")

        if q7 in ["<2 portions", "2–3"]:
            tips.append("Increase fibre: add vegetables + fruit (portion control).")

        if q8 in ["Weekly", "Daily"]:
            tips.append("Reduce alcohol frequency; alcohol can affect weight and blood sugar control.")

        if q9 == "Yes":
            tips.append("Avoid smoking — it increases cardiovascular risk in diabetes.")

        if q10 == "High":
            tips.append("Stress management helps: breathing exercises, walking, prayer/meditation.")

        if q11 == "No":
            tips.append("Learn HbA1c: it estimates average blood sugar for ~3 months.")

        if q12 in ["Never", "Once a year"]:
            tips.append("If you have risk factors, check fasting blood sugar / HbA1c more regularly.")

        tips.append("Choose protein/fibre snacks: roasted gram (kadala), plain yogurt, nuts (small portion).")

        st.session_state["last_tips"] = tips

        st.markdown("### Your tips")
        for t in tips:
            st.write("•", t)

    st.divider()

    st.markdown("### Download PDF leaflet")
    prob = st.session_state.get("last_prob", None)
    label = "N/A"
    if prob is not None:
        label = "Higher Risk" if prob >= THRESHOLD else "Lower Risk"

    tips = st.session_state.get("last_tips", [])
    if st.button("⚠️ Generate PDF leaflet"):
        pdf_bytes = make_pdf_leaflet(name=name, prob=prob, label=label, tips=tips)
        st.download_button(
            "⬇️ Download PDF leaflet",
            data=pdf_bytes,
            file_name="diabetes_awareness_leaflet.pdf",
            mime="application/pdf"
        )

st.caption("⚠️ Disclaimer: Educational screening tool only. Not a medical diagnosis.")
