import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, brier_score_loss


# ----------------------------
# App setup
# ----------------------------
st.set_page_config(page_title="Diabetes Risk Analytics (Sri Lanka)", layout="wide")
st.title("📊 Diabetes Risk Analytics Dashboard (Sri Lanka)")
st.caption("Visual analytics dashboard to explore risk patterns and key predictive factors (PPS aligned).")
st.warning("⚠️ Educational tool only — this is not a medical diagnosis.")


# ----------------------------
# Column names (match your Google Form export)
# ----------------------------
TARGET_COL = "Have you ever been diagnosed with diabetes?"

LEAKAGE_COLS = [
    "Are you currently taking any medications for diabetes or related conditions?"
]

# These symptoms are useful clinically, but if your goal is EARLY screening,
# they can make the model look better than it really is (because they appear late).
LATE_SYMPTOMS = [
    "Do you experience frequent urination?",
    "Do you often feel unusually thirsty?",
    "Do you feel unusually fatigued or tired?",
    "Do you have blurred vision or slow-healing wounds?",
]

NUMERIC_CANDIDATES = [
    "Waist circumference (cm)",
    "Systolic BP (mmHg)",
    "Diastolic BP (mmHg)",
    "BMI (kg/m²)",
]

RANDOM_STATE = 42


# ----------------------------
# Utility functions
# ----------------------------
def read_excel_file(file) -> pd.DataFrame:
    return pd.read_excel(file)


def tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Google Forms often has this column
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET_COL not in df.columns:
        return df

    df[TARGET_COL] = (
        df[TARGET_COL].astype(str).str.strip().map({"Yes": 1, "No": 0})
    )
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_rule_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple rule-based risk score used for analytics visualisation.
    Not a diagnosis, just a way to group people into Low/Medium/High.
    """
    df = df.copy()
    required = ["BMI (kg/m²)", "Waist circumference (cm)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)"]

    if not all(c in df.columns for c in required):
        df["Risk Score (rule)"] = np.nan
        df["Risk Band"] = "Unknown"
        return df

    bmi = df["BMI (kg/m²)"]
    waist = df["Waist circumference (cm)"]
    sbp = df["Systolic BP (mmHg)"]
    dbp = df["Diastolic BP (mmHg)"]

    score = np.zeros(len(df), dtype=float)

    # BMI
    score += (bmi >= 25).fillna(False).astype(int)
    score += (bmi >= 30).fillna(False).astype(int)

    # waist (basic threshold for demonstration)
    score += (waist >= 90).fillna(False).astype(int)

    # BP
    score += (sbp >= 130).fillna(False).astype(int)
    score += (dbp >= 85).fillna(False).astype(int)

    df["Risk Score (rule)"] = score
    df["Risk Band"] = np.where(score <= 1, "Low", np.where(score <= 3, "Medium", "High"))
    return df


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
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

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )


def fig_as_png(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def generate_leaflet_pdf(summary_lines, tips_lines) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    _, h = A4

    y = h - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Diabetes Awareness Leaflet (Sri Lanka)")
    y -= 22

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Summary")
    y -= 18
    c.setFont("Helvetica", 10)

    for line in summary_lines:
        c.drawString(50, y, f"• {line}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Practical Tips")
    y -= 18
    c.setFont("Helvetica", 10)

    for tip in tips_lines[:14]:
        if y < 80:
            c.showPage()
            y = h - 60
            c.setFont("Helvetica", 10)
        c.drawString(50, y, f"• {tip}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Disclaimer: Educational tool only. Not a medical diagnosis.")
    c.save()

    pdf = buf.getvalue()
    buf.close()
    return pdf


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload survey Excel (.xlsx)", type=["xlsx"])

st.sidebar.header("Options")
remove_leakage = st.sidebar.checkbox("Exclude medication question (avoid leakage)", value=True)
remove_late_symptoms = st.sidebar.checkbox("Exclude late symptoms (early screening focus)", value=True)
test_size = st.sidebar.slider("Test size", 0.15, 0.40, 0.25, 0.05)
threshold = st.sidebar.slider("Decision threshold", 0.20, 0.80, 0.40, 0.05)

st.sidebar.caption("If you keep late symptoms, model accuracy can look unrealistically high.")


# ----------------------------
# Load + prep dataset
# ----------------------------
if not uploaded:
    st.info("Upload your Excel file to begin.")
    st.stop()

df = read_excel_file(uploaded)
df = tidy_columns(df)
df = encode_target(df)
df = convert_numeric(df)

drop_cols = []
if remove_leakage:
    drop_cols += [c for c in LEAKAGE_COLS if c in df.columns]
if remove_late_symptoms:
    drop_cols += [c for c in LATE_SYMPTOMS if c in df.columns]

df_model = df.drop(columns=drop_cols, errors="ignore")
df_model = build_rule_risk(df_model)

if TARGET_COL not in df_model.columns:
    st.error(f"Target column not found: {TARGET_COL}")
    st.stop()

X = df_model.drop(columns=[TARGET_COL])
y = df_model[TARGET_COL].astype(int)


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Data Quality", "Risk Analytics", "Model + Explainability", "Awareness + PDF"]
)

# ----------------------------
# Tab 1: Overview
# ----------------------------
with tab1:
    st.subheader("Overview")

    a, b, c = st.columns(3)
    a.metric("Rows", df_model.shape[0])
    b.metric("Features", X.shape[1])
    c.metric("Diabetes (Yes)", int((y == 1).sum()))

    st.write("Preview:")
    st.dataframe(df_model.head(10), use_container_width=True)

    st.markdown("**Target distribution (Plot 1)**")
    counts = y.value_counts().sort_index()
    fig = plt.figure()
    plt.bar(["No (0)", "Yes (1)"], [counts.get(0, 0), counts.get(1, 0)])
    plt.ylabel("Count")
    plt.title("Plot 1: Diabetes class distribution")
    plt.tight_layout()
    st.pyplot(fig)

    st.download_button(
        "⬇️ Download Plot 1 (PNG)",
        data=fig_as_png(fig),
        file_name="plot01_target_distribution.png",
        mime="image/png"
    )

# ----------------------------
# Tab 2: Data Quality
# ----------------------------
with tab2:
    st.subheader("Data Quality")

    st.markdown("**Missing values (Plot 2)**")
    missing = df_model.isna().sum().sort_values(ascending=False)
    top = missing[missing > 0].head(15)

    fig = plt.figure()
    if top.empty:
        plt.text(0.1, 0.5, "No missing values found.", fontsize=12)
        plt.axis("off")
    else:
        plt.bar(top.index.astype(str), top.values)
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("Missing count")
        plt.title("Plot 2: Missing values (top columns)")
        plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**Numeric distributions (Plots 3–6)**")
    numeric_cols = [c for c in NUMERIC_CANDIDATES if c in df_model.columns]
    colA, colB = st.columns(2)
    plot_no = 3

    for i, col in enumerate(numeric_cols[:4]):
        vals = pd.to_numeric(df_model[col], errors="coerce").dropna()
        fig = plt.figure()
        if vals.empty:
            plt.text(0.1, 0.5, f"No numeric data for {col}", fontsize=12)
            plt.axis("off")
        else:
            plt.hist(vals, bins=20)
            plt.title(f"Plot {plot_no}: {col} distribution")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()

        (colA if i % 2 == 0 else colB).pyplot(fig)
        plot_no += 1

    st.markdown("**Correlation heatmap (numeric only) (Plot 7)**")
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df_model[num_cols].corr(numeric_only=True)
        fig = plt.figure()
        plt.imshow(corr.values, aspect="auto")
        plt.title("Plot 7: Correlation (numeric features)")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.colorbar()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

# ----------------------------
# Tab 3: Risk Analytics
# ----------------------------
with tab3:
    st.subheader("Risk Analytics")

    st.markdown("**Risk band distribution (Plot 8)**")
    bands = df_model["Risk Band"].value_counts()
    fig = plt.figure()
    plt.bar(bands.index, bands.values)
    plt.ylabel("Count")
    plt.title("Plot 8: Rule-based risk bands")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**Risk band vs outcome (Plot 9)**")
    ct = pd.crosstab(df_model["Risk Band"], df_model[TARGET_COL], normalize="index")
    fig = plt.figure()
    x = np.arange(len(ct.index))
    p0 = ct.get(0, pd.Series([0]*len(ct.index), index=ct.index)).values
    p1 = ct.get(1, pd.Series([0]*len(ct.index), index=ct.index)).values
    plt.bar(x, p0, label="No diabetes")
    plt.bar(x, p1, bottom=p0, label="Diabetes")
    plt.xticks(x, ct.index)
    plt.ylabel("Proportion")
    plt.title("Plot 9: Outcome proportions per risk band")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

# ----------------------------
# Tab 4: Model + Explainability
# ----------------------------
with tab4:
    st.subheader("Model + Explainability")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=RANDOM_STATE, stratify=y
    )

    pre = make_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    best_pipe, best_name, best_auc = None, None, -1

    for name, model in models.items():
        pipe = Pipeline([("preprocess", pre), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        auc_mean = float(scores.mean())
        auc_std = float(scores.std())
        results.append((name, auc_mean, auc_std))

        if auc_mean > best_auc:
            best_auc = auc_mean
            best_name = name
            best_pipe = pipe

    res_df = pd.DataFrame(results, columns=["Model", "CV ROC-AUC Mean", "CV ROC-AUC Std"]).sort_values(
        "CV ROC-AUC Mean", ascending=False
    )
    st.write("Cross-validation results:")
    st.dataframe(res_df, use_container_width=True)

    best_pipe.fit(X_train, y_train)
    proba = best_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= float(threshold)).astype(int)

    metrics = {
        "Model": best_name,
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "Brier (calibration)": brier_score_loss(y_test, proba),
    }
    st.write("Test metrics (with selected threshold):")
    st.write(pd.DataFrame([metrics]).round(4))

    st.markdown("**Permutation importance (top 12)**")
    perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc")
    perm_df = pd.DataFrame({"Feature": X_test.columns, "Importance": perm.importances_mean}).sort_values(
        "Importance", ascending=False
    ).head(12)

    fig = plt.figure()
    plt.barh(perm_df["Feature"][::-1], perm_df["Importance"][::-1])
    plt.xlabel("ROC-AUC drop")
    plt.title("Top drivers (permutation importance)")
    plt.tight_layout()
    st.pyplot(fig)

# ----------------------------
# Tab 5: Awareness + PDF
# ----------------------------
with tab5:
    st.subheader("Awareness + PDF")

    st.write("This part is for simple awareness tips and a PDF leaflet download.")

    name_for_pdf = st.text_input("Name (optional)", value="")
    tips = ["Maintain healthy diet, regular activity, good sleep, and routine screening."]

    if st.button("Generate PDF leaflet"):
        summary = [
            f"Name: {name_for_pdf if name_for_pdf else 'N/A'}",
            "Tool: Diabetes Risk Analytics Dashboard (Sri Lanka)",
            "Reminder: Educational tool only. Not a diagnosis."
        ]
        pdf = generate_leaflet_pdf(summary, tips)
        st.download_button(
            label="⬇️ Download PDF leaflet",
            data=pdf,
            file_name="diabetes_awareness_leaflet.pdf",
            mime="application/pdf"
        )

st.caption("Academic prototype — built for final year project submission.")
