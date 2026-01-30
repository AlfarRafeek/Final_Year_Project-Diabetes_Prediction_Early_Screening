# streamlit_app.py
# ============================================================
# Diabetes Early Screening - DATA ANALYTICS DASHBOARD (Streamlit)
# - Matches PPS: visualize risk levels + key predictive variables
# - Includes EDA + Risk distribution + Significant factors + Model eval + Prediction
#
# Works in 2 modes:
#   1) If best_pipe.pkl exists and loads -> uses it
#   2) If not / incompatible -> trains a quick baseline pipeline from the uploaded Excel
#
# IMPORTANT (for Streamlit Cloud):
# - Put your dataset in the repo OR upload via the app uploader.
# - Recommended repo structure:
#     streamlit_app.py
#     requirements.txt
#     Diabetes Risk Survey (Responses) (1).xlsx   (optional if you prefer uploader)
#     best_pipe.pkl                               (optional)
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Diabetes Analytics Dashboard (Sri Lanka)", layout="wide")

TARGET_COL = "Have you ever been diagnosed with diabetes?"
DEFAULT_DATA_FILE = "Diabetes Risk Survey (Responses) (1).xlsx"
DEFAULT_MODEL_FILE = "best_pipe.pkl"
RANDOM_STATE = 42

RISK_BINS_DEFAULT = (0.40, 0.70)  # low <0.40, med 0.40-0.69, high >=0.70


# -----------------------------
# HELPER: clean + prepare data
# -----------------------------
def load_data_from_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Drop Timestamp if exists
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    # Strip column names
    df.columns = [str(c).strip() for c in df.columns]

    # Clean target mapping
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().map({"No": 0, "Yes": 1})
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Numeric conversions (safe)
    for col in ["Waist circumference (cm)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)", "BMI (kg/m²)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return pre


def try_load_model(model_path: str):
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"
    try:
        pipe = joblib.load(model_path)
        return pipe, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def train_fallback_model(df: pd.DataFrame):
    """
    If best_pipe.pkl is missing/incompatible, train a quick baseline pipeline.
    This keeps the dashboard working on Streamlit Cloud.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # OPTIONAL: drop likely leakage column if present (post-diagnosis)
    leakage_col = "Are you currently taking any medications for diabetes or related conditions?"
    if leakage_col in X.columns:
        X = X.drop(columns=[leakage_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    pre = build_preprocessor(X_train)
    model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    return pipe, X_train, X_test, y_train, y_test


def get_train_test_from_pipe(df: pd.DataFrame):
    """
    For consistency, we create a split for evaluation visuals.
    This is independent from the uploaded model training split.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    leakage_col = "Are you currently taking any medications for diabetes or related conditions?"
    if leakage_col in X.columns:
        X = X.drop(columns=[leakage_col])

    return train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)


def risk_group(prob: float, low_thr: float, high_thr: float) -> str:
    if prob < low_thr:
        return "Low"
    if prob < high_thr:
        return "Medium"
    return "High"


def fig_target_distribution(y: pd.Series):
    counts = y.value_counts().sort_index()
    fig = plt.figure()
    plt.bar(["No (0)", "Yes (1)"], [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Diabetes Target Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    return fig


def fig_hist(series: pd.Series, title: str, xlabel: str):
    fig = plt.figure()
    plt.hist(series.dropna(), bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    return fig


def fig_box_by_target(df: pd.DataFrame, col: str):
    fig = plt.figure()
    a = df[df[TARGET_COL] == 0][col].dropna()
    b = df[df[TARGET_COL] == 1][col].dropna()
    plt.boxplot([a, b], labels=["No (0)", "Yes (1)"])
    plt.title(f"{col} by Diabetes Status")
    plt.ylabel(col)
    plt.tight_layout()
    return fig


def fig_cat_by_target(df: pd.DataFrame, col: str):
    # counts by target and category
    tmp = df[[col, TARGET_COL]].dropna()
    ct = pd.crosstab(tmp[col], tmp[TARGET_COL])
    # ensure both columns exist
    if 0 not in ct.columns:
        ct[0] = 0
    if 1 not in ct.columns:
        ct[1] = 0
    ct = ct[[0, 1]].sort_index()

    fig = plt.figure()
    x = np.arange(len(ct.index))
    plt.bar(x - 0.2, ct[0].values, width=0.4, label="No (0)")
    plt.bar(x + 0.2, ct[1].values, width=0.4, label="Yes (1)")
    plt.xticks(x, ct.index.astype(str), rotation=45, ha="right")
    plt.title(f"{col} vs Diabetes Status")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    return fig


@st.cache_data
def compute_permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=10):
    perm = permutation_importance(
        pipe, X_test, y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        scoring=scoring
    )

    # feature names after preprocessing (best effort)
    feat_names = None
    try:
        pre = pipe.named_steps.get("preprocess") or pipe.named_steps.get("preprocessor")
        if pre is not None:
            feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = None

    if feat_names is None or len(feat_names) != len(perm.importances_mean):
        feat_names = [f"feature_{i}" for i in range(len(perm.importances_mean))]

    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False)

    return imp_df


def fig_importance_bar(imp_df: pd.DataFrame, top_k=12, title="Permutation Importance (ROC-AUC drop)"):
    top = imp_df.head(top_k).copy()
    fig = plt.figure()
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    return fig


# -----------------------------
# UI HEADER
# -----------------------------
st.title("📊 Diabetes Risk Analytics Dashboard (Sri Lanka)")
st.caption("Data analytics dashboard to visualize diabetes risk levels and significant predictive factors (PPS-aligned).")


# -----------------------------
# SIDEBAR: Data + Model sources
# -----------------------------
st.sidebar.header("Data & Model")

uploaded = st.sidebar.file_uploader("Upload your dataset (Excel .xlsx)", type=["xlsx"])

data_path_used = None
df = None

if uploaded is not None:
    df = pd.read_excel(uploaded)
    df.columns = [str(c).strip() for c in df.columns]
    # Apply same cleaning using a temp file-like approach
    # Convert using same logic after reading:
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().map({"No": 0, "Yes": 1})
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)
    for col in ["Waist circumference (cm)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)", "BMI (kg/m²)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    data_path_used = "Uploaded file"
else:
    if os.path.exists(DEFAULT_DATA_FILE):
        df = load_data_from_excel(DEFAULT_DATA_FILE)
        data_path_used = DEFAULT_DATA_FILE

if df is None:
    st.warning("Upload the Excel dataset in the sidebar, or add it to the repo with the expected filename.")
    st.stop()

if TARGET_COL not in df.columns:
    st.error(f"Target column not found in dataset: {TARGET_COL}")
    st.info("Check your Excel column name matches exactly.")
    st.stop()

st.sidebar.success(f"Loaded data: {data_path_used}")
st.sidebar.write("Rows:", df.shape[0])
st.sidebar.write("Columns:", df.shape[1])

# Risk thresholds
low_thr, high_thr = st.sidebar.slider(
    "Risk group thresholds (Low / Medium / High)",
    min_value=0.05, max_value=0.95, value=RISK_BINS_DEFAULT, step=0.05
)

# Model load preference
use_saved_model = st.sidebar.toggle("Use saved model (best_pipe.pkl) if available", value=True)

pipe = None
pipe_source = None

if use_saved_model:
    pipe, err = try_load_model(DEFAULT_MODEL_FILE)
    if pipe is not None:
        pipe_source = "Loaded best_pipe.pkl"
    else:
        pipe_source = f"Fallback (could not load best_pipe.pkl: {err})"

if pipe is None:
    pipe, X_train, X_test, y_train, y_test = train_fallback_model(df)
    pipe_source = "Trained fallback baseline (Logistic Regression)"

st.sidebar.info(f"Model source: {pipe_source}")


# -----------------------------
# CREATE TRAIN/TEST FOR EVAL VISUALS
# -----------------------------
X_tr, X_te, y_tr, y_te = get_train_test_from_pipe(df)

# Make sure pipeline can run on these columns:
# If best_pipe.pkl expects different columns, evaluation tab will show a friendly message.
model_eval_ok = True
try:
    pipe.predict_proba(X_te.head(2))
except Exception:
    model_eval_ok = False


# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1) Data Overview",
    "2) Risk Levels",
    "3) Significant Factors",
    "4) Relationships (EDA)",
    "5) Model Performance",
    "6) Prediction Tool"
])


# ============================================================
# TAB 1: DATA OVERVIEW
# ============================================================
with tab1:
    st.subheader("Dataset Overview")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total responses", f"{df.shape[0]}")
    colB.metric("Total columns", f"{df.shape[1]}")
    colC.metric("Target column", TARGET_COL)
    colD.metric("Missing cells", f"{int(df.isna().sum().sum())}")

    st.markdown("### Preview (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Target Distribution (Diabetes Yes/No)")
    fig = fig_target_distribution(df[TARGET_COL])
    st.pyplot(fig)

    st.markdown("### Missing Values (top 12 columns)")
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0].head(12)
    if len(miss) == 0:
        st.info("No missing values detected.")
    else:
        fig2 = plt.figure()
        plt.bar(miss.index.astype(str), miss.values)
        plt.title("Missing Values (Top 12 Columns)")
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("Missing count")
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("### Numeric Distributions")
    num_candidates = [c for c in ["BMI (kg/m²)", "Waist circumference (cm)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)"] if c in df.columns]
    if not num_candidates:
        st.info("No standard numeric health columns found for histograms.")
    else:
        cols = st.columns(min(2, len(num_candidates)))
        for i, c in enumerate(num_candidates):
            fig3 = fig_hist(df[c], f"Distribution of {c}", c)
            cols[i % len(cols)].pyplot(fig3)


# ============================================================
# TAB 2: RISK LEVELS (risk distribution dashboard)
# ============================================================
with tab2:
    st.subheader("Risk Level Dashboard (Probability-based)")

    st.write(
        "This section visualizes **risk levels** using the model's predicted probability. "
        "Risk groups are created using your selected thresholds."
    )

    # Compute probabilities for all rows
    X_all = df.drop(columns=[TARGET_COL]).copy()

    # Drop leakage column if present (match training fallback)
    leakage_col = "Are you currently taking any medications for diabetes or related conditions?"
    if leakage_col in X_all.columns:
        X_all = X_all.drop(columns=[leakage_col])

    prob_all = None
    try:
        prob_all = pipe.predict_proba(X_all)[:, 1]
    except Exception as e:
        st.error("Could not compute probabilities for the dataset (model-feature mismatch).")
        st.code(str(e))
        st.info("Fix: Ensure the model was trained using the same columns as this dataset.")
        st.stop()

    df_risk = df.copy()
    df_risk["pred_prob"] = prob_all
    df_risk["risk_level"] = [risk_group(p, low_thr, high_thr) for p in df_risk["pred_prob"]]

    # Risk distribution
    risk_counts = df_risk["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)

    c1, c2, c3 = st.columns(3)
    c1.metric("Low risk", int(risk_counts["Low"]))
    c2.metric("Medium risk", int(risk_counts["Medium"]))
    c3.metric("High risk", int(risk_counts["High"]))

    fig = plt.figure()
    plt.bar(risk_counts.index, risk_counts.values)
    plt.title("Risk Group Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)

    # Top high risk profiles
    st.markdown("### Top 10 Highest-Risk Profiles (for analysis)")
    show_cols = [c for c in df_risk.columns if c not in []]
    top10 = df_risk.sort_values("pred_prob", ascending=False).head(10)
    st.dataframe(top10[show_cols], use_container_width=True)

    st.markdown("### Probability Distribution")
    fig2 = plt.figure()
    plt.hist(df_risk["pred_prob"], bins=20)
    plt.title("Predicted Probability Distribution")
    plt.xlabel("Predicted probability of diabetes")
    plt.ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig2)


# ============================================================
# TAB 3: SIGNIFICANT FACTORS (Permutation Importance)
# ============================================================
with tab3:
    st.subheader("Significant Predictive Factors")

    st.write(
        "To meet the PPS objective (**identify significant factors**), this section uses "
        "**Permutation Importance** (how much performance drops when a feature is shuffled). "
        "This is model-agnostic and suitable for reports."
    )

    if not model_eval_ok:
        st.warning("Model evaluation/importance is not available due to feature mismatch between the saved model and this dataset.")
        st.info("If you're using best_pipe.pkl, retrain and export it with the same columns as this dashboard dataset.")
    else:
        # Fit (or refit) on train for clean importance calculation
        # If loaded model is already trained, fit again is harmless for fallback; for a loaded model it may fail.
        # So we handle safely.
        try:
            pipe.fit(X_tr, y_tr)
        except Exception:
            pass

        imp_df = compute_permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=10)

        top_k = st.slider("Top K features to display", 5, 25, 12, 1)
        fig = fig_importance_bar(imp_df, top_k=top_k)
        st.pyplot(fig)

        st.markdown("### Top Factors Table")
        st.dataframe(imp_df.head(top_k), use_container_width=True)

        st.markdown("### Short interpretation (report-style)")
        st.write(
            "Features with higher permutation importance have a stronger influence on prediction performance. "
            "In healthcare screening, these features are important because they explain which variables contribute most to risk estimation."
        )


# ============================================================
# TAB 4: RELATIONSHIPS (EDA)
# ============================================================
with tab4:
    st.subheader("Feature Relationships (EDA)")

    st.write(
        "This section shows relationships between key variables and diabetes status. "
        "These visuals are what supervisors expect from a **Data Analyst**."
    )

    # Choose a few numeric & categorical columns automatically
    numeric_cols = [c for c in df.columns if c != TARGET_COL and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c != TARGET_COL and not pd.api.types.is_numeric_dtype(df[c])]

    st.markdown("### Numeric vs Diabetes (boxplots)")
    num_show = [c for c in ["BMI (kg/m²)", "Waist circumference (cm)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)"] if c in numeric_cols]
    if not num_show:
        st.info("No suitable numeric columns detected for boxplots.")
    else:
        grid = st.columns(2)
        for i, col in enumerate(num_show):
            fig = fig_box_by_target(df, col)
            grid[i % 2].pyplot(fig)

    st.markdown("### Categorical vs Diabetes (count comparisons)")
    # Pick your key lifestyle variables if present, else sample from cat columns
    preferred_cat = [
        "Age", "Gender", "Occupation",
        "Do you have a family history of diabetes?",
        "How often do you exercise per week?",
        "How would you describe your diet?",
        "Average sleep hours per night",
        "Have you ever been diagnosed with high blood pressure or cholesterol?",
        "Do you experience frequent urination?",
        "Do you often feel unusually thirsty?",
        "Do you feel unusually fatigued or tired?",
        "Do you have blurred vision or slow-healing wounds?"
    ]
    cat_show = [c for c in preferred_cat if c in cat_cols]
    if len(cat_show) == 0:
        cat_show = cat_cols[:6]

    grid2 = st.columns(2)
    for i, col in enumerate(cat_show[:8]):
        fig = fig_cat_by_target(df, col)
        grid2[i % 2].pyplot(fig)


# ============================================================
# TAB 5: MODEL PERFORMANCE
# ============================================================
with tab5:
    st.subheader("Model Performance (Evaluation)")

    if not model_eval_ok:
        st.warning("Cannot evaluate model (feature mismatch). Use the fallback-trained model or retrain/export best_pipe.pkl using the same dataset columns.")
    else:
        # Fit on train, evaluate on test
        try:
            pipe.fit(X_tr, y_tr)
        except Exception:
            pass

        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_te, pred)
        prec = precision_score(y_te, pred, zero_division=0)
        rec = recall_score(y_te, pred, zero_division=0)
        f1 = f1_score(y_te, pred, zero_division=0)
        auc = roc_auc_score(y_te, proba)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1-score", f"{f1:.3f}")
        m5.metric("ROC-AUC", f"{auc:.3f}")

        st.markdown("### Confusion Matrix")
        fig = plt.figure()
        ConfusionMatrixDisplay.from_predictions(y_te, pred)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### ROC Curve")
        fig2 = plt.figure()
        RocCurveDisplay.from_predictions(y_te, proba)
        plt.title("ROC Curve")
        plt.tight_layout()
        st.pyplot(fig2)

        st.markdown("### Notes (what to write in report)")
        st.write(
            "- Accuracy alone can be misleading with imbalanced classes.\n"
            "- Recall is important because it measures how well the model detects diabetic cases.\n"
            "- ROC-AUC summarizes discrimination ability across thresholds."
        )


# ============================================================
# TAB 6: PREDICTION TOOL (supporting feature, not main)
# ============================================================
with tab6:
    st.subheader("Risk Prediction (Supporting Tool)")

    st.write(
        "This form is included as a **supporting feature**. "
        "Your main deliverable is the analytics dashboard (tabs 1–5)."
    )

    # Build form using the dataset's feature columns (excluding target)
    X_cols = [c for c in df.columns if c != TARGET_COL]
    leakage_col = "Are you currently taking any medications for diabetes or related conditions?"
    if leakage_col in X_cols:
        X_cols.remove(leakage_col)

    # Split numeric vs categorical from the dataset
    tmpX = df[X_cols].copy()
    num_cols = tmpX.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in tmpX.columns if c not in num_cols]

    user_row = {}

    with st.form("predict_form"):
        st.markdown("### Enter inputs")

        # Basic numeric inputs (defaults)
        for c in num_cols:
            default_val = float(np.nanmedian(df[c].values)) if df[c].notna().any() else 0.0
            user_row[c] = st.number_input(c, value=float(default_val))

        # Categorical inputs from observed categories (keeps it consistent with training)
        for c in cat_cols:
            opts = [x for x in df[c].dropna().astype(str).unique().tolist()]
            opts = sorted(opts) if opts else ["Unknown"]
            user_row[c] = st.selectbox(c, opts)

        submit = st.form_submit_button("Predict risk")

    if submit:
        X_in = pd.DataFrame([user_row])
        try:
            p = float(pipe.predict_proba(X_in)[:, 1][0])
            level = risk_group(p, low_thr, high_thr)

            st.markdown("### Result")
            st.write(f"Predicted probability: **{p:.2f}**")
            if level == "High":
                st.error(f"Risk level: **{level}**")
            elif level == "Medium":
                st.warning(f"Risk level: **{level}**")
            else:
                st.success(f"Risk level: **{level}**")

            st.caption("Disclaimer: Educational screening tool only. Not a medical diagnosis.")

        except Exception as e:
            st.error("Prediction failed (model-feature mismatch).")
            st.code(str(e))
            st.info("If using best_pipe.pkl, retrain it using the same feature columns as the dataset used in this dashboard.")


st.divider()
st.caption("⚠️ Disclaimer: Educational screening prototype only — not a medical diagnosis.")
