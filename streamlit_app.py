import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_validate, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(page_title="Diabetes Risk Analytics Dashboard (Sri Lanka)", layout="wide")
st.title("📊 Diabetes Risk Analytics Dashboard (Sri Lanka)")
st.caption("Data analytics dashboard to visualize diabetes risk levels and significant predictive factors (PPS-aligned).")


# =========================
# CONSTANTS (EDIT IF NEEDED)
# =========================
RANDOM_STATE = 42
TARGET_COL = "Have you ever been diagnosed with diabetes?"

LEAKAGE_COLS = [
    "Are you currently taking any medications for diabetes or related conditions?"
]

LATE_SYMPTOM_COLS = [
    "Do you have blurred vision or slow-healing wounds?",
    "Do you experience frequent urination?",
    "Do you often feel unusually thirsty?",
    "Do you feel unusually fatigued or tired?"
]


# =========================
# HELPERS
# =========================
def safe_read_excel(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)
    # strip column names
    df.columns = [c.strip() for c in df.columns]
    # drop timestamp if exists
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    return df


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: '{TARGET_COL}'")

    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df[TARGET_COL] = df[TARGET_COL].map({"No": 0, "Yes": 1})
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return pre


def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=600, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def make_pipeline(preprocess, model):
    smote = SMOTE(sampling_strategy=0.9, k_neighbors=3, random_state=RANDOM_STATE)
    return ImbPipeline(steps=[
        ("preprocess", preprocess),
        ("smote", smote),
        ("model", model),
    ])


def align_to_model_columns(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Prevents: ValueError columns are missing"""
    if hasattr(pipe.named_steps.get("preprocess", None), "feature_names_in_"):
        expected = list(pipe.named_steps["preprocess"].feature_names_in_)
        return X.reindex(columns=expected)
    return X


def plot_bar_counts(labels, values, title, xlabel="", ylabel="Count"):
    fig = plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(fig)


def plot_heatmap_corr(df_num, title):
    corr = df_num.corr(numeric_only=True)
    fig = plt.figure()
    plt.imshow(corr.values)
    plt.title(title)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.tight_layout()
    st.pyplot(fig)


# =========================
# SIDEBAR: DATA UPLOAD + SETTINGS
# =========================
st.sidebar.header("1) Data")
uploaded = st.sidebar.file_uploader("Upload your Excel dataset (.xlsx)", type=["xlsx"])

st.sidebar.header("2) Leakage / Scope")
drop_leakage = st.sidebar.checkbox("Drop medication question (leakage)", value=True)
drop_late_symptoms = st.sidebar.checkbox("Drop late symptoms (harder early screening)", value=True)

st.sidebar.header("3) Training")
test_size = st.sidebar.slider("Test size", 0.2, 0.4, 0.25, 0.05)
threshold = st.sidebar.slider("Risk threshold", 0.2, 0.8, 0.4, 0.05)

train_btn = st.sidebar.button("🚀 Train & Compare Models")

# session state init
if "trained" not in st.session_state:
    st.session_state.trained = False
if "best_pipe" not in st.session_state:
    st.session_state.best_pipe = None
if "best_name" not in st.session_state:
    st.session_state.best_name = None
if "metrics_table" not in st.session_state:
    st.session_state.metrics_table = None
if "feature_signature" not in st.session_state:
    st.session_state.feature_signature = None


# =========================
# LOAD DATA
# =========================
if uploaded is None:
    st.info("Upload your Excel dataset to start (the app will read your Google Form export).")
    st.stop()

df = safe_read_excel(uploaded)
df = map_target(df)

# choose columns to drop
to_drop = []
if drop_leakage:
    to_drop += [c for c in LEAKAGE_COLS if c in df.columns]
if drop_late_symptoms:
    to_drop += [c for c in LATE_SYMPTOM_COLS if c in df.columns]

df_model = df.drop(columns=to_drop, errors="ignore")

# build X/y
X = df_model.drop(columns=[TARGET_COL])
y = df_model[TARGET_COL].astype(int)

# RESET MODEL AUTOMATICALLY IF COLUMNS CHANGED (prevents missing-columns errors)
current_signature = tuple(X.columns)
if st.session_state.feature_signature != current_signature:
    st.session_state.feature_signature = current_signature
    st.session_state.trained = False
    st.session_state.best_pipe = None
    st.session_state.best_name = None
    st.session_state.metrics_table = None


# =========================
# MAIN TABS
# =========================
tab_overview, tab_models, tab_factors, tab_screening, tab_awareness = st.tabs([
    "📌 Data Overview",
    "🤖 Model Comparison",
    "⭐ Significant Factors",
    "🩺 Risk Screening",
    "📚 Awareness"
])


# =========================
# TAB 1: DATA OVERVIEW
# =========================
with tab_overview:
    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("Dataset Summary")
        st.write("Rows:", df_model.shape[0])
        st.write("Columns (including target):", df_model.shape[1])
        st.write("Dropped columns:", to_drop if to_drop else "None")
        st.markdown("**Target distribution**")
        counts = y.value_counts().sort_index()
        st.write(counts)

        plot_bar_counts(
            labels=["No (0)", "Yes (1)"],
            values=[counts.get(0, 0), counts.get(1, 0)],
            title="Class Distribution (Before balancing)"
        )

    with colB:
        st.subheader("Missing Values")
        miss = df_model.isna().sum().sort_values(ascending=False)
        miss_nonzero = miss[miss > 0].head(15)

        if len(miss_nonzero) == 0:
            st.success("No missing values detected.")
        else:
            fig = plt.figure()
            plt.bar(miss_nonzero.index.astype(str), miss_nonzero.values)
            plt.xticks(rotation=75, ha="right")
            plt.title("Top Missing Columns")
            plt.tight_layout()
            st.pyplot(fig)

    st.subheader("Numeric Distributions")
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    show_nums = [c for c in ["BMI (kg/m²)", "Waist circumference (cm)", "Systolic BP (mmHg)", "Diastolic BP (mmHg)"] if c in X.columns]
    if not show_nums and num_cols:
        show_nums = num_cols[:4]

    cols = st.columns(2)
    for i, col in enumerate(show_nums[:4]):
        with cols[i % 2]:
            fig = plt.figure()

series = df_model[col].dropna()

# Check if column is truly numeric
if pd.api.types.is_numeric_dtype(series):
    plt.hist(series, bins=20)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {col}")
else:
    # categorical → bar chart
    counts = series.value_counts()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(f"Distribution of {col}")

plt.tight_layout()
st.pyplot(fig)


    st.subheader("Correlation Heatmap (numeric only)")
    if len(num_cols) >= 2:
        plot_heatmap_corr(df_model[num_cols], "Correlation Heatmap (Numeric Features)")
    else:
        st.info("Not enough numeric columns for correlation heatmap.")


# =========================
# TRAINING + EVALUATION
# =========================
def train_and_compare(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    preprocess = build_preprocess(X_train)
    models = make_models()

    # repeated CV = more reliable than one split
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "acc": "accuracy", "f1": "f1", "recall": "recall", "precision": "precision"}

    rows = []
    best_name = None
    best_auc = -1
    best_pipe = None

    for name, model in models.items():
        pipe = make_pipeline(preprocess, model)
        scores = cross_validate(pipe, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)

        row = {
            "Model": name,
            "CV ROC-AUC": float(np.mean(scores["test_roc_auc"])),
            "CV Accuracy": float(np.mean(scores["test_acc"])),
            "CV F1": float(np.mean(scores["test_f1"])),
            "CV Recall": float(np.mean(scores["test_recall"])),
            "CV Precision": float(np.mean(scores["test_precision"])),
        }
        rows.append(row)

        if row["CV ROC-AUC"] > best_auc:
            best_auc = row["CV ROC-AUC"]
            best_name = name
            best_pipe = pipe

    table = pd.DataFrame(rows).sort_values("CV ROC-AUC", ascending=False)

    # fit best on full train
    best_pipe.fit(X_train, y_train)

    # align X_test to model columns (prevents missing-columns)
    X_test_aligned = align_to_model_columns(best_pipe, X_test)

    proba = best_pipe.predict_proba(X_test_aligned)[:, 1]
    pred = (proba >= threshold).astype(int)

    test_metrics = {
        "Test ROC-AUC": roc_auc_score(y_test, proba),
        "Test Accuracy": accuracy_score(y_test, pred),
        "Test Precision": precision_score(y_test, pred, zero_division=0),
        "Test Recall": recall_score(y_test, pred, zero_division=0),
        "Test F1": f1_score(y_test, pred, zero_division=0),
        "Brier Score": brier_score_loss(y_test, proba),
    }

    return table, best_name, best_pipe, (X_train, X_test, y_train, y_test, proba, pred, test_metrics)


# =========================
# TAB 2: MODEL COMPARISON
# =========================
with tab_models:
    st.subheader("Model Comparison (Repeated CV)")

    if train_btn:
        with st.spinner("Training models + evaluating with repeated CV..."):
            table, best_name, best_pipe, pack = train_and_compare(X, y)

        st.session_state.trained = True
        st.session_state.best_name = best_name
        st.session_state.best_pipe = best_pipe
        st.session_state.metrics_table = table
        st.session_state.pack = pack

    if st.session_state.metrics_table is None:
        st.info("Click **Train & Compare Models** in the sidebar.")
    else:
        st.dataframe(st.session_state.metrics_table, use_container_width=True)

        fig = plt.figure()
        plt.bar(st.session_state.metrics_table["Model"], st.session_state.metrics_table["CV ROC-AUC"])
        plt.xticks(rotation=35, ha="right")
        plt.title("Cross-Validated ROC-AUC (Mean)")
        plt.ylabel("ROC-AUC")
        plt.tight_layout()
        st.pyplot(fig)

        st.success(f"🏆 Best model (by CV ROC-AUC): **{st.session_state.best_name}**")

        X_train, X_test, y_train, y_test, proba, pred, test_metrics = st.session_state.pack
        st.markdown("### Test set results (threshold applied)")
        st.write({k: round(v, 4) for k, v in test_metrics.items()})

        c1, c2 = st.columns(2)
        with c1:
            fig = plt.figure()
            ConfusionMatrixDisplay.from_predictions(y_test, pred)
            plt.title("Confusion Matrix (Test)")
            st.pyplot(fig)

        with c2:
            fig = plt.figure()
            RocCurveDisplay.from_predictions(y_test, proba)
            plt.title("ROC Curve (Test)")
            st.pyplot(fig)

        fig = plt.figure()
        PrecisionRecallDisplay.from_predictions(y_test, proba)
        plt.title("Precision-Recall Curve (Test)")
        st.pyplot(fig)

        fig = plt.figure()
        CalibrationDisplay.from_predictions(y_test, proba, n_bins=10, strategy="quantile")
        plt.title("Calibration Curve (Test)")
        st.pyplot(fig)


# =========================
# TAB 3: SIGNIFICANT FACTORS
# =========================
with tab_factors:
    st.subheader("Significant Predictive Factors (Permutation Importance)")

    if not st.session_state.trained or st.session_state.best_pipe is None:
        st.info("Train the models first in **Model Comparison** tab.")
    else:
        X_train, X_test, y_train, y_test, proba, pred, test_metrics = st.session_state.pack
        bp = st.session_state.best_pipe

        X_test_aligned = align_to_model_columns(bp, X_test)

        with st.spinner("Computing permutation importance (this may take ~10–30s)..."):
            perm = permutation_importance(
                bp, X_test_aligned, y_test,
                n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc"
            )

        # feature names after preprocessing (one-hot expands)
        try:
            feat_names = bp.named_steps["preprocess"].get_feature_names_out()
        except Exception:
            feat_names = [f"feature_{i}" for i in range(len(perm.importances_mean))]

        # align lengths
        k = min(len(feat_names), len(perm.importances_mean))
        feat_names = feat_names[:k]
        importances = perm.importances_mean[:k]

        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=False).head(15)

        st.markdown("**Top 15 factors (drop in ROC-AUC when shuffled)**")
        st.dataframe(imp_df, use_container_width=True)

        fig = plt.figure()
        plt.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1])
        plt.title("Top 15 Significant Factors (Permutation Importance)")
        plt.xlabel("ROC-AUC drop")
        plt.tight_layout()
        st.pyplot(fig)

        st.caption("This explains which variables contribute most to the model’s prediction performance (PPS-aligned: “key predictive variables”).")


# =========================
# TAB 4: RISK SCREENING (FORM)
# =========================
with tab_screening:
    st.subheader("Individual Screening (uses your trained best model)")

    if not st.session_state.trained or st.session_state.best_pipe is None:
        st.info("Train the models first, then come back here.")
    else:
        bp = st.session_state.best_pipe

        # Use the ORIGINAL survey columns (not one-hot)
        expected_cols = list(bp.named_steps["preprocess"].feature_names_in_)
        st.write("Model expects", len(expected_cols), "inputs (survey columns).")

        # Build a simple form based on column type inferred from data
        user_row = {}
        c1, c2 = st.columns(2)

        for i, col in enumerate(expected_cols):
            target_col = c1 if i % 2 == 0 else c2
            with target_col:
                if col in X.columns:
                    # decide numeric vs categorical based on dataset dtype
                    if pd.api.types.is_numeric_dtype(X[col]):
                        user_row[col] = st.number_input(col, value=float(np.nanmedian(pd.to_numeric(X[col], errors="coerce")) if col in X.columns else 0.0))
                    else:
                        opts = sorted([str(v) for v in X[col].dropna().unique().tolist()])
                        if len(opts) == 0:
                            opts = ["No", "Yes"]
                        user_row[col] = st.selectbox(col, opts)
                else:
                    # if missing from current X (shouldn't happen due to reset), fallback
                    user_row[col] = st.text_input(col, "")

        if st.button("Predict risk"):
            X_in = pd.DataFrame([user_row])
            X_in = X_in.reindex(columns=expected_cols)

            proba = float(bp.predict_proba(X_in)[:, 1][0])
            label = "Higher Risk" if proba >= threshold else "Lower Risk"

            if label == "Higher Risk":
                st.error(f"Risk: **{label}** | Probability={proba:.2f} | Threshold={threshold:.2f}")
            else:
                st.success(f"Risk: **{label}** | Probability={proba:.2f} | Threshold={threshold:.2f}")

            st.caption("Educational screening only. Not a medical diagnosis.")


# =========================
# TAB 5: AWARENESS (PPS SUPPORT)
# =========================
with tab_awareness:
    st.subheader("Awareness & Prevention (Sri Lanka context)")
    st.write("Short quiz + practical tips (diet, lifestyle, screening).")

    q1 = st.radio("How often do you exercise (≥30 mins)?", ["Rarely", "1–2 days/week", "3–5 days/week", "Almost daily"])
    q2 = st.radio("Sweet tea / sugary drinks per day?", ["0", "1", "2", "3+"])
    q3 = st.radio("Typical rice portion?", ["Small", "Medium", "Large"])
    q4 = st.radio("Family history of diabetes?", ["No", "Yes"])
    q5 = st.radio("Sleep most nights?", ["<6 hours", "6–7 hours", "7–8 hours", "8+ hours"])
    q6 = st.radio("Fried foods per week?", ["Rarely", "1–2", "3–4", "Almost daily"])
    q7 = st.radio("Vegetables per day?", ["<2 portions", "2–3 portions", "4–5 portions", "5+"])
    q8 = st.radio("How often check blood sugar (FBS/HbA1c)?", ["Never", "Only when sick", "Once a year", "Regularly"])
    q9 = st.radio("Stress level?", ["Low", "Moderate", "High"])
    q10 = st.radio("Sitting time per day?", ["<4 hours", "4–6 hours", "6–8 hours", "8+ hours"])

    if st.button("Get my tips"):
        tips = []
        if q1 in ["Rarely", "1–2 days/week"]:
            tips.append("Add a 30-minute walk (after dinner is an easy start).")
        if q2 in ["2", "3+"]:
            tips.append("Reduce sugar in tea gradually (half → quarter → none).")
        if q3 == "Large":
            tips.append("Reduce rice portion; add more vegetables (gotukola, mukunuwenna, beans, cabbage).")
        if q4 == "Yes":
            tips.append("Family history increases risk: check FBS/HbA1c regularly.")
        if q5 == "<6 hours":
            tips.append("Aim 7–8 hours sleep to reduce insulin resistance.")
        if q6 in ["3–4", "Almost daily"]:
            tips.append("Limit fried foods; choose boiled/steamed/grilled options.")
        if q7 in ["<2 portions", "2–3 portions"]:
            tips.append("Increase vegetables to 4–5 portions/day.")
        if q8 in ["Never", "Only when sick"]:
            tips.append("Do screening at least yearly if you have risk factors.")
        if q9 == "High":
            tips.append("Try stress control: walking, breathing, prayer, relaxation.")
        if q10 in ["6–8 hours", "8+ hours"]:
            tips.append("Stand/walk 5 minutes every hour; reduce long sitting.")

        tips.append("Practical snacks: roasted gram (kadala), plain yogurt, nuts (small portion).")
        tips.append("If BP is high, reduce salt and follow medical advice.")
        tips.append("If symptoms persist (thirst/urination/fatigue/blurred vision), consult a clinic for tests.")

        st.markdown("### Your tips")
        for t in tips:
            st.write("•", t)

st.divider()
st.caption("⚠️ Disclaimer: Educational screening tool only. Not a medical diagnosis.")
