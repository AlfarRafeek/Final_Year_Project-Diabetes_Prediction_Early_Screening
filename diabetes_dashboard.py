
# IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import pickle, io, warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve,
                              ConfusionMatrixDisplay, classification_report)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rlc
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.units import cm
warnings.filterwarnings('ignore')

# PAGE CONFIG

st.set_page_config(
    page_title="Type 2 Diabetes Screening Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# GLOBAL CSS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"]        { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header          { visibility: hidden; }
.stApp                             { background: #0B0B18; }

.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(135deg,#13132a,#1a1a35);
    border-radius: 14px; padding: 6px 8px; gap: 6px;
    border: 1px solid rgba(108,92,231,.25);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; padding: 10px 26px;
    color: #666; font-size: 14px; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#6C5CE7,#a29bfe) !important;
    color: #fff !important;
    box-shadow: 0 4px 18px rgba(108,92,231,.45);
}

.hero {
    background: linear-gradient(135deg,#13132a 0%,#1a1a35 50%,#0f3460 100%);
    padding: 2.2rem 2.8rem; border-radius: 16px; margin-bottom: 1.6rem;
    border: 1px solid rgba(108,92,231,.3); position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; width: 320px; height: 320px;
    background: radial-gradient(circle,rgba(108,92,231,.18) 0%,transparent 70%);
    top: -80px; right: -60px; border-radius: 50%;
}
.hero-badge {
    display: inline-block; background: rgba(108,92,231,.18);
    border: 1px solid rgba(108,92,231,.4); color: #a29bfe;
    padding: 3px 14px; border-radius: 20px; font-size: 11px;
    font-weight: 600; letter-spacing: .06em; margin-bottom: 10px;
}
.hero-title { font-family:'Playfair Display',serif; font-size:2.3rem; color:#fff; margin:0 0 6px; line-height:1.1; }
.hero-sub   { color:#777; font-size:.95rem; font-weight:300; }

.kpi-strip  { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-bottom:1.3rem; }
.kpi        { background:linear-gradient(135deg,rgba(108,92,231,.12),rgba(108,92,231,.04));
               border:1px solid rgba(108,92,231,.28); border-radius:12px; padding:.85rem 1rem; text-align:center; }
.kpi-label  { font-size:10px; color:#666; text-transform:uppercase; letter-spacing:.06em; margin:0 0 4px; }
.kpi-val    { font-size:1.75rem; font-weight:600; color:#fff; margin:0; }
.kpi-sub    { font-size:10px; color:#555; margin:2px 0 0; }

.gc         { background:rgba(255,255,255,.03); backdrop-filter:blur(10px);
               border:1px solid rgba(255,255,255,.07); border-radius:14px;
               padding:1.3rem 1.5rem; margin-bottom:1rem; }
.gc-title   { font-size:13px; font-weight:600; color:#ccc; margin:0 0 .8rem; letter-spacing:.02em; }

.sh         { font-family:'Playfair Display',serif; font-size:1.25rem; color:#fff;
               margin:1.4rem 0 .7rem; border-left:3px solid #6C5CE7; padding-left:12px; }

.fbar       { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07);
               border-radius:12px; padding:.9rem 1.1rem; margin-bottom:1rem; }

.mc         { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07);
               border-radius:16px; padding:1.4rem 1.6rem; margin-bottom:1.3rem;
               border-top:3px solid var(--mc-color); }
.mc-name    { font-family:'Playfair Display',serif; font-size:1.1rem; color:#fff; margin:0 0 5px; }
.mc-desc    { font-size:12px; color:#888; margin:0 0 .9rem; line-height:1.6; }
.pill-row   { display:flex; flex-wrap:wrap; gap:6px; margin:.5rem 0 .7rem; }
.pill       { background:rgba(108,92,231,.18); border:1px solid rgba(108,92,231,.35);
               color:#a29bfe; font-size:10px; padding:2px 9px; border-radius:10px; }
.mpill      { background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
               border-radius:8px; padding:4px 12px; font-size:12px; color:#ddd; }
.mpill b    { color:#a29bfe; }
.best-tag   { display:inline-block; background:linear-gradient(135deg,#00b894,#55efc4);
               color:#004d40; font-size:10px; font-weight:700; padding:2px 10px;
               border-radius:10px; margin-left:8px; }

.rcard      { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.1);
               border-radius:16px; padding:2rem; text-align:center; }
.r-high     { border-top:4px solid #e17055; }
.r-medium   { border-top:4px solid #fdcb6e; }
.r-low      { border-top:4px solid #00b894; }
.r-pct      { font-family:'Playfair Display',serif; font-size:3.8rem; line-height:1; margin:8px 0; }
.r-label    { font-size:.95rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; margin:0; }

.fb         { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.07);
               border-radius:10px; padding:.85rem 1.1rem; margin-bottom:7px; }
.fb-top     { display:flex; justify-content:space-between; margin-bottom:5px; }
.fb-name    { font-weight:500; font-size:13px; color:#ddd; }
.fb-bg      { background:rgba(255,255,255,.08); border-radius:5px; height:8px; }
.fb-note    { font-size:11px; color:#666; margin-top:4px; }

.tip        { border-radius:10px; padding:.9rem 1.1rem; margin-bottom:8px; border-left:4px solid; }
.tip-r      { background:rgba(225,112,85,.1);  border-color:#e17055; }
.tip-a      { background:rgba(253,203,110,.08); border-color:#fdcb6e; }
.tip-g      { background:rgba(0,184,148,.08);   border-color:#00b894; }
.tip-h      { font-size:13px; font-weight:600; margin:0 0 4px; }
.tip-b      { font-size:12px; color:#999; margin:0; line-height:1.6; }

.chips      { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }
.chip       { background:rgba(108,92,231,.18); color:#a29bfe; padding:4px 12px; border-radius:20px; font-size:11px; }
.disc       { background:rgba(253,203,110,.08); border:1px solid rgba(253,203,110,.25);
               border-radius:8px; padding:8px 12px; font-size:11px; color:#fdcb6e; margin-top:10px; }

.risk-info-box {
    background: rgba(108,92,231,.08); border: 1px solid rgba(108,92,231,.3);
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
}
.risk-step {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: .8rem;
    border-left: 4px solid #6C5CE7;
}

p, label, span, div { color: #ccc; }
h1,h2,h3,h4 { color: #fff; }
</style>
""", unsafe_allow_html=True)

# CONSTANTS & MAPPINGS

C_YES  = '#e17055'
C_NO   = '#00b894'
PAL    = {'Yes': C_YES, 'No': C_NO}
PURPLE = '#6C5CE7'

AGE_ORD  = ['20 - 29','30 - 39','40 - 49','50 - 59','60 and above']
EX_ORD   = ['Never','1 - 3 times','3 - 6 times','Daily']
BP_ORD   = ['Normal','Elevated','High Stage 1','High Stage 2']
SL_ORD   = ['0-5 hours','5-6 hours','6-7 hours','7-8 hours','8 hours and above']
BMI_ORD  = ['Underweight','Normal','Overweight','Obese']
W_ORD    = ['Below 50 kg','50 - 59 kg','60 - 69 kg','70 - 79 kg','80 and above']

HM = {'Below 150 cm':145,'150 - 159 cm':155,'160 - 169 cm':165,'170 - 179 cm':175,'180 cm and above':183}
WM = {'Below 50 kg':47,'50 - 59 kg':55,'60 - 69 kg':65,'70 - 79 kg':75,'80 and above':87}

FEAT_COLS = ['age_enc','gender_enc','occupation_enc','height_enc','weight_enc','bmi',
             'bmi_cat_enc','waist_cm','systolic_bp','diastolic_bp','bp_cat_enc',
             'exercise_enc','diet_enc','sleep_enc','family_history_enc','bp_cholesterol_enc',
             'frequent_urination_bin','unusual_thirst_bin','weight_change_bin',
             'fatigue_bin','blurred_vision_bin','symptom_score']

FEAT_LBL = {
    'age_enc':'Age','bp_cholesterol_enc':'BP/Chol history','symptom_score':'Symptom score',
    'blurred_vision_bin':'Blurred vision','systolic_bp':'Systolic BP','fatigue_bin':'Fatigue',
    'bmi':'BMI','unusual_thirst_bin':'Unusual thirst','sleep_enc':'Sleep quality',
    'diastolic_bp':'Diastolic BP','exercise_enc':'Exercise frequency','waist_cm':'Waist circumference',
    'family_history_enc':'Family history','bmi_cat_enc':'BMI category',
    'frequent_urination_bin':'Frequent urination','bp_cat_enc':'BP category',
    'weight_enc':'Weight range','height_enc':'Height range','diet_enc':'Diet type',
    'gender_enc':'Gender','occupation_enc':'Occupation','weight_change_bin':'Weight change',
}
BMI_LBL = {0:'Underweight',1:'Normal',2:'Overweight',3:'Obese'}
BP_LBL  = {0:'Normal',1:'Elevated',2:'High Stage 1',3:'High Stage 2'}

MODEL_INFO = {
    'Logistic Regression': {
        'color':'#74b9ff',
        'desc':'Linear probabilistic classifier. Uses sigmoid function on weighted sum of features. Simple, fast, interpretable. Requires feature scaling.',
        'params':'C=1 · solver=lbfgs · penalty=l2 · class_weight=balanced',
        'why':'Best hyperparams from RandomizedSearchCV (n_iter=30, scoring=F1, 5-fold CV)',
        'acc':87.27,'pre':89.19,'rec':91.67,'f1':90.41,'auc':91.23,'cv_m':88.21,'cv_s':2.31,
        'cm':[[15,4],[3,33]],'scaled':True,
    },
    'SVM (RBF)': {
        'color':'#a29bfe',
        'desc':'Maximum-margin classifier with RBF kernel. Maps to high-dimensional space for non-linear boundaries. High precision but lower recall.',
        'params':'C=100 · gamma=scale · kernel=rbf · class_weight=balanced',
        'why':'High C allows more misclassification in training for wider generalisation boundary.',
        'acc':80.0,'pre':93.1,'rec':75.0,'f1':83.08,'auc':94.44,'cv_m':92.14,'cv_s':3.37,
        'cm':[[17,2],[9,27]],'scaled':True,
    },
    'Random Forest': {
        'color':'#55efc4',
        'desc':'Ensemble of decision trees via bagging. Each tree trained on bootstrap sample with random feature subset. Majority vote for prediction.',
        'params':'n_estimators=100 · max_depth=20 · min_samples_split=10 · max_features=sqrt · class_weight=balanced',
        'why':'max_depth=20 allows moderately deep trees; sqrt features reduces correlation between trees.',
        'acc':85.45,'pre':91.18,'rec':86.11,'f1':88.57,'auc':93.27,'cv_m':93.24,'cv_s':1.70,
        'cm':[[16,3],[5,31]],'scaled':False,
    },
    'Extra Trees': {
        'color':'#fd79a8',
        'desc':'Extremely Randomised Trees. Uses random split thresholds (not optimal). More variance reduction, faster. No bootstrap sampling.',
        'params':'n_estimators=300 · max_depth=10 · min_samples_split=2 · max_features=log2 · class_weight=balanced',
        'why':'300 trees with log2 features for higher randomisation; max_depth=10 prevents overfitting.',
        'acc':87.27,'pre':91.43,'rec':88.89,'f1':90.14,'auc':93.57,'cv_m':93.37,'cv_s':3.52,
        'cm':[[16,3],[4,32]],'scaled':False,
    },
    'HistGradient Boosting': {
        'color':'#ffeaa7',
        'desc':'Sequential boosting with histogram binning (LightGBM-inspired). Each tree corrects previous errors. Handles imbalance natively. Best overall.',
        'params':'learning_rate=0.1 · max_depth=7 · max_iter=200 · min_samples_leaf=30',
        'why':'learning_rate=0.1 with max_iter=200 balances underfitting/overfitting. min_samples_leaf=30 prevents noisy leaf splits.',
        'acc':89.09,'pre':94.12,'rec':88.89,'f1':91.43,'auc':95.32,'cv_m':95.10,'cv_s':3.22,
        'cm':[[17,2],[4,32]],'scaled':False,
    },
}

plt.rcParams.update({
    'figure.facecolor':'none','axes.facecolor':'#0f0f22',
    'axes.edgecolor':'#1e1e3a','axes.labelcolor':'#aaa',
    'xtick.color':'#666','ytick.color':'#666','text.color':'#ccc',
    'grid.color':'#1a1a35','grid.linewidth':0.4,
    'axes.spines.top':False,'axes.spines.right':False,
})


# GLOBAL HELPER FUNCTIONS
def render_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def pct_bar(ax, data, col, order, title, rot=20, xlabels=None):
    ct  = data.groupby([col,'diabetes_diagnosis']).size().unstack(fill_value=0)
    ct  = ct.reindex(order, fill_value=0)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    avail = [c for c in ['No','Yes'] if c in pct.columns]
    pct[avail].plot(kind='bar', ax=ax, color=[PAL[c] for c in avail],
                    edgecolor='none', width=.7)
    ax.set_xticklabels(xlabels if xlabels else order, rotation=rot, ha='right', fontsize=8)
    ax.set_ylabel('%', fontsize=9); ax.set_xlabel('')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0f}%'))
    ax.legend(title='Diabetic', fontsize=8, title_fontsize=8)

def count_bar(ax, data, col, order, title, rot=20, xlabels=None):
    ct    = data.groupby([col,'diabetes_diagnosis']).size().unstack(fill_value=0)
    ct    = ct.reindex(order, fill_value=0)
    avail = [c for c in ['No','Yes'] if c in ct.columns]
    ct[avail].plot(kind='bar', ax=ax, color=[PAL[c] for c in avail],
                   edgecolor='none', width=.7)
    ax.set_xticklabels(xlabels if xlabels else order, rotation=rot, ha='right', fontsize=8)
    ax.set_ylabel('Count', fontsize=9); ax.set_xlabel('')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.legend(title='Diabetic', fontsize=8, title_fontsize=8)

def box_pair(ax, data_df, col, title, ylabel):
    nd = data_df[data_df['diabetes_diagnosis']=='No'][col].dropna()
    yd = data_df[data_df['diabetes_diagnosis']=='Yes'][col].dropna()
    bp = ax.boxplot([nd, yd], patch_artist=True,
                    labels=['Non-Diabetic','Diabetic'],
                    medianprops={'color':'white','linewidth':2},
                    whiskerprops={'color':'#555'},
                    capprops={'color':'#555'},
                    flierprops={'marker':'o','markerfacecolor':'#444','markersize':3})
    bp['boxes'][0].set_facecolor(C_NO + '88')
    bp['boxes'][1].set_facecolor(C_YES + '88')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

def hist_pair(ax, data_df, col, title, xlabel, bins=18, vlines=None):
    for lbl, color in PAL.items():
        sub = data_df[data_df['diabetes_diagnosis']==lbl][col].dropna()
        if len(sub): ax.hist(sub, bins=bins, alpha=.72, color=color, label=lbl, edgecolor='none')
    if vlines:
        for val, lbl, clr in vlines:
            ax.axvline(val, color=clr, linestyle='--', lw=1.6, label=lbl)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel('Count', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.legend(fontsize=8)

def mini_bar(ax, pct, xlabel, title):
    """Simple percentage bar chart for risk factor deep-dive section."""
    avail = [c for c in ['No','Yes'] if c in pct.columns]
    pct[avail].plot(kind='bar', ax=ax, color=[PAL[c] for c in avail],
                    edgecolor='none', width=.65)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('%', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0f}%'))
    ax.legend(title='Diabetic', fontsize=8)
    ax.tick_params(labelsize=8)

def compute_p_values(df, target_col='diabetes_diagnosis'):
    results = []

    for col in df.columns:
        if col == target_col:
            continue

        try:
            # Numerical → t-test
            if pd.api.types.is_numeric_dtype(df[col]):
                group1 = df[df[target_col] == 'Yes'][col].dropna()
                group0 = df[df[target_col] == 'No'][col].dropna()

                if len(group1) > 1 and len(group0) > 1:
                    stat, p = ttest_ind(group1, group0, equal_var=False)
                else:
                    p = 1.0

            # Categorical → Chi-square
            else:
                contingency = pd.crosstab(df[col], df[target_col])
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    chi2, p, _, _ = chi2_contingency(contingency)
                else:
                    p = 1.0

            results.append((col, p))

        except:
            results.append((col, 1.0))

    return pd.DataFrame(results, columns=['Feature', 'p_value']).sort_values('p_value')

# CACHED LOADERS

@st.cache_resource
def load_model_and_fi():
    try:
        with open('best_model_final.pkl','rb') as f: model = pickle.load(f)
        with open('feat_imp_final.pkl','rb') as f:  fi    = pickle.load(f)
    except FileNotFoundError:
        df = pd.read_csv('diabetes_cleaned_v2.csv')
        X  = df.drop('diabetes_diagnosis_enc',axis=1)
        y  = df['diabetes_diagnosis_enc']
        Xtr,_,ytr,_ = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
        model = HistGradientBoostingClassifier(
            learning_rate=0.1,max_depth=7,max_iter=200,min_samples_leaf=30,random_state=42)
        model.fit(Xtr,ytr)
        et = ExtraTreesClassifier(n_estimators=300,max_depth=10,
                                   class_weight='balanced',random_state=42)
        et.fit(Xtr,ytr)
        fi = dict(zip(X.columns.tolist(), et.feature_importances_.tolist()))
    return model, fi

@st.cache_data
def load_full_df():
    return pd.read_csv('diabetes_full_cleaned_v2.csv')

best_model, feat_imp_dict = load_model_and_fi()
feat_imp   = pd.Series(feat_imp_dict)
df_default = load_full_df()


# HERO BANNER
st.markdown("""
<div class="hero">
  <span class="hero-badge">🩺 FINAL YEAR PROJECT · KATTANKUDY BASE HOSPITAL</span>
  <h1 class="hero-title">Type 2 Diabetes Early Screening Dashboard</h1>
  <p class="hero-sub">
    Risk Prediction &amp; PDF Leaflet &nbsp;·&nbsp;
    Significant Risk Factors &nbsp;·&nbsp;
    EDA Explorer &nbsp;·&nbsp;
    5-Model ML Comparison with Hyperparameter Tuning &nbsp;|&nbsp;
    <span style="color:#a29bfe">HistGradient Boosting · 89.1% Accuracy · 95.3% ROC-AUC</span>
  </p>
</div>
""", unsafe_allow_html=True)


# TABS  — NEW ORDER
tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction & Resources",
    "Risk Factors",
    "EDA Explorer",
    "ML Models & Tuning",
])


# TAB 1 — PREDICTION & RESOURCES
with tab1:
    st.markdown('<p class="sh">🩺 Diabetes Risk Screening Questionnaire</p>', unsafe_allow_html=True)
    st.info("Complete all 6 sections and click **Calculate My Risk** to get your personalised risk score, factor breakdown, health tips, and downloadable PDF report.")

    with st.form("risk_form"):
        st.markdown("#### Section 1 — Demographics")
        qa, qb, qc = st.columns(3)
        age        = qa.selectbox("Age group *",  AGE_ORD)
        gender     = qb.radio("Gender *", ["Male","Female"], horizontal=True)
        occupation = qc.selectbox("Occupation *", ["Employed","Unemployed","Student","Retired"])

        st.markdown("#### Section 2 — Physical Measurements")
        qd, qe, qf = st.columns(3)
        height = qd.selectbox("Height *",  list(HM.keys()))
        weight = qe.selectbox("Weight *",  list(WM.keys()))
        waist  = qf.slider("Waist circumference (cm) *", 60, 140, 88)

        st.markdown("#### Section 3 — Blood Pressure")
        qg, qh = st.columns(2)
        systolic  = qg.slider("Systolic BP (mmHg) *",  80, 190, 125)
        diastolic = qh.slider("Diastolic BP (mmHg) *", 50, 125,  83)

        st.markdown("#### Section 4 — Lifestyle")
        qi, qj, qk = st.columns(3)
        exercise = qi.selectbox("Exercise per week *", EX_ORD)
        diet     = qj.selectbox("Diet type *", ["Mostly unhealthy","Mixed","Mostly healthy"])
        sleep    = qk.selectbox("Sleep hours per night *", SL_ORD)

        st.markdown("#### Section 5 — Medical History")
        qm1, qm2 = st.columns(2)
        family_hist = qm1.checkbox("Family history of diabetes")
        bp_chol     = qm2.checkbox("Previously diagnosed with high BP or cholesterol")

        st.markdown("#### Section 6 — Current Symptoms")
        qs1, qs2, qs3, qs4, qs5 = st.columns(5)
        freq_urine  = qs1.checkbox("Frequent urination")
        thirst      = qs2.checkbox("Unusual thirst")
        weight_chng = qs3.checkbox("Weight change")
        fatigue_s   = qs4.checkbox("Fatigue")
        blurred     = qs5.checkbox("Blurred vision")

        submitted = st.form_submit_button("Calculate My Risk", use_container_width=True)

    if submitted:
        # Step 1: Encode all inputs
        h_m = HM[height]; w_m = WM[weight]
        bmi     = round(w_m / ((h_m/100)**2), 1)
        bmi_cat = 0 if bmi<18.5 else 1 if bmi<25 else 2 if bmi<30 else 3
        if   systolic<120 and diastolic<80: bp_cat=0
        elif systolic<130 and diastolic<80: bp_cat=1
        elif systolic<140 or  diastolic<90: bp_cat=2
        else:                               bp_cat=3
        sv = [int(freq_urine),int(thirst),int(weight_chng),int(fatigue_s),int(blurred)]
        ss = sum(sv)

        inp = {
            'age_enc':               {'20 - 29':1,'30 - 39':2,'40 - 49':3,'50 - 59':4,'60 and above':5}[age],
            'gender_enc':            0 if gender=="Female" else 1,
            'occupation_enc':        {"Employed":0,"Retired":1,"Student":2,"Unemployed":3}[occupation],
            'height_enc':            {'Below 150 cm':1,'150 - 159 cm':2,'160 - 169 cm':3,'170 - 179 cm':4,'180 cm and above':5}[height],
            'weight_enc':            {'Below 50 kg':1,'50 - 59 kg':2,'60 - 69 kg':3,'70 - 79 kg':4,'80 and above':5}[weight],
            'bmi':                   bmi, 'bmi_cat_enc':bmi_cat, 'waist_cm':waist,
            'systolic_bp':           systolic, 'diastolic_bp':diastolic, 'bp_cat_enc':bp_cat,
            'exercise_enc':          {'Never':0,'1 - 3 times':1,'3 - 6 times':2,'Daily':3}[exercise],
            'diet_enc':              {'Mostly unhealthy':0,'Mixed':1,'Mostly healthy':2}[diet],
            'sleep_enc':             {'0-5 hours':1,'5-6 hours':2,'6-7 hours':3,'7-8 hours':4,'8 hours and above':5}[sleep],
            'family_history_enc':    int(family_hist),
            'bp_cholesterol_enc':    int(bp_chol),
            'frequent_urination_bin':int(freq_urine),
            'unusual_thirst_bin':    int(thirst),
            'weight_change_bin':     int(weight_chng),
            'fatigue_bin':           int(fatigue_s),
            'blurred_vision_bin':    int(blurred),
            'symptom_score':         ss,
        }

        # Step 2: Predict
        Xi   = pd.DataFrame([inp])[FEAT_COLS]
        prob = best_model.predict_proba(Xi)[0][1]
        pct  = round(prob*100, 1)

        if   pct>=70: risk,card,rc,emoji="HIGH RISK","r-high","#e17055","⚠️"
        elif pct>=40: risk,card,rc,emoji="MODERATE RISK","r-medium","#fdcb6e","🔶"
        else:         risk,card,rc,emoji="LOW RISK","r-low","#00b894","✅"

        adv = {
            "HIGH RISK":     "Please consult a healthcare professional immediately for blood glucose testing.",
            "MODERATE RISK": "Consider lifestyle improvements and discuss results with a doctor soon.",
            "LOW RISK":      "Your profile suggests low current risk. Maintain healthy habits.",
        }[risk]

        # ── Display: Risk card + Factor breakdown ────────────────────────────
        res_col, fac_col = st.columns([1, 1.6], gap="large")

        with res_col:
            st.markdown(f"""
            <div class="rcard {card}">
              <p style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:.05em">DIABETES RISK SCORE</p>
              <p class="r-pct" style="color:{rc}">{pct}%</p>
              <p class="r-label" style="color:{rc}">{emoji} {risk}</p>
              <hr style="border:none;border-top:1px solid #222;margin:12px 0">
              <p style="font-size:12px;color:#888;line-height:1.6">{adv}</p>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<br>
            <div class="chips">
              <span class="chip">BMI {bmi} — {BMI_LBL[bmi_cat]}</span>
              <span class="chip">BP {BP_LBL[bp_cat]}</span>
              <span class="chip">{ss}/5 symptoms</span>
              <span class="chip">{'Family Hx ✓' if family_hist else 'No Family Hx'}</span>
              <span class="chip">Exercise: {exercise}</span>
              <span class="chip">Sleep: {sleep}</span>
            </div>
            <div class="disc">⚠️ <b>Disclaimer:</b> For educational &amp; research purposes only.
            Not a clinical diagnosis. Always consult a qualified healthcare professional.</div>
            """, unsafe_allow_html=True)

        with fac_col:
            st.markdown('<p class="sh" style="font-size:1.1rem">Your Personal Risk Factor Breakdown</p>',
                        unsafe_allow_html=True)
            top_fi = feat_imp[list(FEAT_LBL.keys())[:16]].sort_values(ascending=False)
            mx     = top_fi.max()

            def get_nc(feat, val):
                risk_up = {'bp_cholesterol_enc','blurred_vision_bin','fatigue_bin',
                           'unusual_thirst_bin','family_history_enc','frequent_urination_bin'}
                if feat in risk_up:
                    return ("Present ⚠️","#e17055") if val else ("Not reported","#00b894")
                if feat=='bmi':         return f"{bmi} ({BMI_LBL[bmi_cat]})", ("#e17055" if bmi>=25 else "#00b894")
                if feat=='age_enc':     return age, ("#e17055" if val>=3 else "#00b894")
                if feat=='symptom_score': return f"{val}/5", ("#e17055" if val>=2 else "#00b894")
                if feat=='exercise_enc':  return exercise, ("#00b894" if val>=1 else "#e17055")
                if feat=='sleep_enc':     return sleep, ("#00b894" if val>=3 else "#e17055")
                if feat=='systolic_bp':   return f"{systolic} mmHg", ("#e17055" if systolic>=130 else "#00b894")
                if feat=='waist_cm':      return f"{waist} cm", ("#e17055" if waist>=88 else "#00b894")
                if feat=='bp_cat_enc':    return BP_LBL.get(bp_cat,'—'), ("#e17055" if bp_cat>=2 else "#00b894")
                return str(val), "#888"

            for feat, imp in top_fi.items():
                if feat not in inp: continue
                val       = inp[feat]
                note, col = get_nc(feat, val)
                bp2       = round((imp/mx)*100, 1)
                st.markdown(f"""
                <div class="fb">
                  <div class="fb-top">
                    <span class="fb-name">{FEAT_LBL.get(feat,feat)}</span>
                    <span style="font-size:10px;color:#444">{imp*100:.1f}% importance</span>
                  </div>
                  <div class="fb-bg">
                    <div style="height:8px;width:{bp2}%;background:{col};border-radius:5px"></div>
                  </div>
                  <p class="fb-note">Your value: <b style="color:{col}">{note}</b></p>
                </div>""", unsafe_allow_html=True)

        # Symptoms chart + Health Tips
        st.markdown("---")
        sc_col, tip_col = st.columns(2)

        with sc_col:
            st.markdown('<p class="sh" style="font-size:1rem">Your Reported Symptoms</p>', unsafe_allow_html=True)
            sn = ['Freq.\nUrination','Unusual\nThirst','Weight\nChange','Fatigue','Blurred\nVision']
            sc = ['#e17055' if v else '#1e1e3a' for v in sv]
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            ax.bar(sn, sv, color=sc, edgecolor='none', width=.5)
            ax.set_ylim(0,1.5); ax.set_yticks([0,1]); ax.set_yticklabels(['No','Yes'])
            ax.set_title(f"Symptoms Reported: {ss}/5", fontsize=11, fontweight='bold')
            ax.legend(handles=[mpatches.Patch(color='#e17055',label='Reported'),
                                mpatches.Patch(color='#1e1e3a',label='Not reported')], fontsize=9)
            plt.tight_layout(); render_fig(fig)

        with tip_col:
            st.markdown('<p class="sh" style="font-size:1rem">💡 Personalised Health Tips</p>', unsafe_allow_html=True)
            tips = []
            if   exercise=="Never":       tips.append(("r","🏃 Start Exercising","30 min brisk walk 3× per week reduces risk significantly. Aim for 150 min/week of moderate activity."))
            elif exercise=="1 - 3 times": tips.append(("a","🏃 Increase Exercise","Good start! Try increasing to 4–5 sessions per week for maximum metabolic benefit."))
            else:                          tips.append(("g","🏃 Keep It Up!","Excellent exercise habits — one of the strongest protective factors against diabetes."))
            if   bmi>=30: tips.append(("r","⚖️ Weight Management","BMI is in the obese range. Even a 5–10% weight reduction dramatically lowers diabetes risk."))
            elif bmi>=25: tips.append(("a","⚖️ Healthy Weight","Slightly overweight. Balanced diet and regular exercise can help reach a healthy BMI of 18.5–24.9."))
            else:          tips.append(("g","⚖️ Healthy Weight","BMI is in the healthy range. Maintain it with balanced nutrition and regular activity."))
            if   sleep in ["0-5 hours","5-6 hours"]: tips.append(("r","😴 Improve Sleep","Poor sleep is linked to insulin resistance. Aim for 7–8 hours per night consistently."))
            elif sleep=="6-7 hours":                  tips.append(("a","😴 Sleep Quality","Slightly below optimal. Try reaching 7–8 hours for best metabolic health."))
            else:                                      tips.append(("g","😴 Good Sleep","Excellent sleep duration — important for insulin sensitivity and overall health."))
            if   bp_cat>=2: tips.append(("r","💓 Blood Pressure","BP is elevated. Reduce sodium intake, increase potassium-rich foods, and consult a doctor."))
            elif bp_cat==1: tips.append(("a","💓 Blood Pressure","Slightly elevated. Reduce processed foods, increase activity, monitor regularly."))
            else:            tips.append(("g","💓 Blood Pressure","Blood pressure is in a healthy range. Maintain a low-sodium, heart-healthy diet."))
            if family_hist: tips.append(("a","🧬 Family History","Genetic risk is present. Annual blood glucose screening is strongly recommended."))
            if ss>=3:       tips.append(("r","🚨 Multiple Symptoms",f"You reported {ss}/5 diabetes symptoms. Please seek medical attention promptly — do not delay."))

            tip_cls = {"r":"tip-r","a":"tip-a","g":"tip-g"}
            tip_clr = {"r":"#e17055","a":"#fdcb6e","g":"#00b894"}
            for lvl, title, body in tips[:5]:
                st.markdown(f"""
                <div class="tip {tip_cls[lvl]}">
                  <p class="tip-h" style="color:{tip_clr[lvl]}">{title}</p>
                  <p class="tip-b">{body}</p>
                </div>""", unsafe_allow_html=True)

        # PDF Leaflet
        st.markdown("---")
        st.markdown('<p class="sh" style="font-size:1rem">📄 Download Your Health Leaflet</p>', unsafe_allow_html=True)
        st.write("Get a printable A4 PDF with your risk score, full profile, personalised tips, and model information.")

        def build_pdf():
            buf    = io.BytesIO()
            doc    = SimpleDocTemplate(buf, pagesize=A4,
                                        leftMargin=2*cm, rightMargin=2*cm,
                                        topMargin=2*cm, bottomMargin=2*cm)
            DARK   = rlc.HexColor('#1A1A2E'); PURPLE2=rlc.HexColor('#6C5CE7')
            RD     = rlc.HexColor('#e17055'); GR=rlc.HexColor('#00b894')
            AM     = rlc.HexColor('#BA7517'); GRAY=rlc.HexColor('#888888')
            rc2    = {'HIGH RISK':RD,'MODERATE RISK':AM,'LOW RISK':GR}[risk]
            ts     = ParagraphStyle('t',fontSize=22,textColor=DARK,fontName='Helvetica-Bold',spaceAfter=4)
            ss2    = ParagraphStyle('s',fontSize=11,textColor=GRAY,fontName='Helvetica',spaceAfter=12)
            h2     = ParagraphStyle('h2',fontSize=13,textColor=DARK,fontName='Helvetica-Bold',spaceAfter=6,spaceBefore=8)
            body_s = ParagraphStyle('b',fontSize=10,textColor=DARK,fontName='Helvetica',spaceAfter=4,leading=14)
            story  = []
            story.append(Paragraph("Diabetes Risk Screening Report", ts))
            story.append(Paragraph("Type 2 Diabetes Early Screening · HistGradient Boosting · Leakage-Free", ss2))
            story.append(HRFlowable(width="100%",thickness=1,color=PURPLE2,spaceAfter=12))
            story.append(Paragraph(f"{pct}%", ParagraphStyle('p',fontSize=32,textColor=rc2,fontName='Helvetica-Bold',alignment=1,spaceAfter=4)))
            story.append(Paragraph(f"{emoji} {risk}", ParagraphStyle('l',fontSize=14,textColor=rc2,fontName='Helvetica-Bold',alignment=1,spaceAfter=4)))
            story.append(Paragraph(adv, ParagraphStyle('a',fontSize=10,textColor=DARK,fontName='Helvetica',alignment=1,spaceAfter=16,leading=15)))
            story.append(HRFlowable(width="100%",thickness=0.5,color=rlc.lightgrey,spaceAfter=10))
            story.append(Paragraph("Patient Profile", h2))
            profile = [
                ["Field","Value","Field","Value"],
                ["Age group",age,"Gender",gender],
                ["Occupation",occupation,"BMI",f"{bmi} ({BMI_LBL[bmi_cat]})"],
                ["Waist",f"{waist} cm","BP category",BP_LBL[bp_cat]],
                ["Exercise",exercise,"Sleep",sleep],
                ["Symptom score",f"{ss}/5","Family history","Yes" if family_hist else "No"],
                ["BP/Chol history","Yes" if bp_chol else "No","Systolic BP",f"{systolic} mmHg"],
            ]
            t = Table(profile, colWidths=[3.8*cm,4.5*cm,3.8*cm,4.5*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),PURPLE2),('TEXTCOLOR',(0,0),(-1,0),rlc.white),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[rlc.HexColor('#F7F5F0'),rlc.white]),
                ('GRID',(0,0),(-1,-1),0.5,rlc.HexColor('#DDDDDD')),('PADDING',(0,0),(-1,-1),6),
            ]))
            story.append(t); story.append(Spacer(1,12))
            story.append(Paragraph("Personalised Health Tips", h2))
            story.append(HRFlowable(width="100%",thickness=0.5,color=rlc.lightgrey,spaceAfter=8))
            tip_colors = {"r":RD,"a":AM,"g":GR}
            for lvl, title_t, tbody in tips:
                story.append(Paragraph(title_t, ParagraphStyle('th',fontSize=10,textColor=tip_colors[lvl],fontName='Helvetica-Bold',spaceAfter=2,spaceBefore=6)))
                story.append(Paragraph(tbody, ParagraphStyle('tb',fontSize=9.5,textColor=DARK,fontName='Helvetica',spaceAfter=3,leading=14,leftIndent=12)))
            story.append(Spacer(1,12))
            story.append(HRFlowable(width="100%",thickness=0.5,color=rlc.lightgrey,spaceAfter=8))
            story.append(Spacer(1,10))
            story.append(Paragraph(
                "DISCLAIMER: This report is generated by a machine learning model for educational "
                "and research purposes only. It does not constitute a medical diagnosis. "
                "Always consult a qualified healthcare professional for clinical advice.",
                ParagraphStyle('d',fontSize=8.5,textColor=AM,fontName='Helvetica-Oblique',leading=13)))
            doc.build(story)
            buf.seek(0); return buf.read()

        pdf_bytes = build_pdf()
        st.download_button(
            label="⬇️ Download PDF Health Leaflet",
            data=pdf_bytes,
            file_name=f"diabetes_risk_report_{risk.lower().replace(' ','_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown('<p class="sh" style="font-size:.95rem">Model Performance · HistGradient Boosting · Test Set (n=55)</p>', unsafe_allow_html=True)
        m1,m2,m3,m4,m5 = st.columns(5)
        for col, lbl, val, delta in zip([m1,m2,m3,m4,m5],
            ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
            ["89.09%","94.12%","88.89%","91.43%","95.32%"],
            ["Leakage-free","High precision","Good recall","Best balance","Excellent AUC"]):
            col.metric(lbl, val, delta)

    else:
        st.markdown("""
        <div class="gc" style="text-align:center;padding:3rem">
          <p style="font-size:2.5rem;margin:0">🔬</p>
          <p style="font-size:14px;color:#555;margin:10px 0 0">
            Complete the questionnaire above and click
            <b style="color:#a29bfe">Calculate My Risk</b> to see your results.
          </p>
        </div>""", unsafe_allow_html=True)

# TAB 2 — RISK FACTORS
with tab2:
    st.markdown('<p class="sh">What Drives Diabetes Risk in This Dataset?</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="gc">
    <p style="font-size:13px;color:#aaa;line-height:1.7;margin:0">
    The charts below are derived from the <b style="color:#a29bfe">leakage-free</b> dataset
    (Timestamp and <code>on_medication</code> removed).
    Feature importances come from a trained <b style="color:#a29bfe">Extra Trees Classifier</b>
    (used as a proxy because HistGradient Boosting does not expose importances directly).
    Correlations are <b>Pearson coefficients</b> with the binary target (1=Diabetic, 0=Not Diabetic).
    Data source: <b style="color:#a29bfe">Kattankudy Base Hospital patient survey (n=271)</b>.
    </p></div>
    """, unsafe_allow_html=True)

    df_model = pd.read_csv("diabetes_cleaned_v2.csv")
    df_full  = df_default.copy()

    # Feature importance + Correlation
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="gc"><p class="gc-title">Top 15 Feature Importances (Extra Trees)</p>', unsafe_allow_html=True)
        top15      = feat_imp.sort_values(ascending=True).tail(15)
        labels_15  = [FEAT_LBL.get(f, f) for f in top15.index]  
        fig, ax    = plt.subplots(figsize=(6, 5.5))
        bars       = ax.barh(labels_15, top15.values*100, color='#6C5CE7',
                             edgecolor='none', height=.65)
        for bar, val in zip(bars, top15.values):
            ax.text(val*100+.15, bar.get_y()+bar.get_height()/2,
                    f'{val*100:.1f}%', va='center', fontsize=8, color='#ccc')
        ax.set_xlabel('Importance (%)', fontsize=9)
        ax.set_title('Higher = more influential in prediction', fontsize=9, color='#888')
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="gc"><p class="gc-title">Feature Correlation with Diabetes Diagnosis</p>', unsafe_allow_html=True)
        target_corr = df_model.corr(numeric_only=True)['diabetes_diagnosis_enc'].drop('diabetes_diagnosis_enc').sort_values()
        corr_labels = [FEAT_LBL.get(f, f) for f in target_corr.index]
        bar_colors  = [C_YES if v>0 else '#378ADD' for v in target_corr.values]
        fig, ax     = plt.subplots(figsize=(6, 5.5))
        ax.barh(corr_labels, target_corr.values, color=bar_colors,
                edgecolor='none', height=.65)
        ax.axvline(0, color='gray', lw=.8)
        ax.set_xlabel('Pearson r', fontsize=9)
        ax.set_title('Red = risk factor  |  Blue = protective', fontsize=9, color='#888')
        ax.legend(handles=[
            mpatches.Patch(color=C_YES, label='Positive (risk factor)'),
            mpatches.Patch(color='#378ADD', label='Negative (protective)')
        ], fontsize=8, loc='lower right')
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)



 


# TAB 3 — EDA EXPLORER
with tab3:
    # Upload
    st.markdown('<p class="sh"> Dataset</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload diabetes_full_cleaned_v2.csv · or use the default dataset loaded below",
        type=["csv"], help="Must be the cleaned v2 dataset produced in the preprocessing step.")
    df = pd.read_csv(uploaded) if uploaded else df_default.copy()
    st.caption(f"{' Uploaded file' if uploaded else ' Default dataset'} · "
               f"**{len(df)} rows** · {df.shape[1]} columns")

    # Filters
    st.markdown('<p class="sh">Filters — All Charts Update Automatically</p>', unsafe_allow_html=True)
    st.markdown('<div class="fbar">', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    f_age  = c1.multiselect("Age group",  AGE_ORD,  AGE_ORD,         key="f1")
    f_gen  = c2.multiselect("Gender",  ["Male","Female"],["Male","Female"], key="f2")
    f_bmi  = c3.multiselect("BMI category",BMI_ORD, BMI_ORD,         key="f3")
    f_bp   = c4.multiselect("BP category", BP_ORD,  BP_ORD,          key="f4")
    f_diag = c5.multiselect("Diagnosis", ["Yes","No"],["Yes","No"],   key="f5")
    f_ex   = c6.multiselect("Exercise",  EX_ORD,  EX_ORD,            key="f6")
    st.markdown('</div>', unsafe_allow_html=True)

    d = df[
        df['Age'].isin(f_age) & df['Gender'].isin(f_gen) &
        df['bmi_category'].isin(f_bmi) & df['bp_category'].isin(f_bp) &
        df['diabetes_diagnosis'].isin(f_diag) & df['exercise_freq'].isin(f_ex)
    ].copy()

    if len(d) == 0:
        st.warning(" No rows match the selected filters. Please adjust your selections.")
        st.stop()
    st.caption(f"Filtered dataset: **{len(d)} rows**")

    # KPI strip
    st.markdown('<p class="sh"> Overview</p>', unsafe_allow_html=True)
    total = len(d); yes_n = (d['diabetes_diagnosis']=='Yes').sum(); no_n = total-yes_n
    st.markdown(f"""
    <div class="kpi-strip">
      <div class="kpi"><p class="kpi-label">Total</p><p class="kpi-val">{total}</p><p class="kpi-sub">filtered rows</p></div>
      <div class="kpi"><p class="kpi-label">Diabetic</p><p class="kpi-val" style="color:{C_YES}">{yes_n}</p><p class="kpi-sub">{yes_n/total*100:.1f}%</p></div>
      <div class="kpi"><p class="kpi-label">Non-Diabetic</p><p class="kpi-val" style="color:{C_NO}">{no_n}</p><p class="kpi-sub">{no_n/total*100:.1f}%</p></div>
      <div class="kpi"><p class="kpi-label">Avg BMI</p><p class="kpi-val">{d['bmi'].mean():.1f}</p><p class="kpi-sub">kg/m²</p></div>
      <div class="kpi"><p class="kpi-label">Avg Symptoms</p><p class="kpi-val">{d['symptom_score'].mean():.1f}</p><p class="kpi-sub">out of 5</p></div>
    </div>""", unsafe_allow_html=True)

    # SECTION 1: Demographics
    st.markdown('<p class="sh">Section 1 · Demographics</p>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown('<div class="gc"><p class="gc-title">Target Distribution — Yes vs No</p>', unsafe_allow_html=True)
        counts = d['diabetes_diagnosis'].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        wedges,_,autos = axes[0].pie(counts, labels=counts.index,
            colors=[PAL.get(k,'#888') for k in counts.index],
            autopct='%1.1f%%', startangle=90, pctdistance=.75,
            wedgeprops={'edgecolor':'#0B0B18','linewidth':3})
        for a in autos: a.set_fontsize(11); a.set_color('white'); a.set_fontweight('bold')
        axes[0].set_title('Pie', fontsize=9, color='#777')
        bars = axes[1].bar(counts.index, counts.values,
                            color=[PAL.get(k,'#888') for k in counts.index],
                            edgecolor='none', width=.5)
        for bar, v, p in zip(bars, counts.values, counts.values/len(d)*100):
            axes[1].text(bar.get_x()+bar.get_width()/2, v+2,
                         f'{v}\n({p:.1f}%)', ha='center', fontsize=9, color='#ccc', fontweight='bold')
        axes[1].set_ylim(0, counts.max()*1.25); axes[1].set_ylabel('Count', fontsize=9)
        axes[1].set_title('Count', fontsize=9, color='#777')
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    with r1c2:
        st.markdown('<div class="gc"><p class="gc-title">Age Group — Count &amp; Rate</p>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        count_bar(axes[0], d, 'Age', AGE_ORD, 'Count', 25, ['20s','30s','40s','50s','60+'])
        pct_bar  (axes[1], d, 'Age', AGE_ORD, 'Rate (%)', 25, ['20s','30s','40s','50s','60+'])
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    r1c3, r1c4 = st.columns(2)
    with r1c3:
        st.markdown('<div class="gc"><p class="gc-title">Gender vs Diabetes Rate</p>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        count_bar(axes[0], d, 'Gender', ['Female','Male'], 'Count', 0)
        pct_bar  (axes[1], d, 'Gender', ['Female','Male'], 'Rate (%)', 0)
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    with r1c4:
        st.markdown('<div class="gc"><p class="gc-title">Occupation vs Diabetes Rate</p>', unsafe_allow_html=True)
        occ_ord = [o for o in ['Employed','Unemployed','Student','Retired'] if o in d['Occupation'].unique()]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        count_bar(axes[0], d, 'Occupation', occ_ord, 'Count', 15)
        pct_bar  (axes[1], d, 'Occupation', occ_ord, 'Rate (%)', 15)
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # SECTION 2: Physical Measurements
    st.markdown('<p class="sh">Section 2 · Physical Measurements</p>', unsafe_allow_html=True)
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown('<div class="gc"><p class="gc-title">BMI — Histogram, Boxplot &amp; Category Rate</p>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        hist_pair(axes[0], d, 'bmi', 'BMI Histogram', 'BMI (kg/m²)', vlines=[(25,'Overweight','#fdcb6e'),(30,'Obese','#e17055')])
        box_pair (axes[1], d, 'bmi', 'BMI Boxplot', 'BMI')
        pct_bar  (axes[2], d, 'bmi_category', BMI_ORD, 'BMI Category Rate', 15)
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    with r2c2:
        st.markdown('<div class="gc"><p class="gc-title">Waist Circumference &amp; Weight Range</p>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        hist_pair(axes[0], d, 'waist_cm', 'Waist Histogram', 'Waist (cm)', vlines=[(88,'Risk ♀ (88cm)','#fdcb6e'),(102,'Risk ♂ (102cm)','#e17055')])
        box_pair (axes[1], d, 'waist_cm', 'Waist Boxplot', 'Waist (cm)')
        pct_bar  (axes[2], d, 'weight_range', W_ORD, 'Weight Range Rate', 25, ['<50kg','50-59','60-69','70-79','≥80'])
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # SECTION 3: Blood Pressure
    st.markdown('<p class="sh">Section 3 · Blood Pressure</p>', unsafe_allow_html=True)
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.markdown('<div class="gc"><p class="gc-title">Systolic BP — Histogram, Boxplot &amp; Category Rate</p>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        hist_pair(axes[0], d, 'systolic_bp', 'Systolic Histogram', 'Systolic (mmHg)', vlines=[(130,'Elevated','#fdcb6e'),(140,'High Stg 1','#e17055')])
        box_pair (axes[1], d, 'systolic_bp', 'Systolic Boxplot', 'mmHg')
        pct_bar  (axes[2], d, 'bp_category', BP_ORD, 'BP Category Rate', 15)
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    with r3c2:
        st.markdown('<div class="gc"><p class="gc-title">Diastolic BP &amp; Scatter</p>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        hist_pair(axes[0], d, 'diastolic_bp', 'Diastolic Histogram', 'Diastolic (mmHg)', vlines=[(90,'High (90)','#e17055')])
        box_pair (axes[1], d, 'diastolic_bp', 'Diastolic Boxplot', 'mmHg')
        for lbl, color in PAL.items():
            sub = d[d['diabetes_diagnosis']==lbl]
            axes[2].scatter(sub['systolic_bp'], sub['diastolic_bp'], c=color, alpha=.5, s=30, edgecolors='none', label=lbl)
        axes[2].axvline(130, color='#fdcb6e', lw=1.2, linestyle='--', alpha=.7)
        axes[2].axhline(90,  color='#fdcb6e', lw=1.2, linestyle='--', alpha=.7)
        axes[2].set_xlabel('Systolic', fontsize=9); axes[2].set_ylabel('Diastolic', fontsize=9)
        axes[2].set_title('Systolic vs Diastolic', fontsize=10, fontweight='bold', pad=8)
        axes[2].legend(fontsize=8)
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # SECTION 4: Lifestyle
    st.markdown('<p class="sh">Section 4 · Lifestyle Factors</p>', unsafe_allow_html=True)
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        st.markdown('<div class="gc"><p class="gc-title">Exercise, Sleep &amp; Diet vs Diabetes Rate</p>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        pct_bar(axes[0], d, 'exercise_freq', EX_ORD, 'Exercise Rate', 15, ['Never','1-3×','3-6×','Daily'])
        pct_bar(axes[1], d, 'sleep_hours',   SL_ORD, 'Sleep Rate', 15, ['<5h','5-6h','6-7h','7-8h','>8h'])
        pct_bar(axes[2], d, 'diet_type', ['Mostly unhealthy','Mixed','Mostly healthy'], 'Diet Rate', 20)
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    with r4c2:
        st.markdown('<div class="gc"><p class="gc-title">Family History &amp; Exercise Count vs Diabetes</p>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        fam_ct  = d.groupby(['family_history','diabetes_diagnosis']).size().unstack(fill_value=0)
        fam_pct = fam_ct.div(fam_ct.sum(axis=1), axis=0) * 100
        avail   = [c for c in ['No','Yes'] if c in fam_pct.columns]
        fam_pct[avail].plot(kind='bar', ax=axes[0], color=[PAL[c] for c in avail], edgecolor='none', width=.55)
        axes[0].set_xticklabels(['No History','Family Hx'], rotation=0, fontsize=9)
        axes[0].set_ylabel('%', fontsize=9); axes[0].set_title('Family History Rate', fontsize=10, fontweight='bold')
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0f}%'))
        axes[0].legend(title='Diabetic', fontsize=8)
        count_bar(axes[1], d, 'exercise_freq', EX_ORD, 'Exercise Count', 15, ['Never','1-3×','3-6×','Daily'])
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # SECTION 5: Symptoms
    st.markdown('<p class="sh">Section 5 · Symptom Analysis</p>', unsafe_allow_html=True)
    SYMP_COLS   = ['frequent_urination','unusual_thirst','weight_change','fatigue','blurred_vision']
    SYMP_LABELS = ['Frequent\nUrination','Unusual\nThirst','Weight\nChange','Fatigue','Blurred\nVision']
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        st.markdown('<div class="gc"><p class="gc-title">Symptom Score — Count, Rate &amp; Boxplot</p>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        sc_ct = d.groupby(['symptom_score','diabetes_diagnosis']).size().unstack(fill_value=0)
        avail = [c for c in ['No','Yes'] if c in sc_ct.columns]
        sc_ct[avail].plot(kind='bar', ax=axes[0], color=[PAL[c] for c in avail], edgecolor='none', width=.7)
        axes[0].set_xticklabels(range(len(sc_ct)), rotation=0, fontsize=8)
        axes[0].set_xlabel('Score (0–5)'); axes[0].set_title('Score Count', fontsize=10, fontweight='bold')
        axes[0].legend(title='Diabetic', fontsize=8)
        sc_pct = sc_ct.div(sc_ct.sum(axis=1), axis=0) * 100
        sc_pct[avail].plot(kind='bar', ax=axes[1], color=[PAL[c] for c in avail], edgecolor='none', width=.7)
        axes[1].set_xticklabels(range(len(sc_pct)), rotation=0, fontsize=8)
        axes[1].set_xlabel('Score (0–5)'); axes[1].set_title('Score Rate', fontsize=10, fontweight='bold')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0f}%'))
        axes[1].legend(title='Diabetic', fontsize=8)
        box_pair(axes[2], d, 'symptom_score', 'Score Boxplot', 'Symptom Score (0–5)')
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    with r5c2:
        st.markdown('<div class="gc"><p class="gc-title">Each Symptom — % Diabetic Present vs Absent &amp; Group Prevalence</p>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        pct_p, pct_a, prev_y, prev_n = [], [], [], []
        for col in SYMP_COLS:
            p  = d[d[col]=='Yes']; a  = d[d[col]=='No']
            py = d[d['diabetes_diagnosis']=='Yes']; pn = d[d['diabetes_diagnosis']=='No']
            pct_p.append((p['diabetes_diagnosis']=='Yes').mean()*100 if len(p) else 0)
            pct_a.append((a['diabetes_diagnosis']=='Yes').mean()*100 if len(a) else 0)
            prev_y.append((py[col]=='Yes').mean()*100 if len(py) else 0)
            prev_n.append((pn[col]=='Yes').mean()*100 if len(pn) else 0)
        x = np.arange(len(SYMP_LABELS)); w = .36
        axes[0].bar(x-w/2, pct_p, w, color=C_YES, label='Symptom Present', edgecolor='none')
        axes[0].bar(x+w/2, pct_a, w, color=C_NO,  label='Symptom Absent',  edgecolor='none')
        axes[0].set_xticks(x); axes[0].set_xticklabels(SYMP_LABELS, fontsize=7)
        axes[0].set_ylabel('% Diabetic'); axes[0].set_title('% Diabetic — Present vs Absent', fontsize=9, fontweight='bold')
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0f}%'))
        axes[0].legend(fontsize=8)
        axes[1].bar(x-w/2, prev_y, w, color=C_YES, label='Diabetic Group',     edgecolor='none')
        axes[1].bar(x+w/2, prev_n, w, color=C_NO,  label='Non-Diabetic Group', edgecolor='none')
        axes[1].set_xticks(x); axes[1].set_xticklabels(SYMP_LABELS, fontsize=7)
        axes[1].set_ylabel('% Reporting Symptom'); axes[1].set_title('Symptom Prevalence in Each Group', fontsize=9, fontweight='bold')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0f}%'))
        axes[1].legend(fontsize=8)
        plt.tight_layout(); render_fig(fig)
        st.markdown('</div>', unsafe_allow_html=True)




# TAB 4 — ML MODELS & TUNING

with tab4:
    st.markdown('<p class="sh"> 5 ML Models — Hyperparameter Tuning Results</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="gc"><p style="font-size:13px;color:#999;line-height:1.7;margin:0">
    All 5 models tuned with <b style="color:#a29bfe">RandomizedSearchCV</b>
    (n_iter=30, cv=StratifiedKFold(5), scoring=F1).
    Class imbalance handled via <code>class_weight='balanced'</code>.
    Feature scaling applied to LR &amp; SVM only (tree models are scale-invariant).
    <br><b style="color:#ffeaa7"> Best model: HistGradient Boosting</b>
    — Accuracy 89.1% · F1 91.4% · ROC-AUC 95.3%
    </p></div>""", unsafe_allow_html=True)

    # Overall comparison
    st.markdown('<p class="sh">Overall Comparison — All 5 Models</p>', unsafe_allow_html=True)
    st.markdown('<div class="gc">', unsafe_allow_html=True)
    m_keys = ['acc','pre','rec','f1','auc']
    m_labs = ['Accuracy','Precision','Recall','F1','ROC-AUC']
    names  = list(MODEL_INFO.keys()); x = np.arange(len(m_labs)); w = .15
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for i, name in enumerate(names):
        mi   = MODEL_INFO[name]
        vals = [mi[k] for k in m_keys]
        axes[0].bar(x+i*w, vals, w, label=name[:12], color=mi['color'], edgecolor='none')
    axes[0].set_xticks(x+w*2); axes[0].set_xticklabels(m_labs, fontsize=10)
    axes[0].set_ylim(70,102); axes[0].set_ylabel('%')
    axes[0].set_title('All Metrics — Post Hyperparameter Tuning', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8, loc='lower right')
    axes[1].plot([0,1],[0,1],'#444',linestyle='--',lw=1,label='Random baseline')
    try:
        df_m = pd.read_csv('diabetes_cleaned_v2.csv')
        Xm = df_m.drop('diabetes_diagnosis_enc',axis=1)
        ym = df_m['diabetes_diagnosis_enc']
        Xtr2,Xte2,ytr2,yte2 = train_test_split(Xm,ym,test_size=.2,random_state=42,stratify=ym)
        sc2 = StandardScaler(); Xtr2_sc=sc2.fit_transform(Xtr2); Xte2_sc=sc2.transform(Xte2)
        roc_configs = {
            'Logistic Regression':   (LogisticRegression(C=1,solver='lbfgs',class_weight='balanced',max_iter=2000,random_state=42), Xtr2_sc, Xte2_sc),
            'SVM (RBF)':             (SVC(C=100,gamma='scale',probability=True,class_weight='balanced',random_state=42), Xtr2_sc, Xte2_sc),
            'Random Forest':         (RandomForestClassifier(n_estimators=100,max_depth=20,min_samples_split=10,max_features='sqrt',class_weight='balanced',random_state=42), Xtr2, Xte2),
            'Extra Trees':           (ExtraTreesClassifier(n_estimators=300,max_depth=10,max_features='log2',class_weight='balanced',random_state=42), Xtr2, Xte2),
            'HistGradient Boosting': (HistGradientBoostingClassifier(learning_rate=0.1,max_depth=7,max_iter=200,min_samples_leaf=30,random_state=42), Xtr2, Xte2),
        }
        for name, (m2, tr2, te2) in roc_configs.items():
            m2.fit(tr2,ytr2)
            ypr2     = m2.predict_proba(te2)[:,1]
            fpr2,tpr2,_ = roc_curve(yte2, ypr2)
            auc2     = roc_auc_score(yte2,ypr2)*100
            axes[1].plot(fpr2, tpr2, color=MODEL_INFO[name]['color'], lw=2,
                         label=f"{name[:12]} ({auc2:.1f}%)")
    except Exception:
        axes[1].text(0.3,0.5,'ROC curves require diabetes_cleaned_v2.csv',color='#666',fontsize=9)
    axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
    axes[1].set_title('ROC Curves — All 5 Models', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=8, loc='lower right')
    plt.tight_layout(); render_fig(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # Individual model cards
    st.markdown('<p class="sh">Individual Model Details</p>', unsafe_allow_html=True)
    for name, mi in MODEL_INFO.items():
        is_best  = (name == 'HistGradient Boosting')
        best_tag = '<span class="best-tag">★ Best Model</span>' if is_best else ''
        st.markdown(
            f'<div class="mc" style="--mc-color:{mi["color"]}">'
            f'<p class="mc-name">{name}{best_tag}</p>'
            f'<p class="mc-desc">{mi["desc"]}</p>',
            unsafe_allow_html=True)
        st.markdown('<p style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:.04em;margin:0 0 4px">Best Hyperparameters</p>', unsafe_allow_html=True)
        pills = ''.join(f'<span class="pill">{p.strip()}</span>' for p in mi['params'].split('·'))
        st.markdown(f'<div class="pill-row">{pills}</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:11px;color:#555;font-style:italic;margin:0 0 .7rem">↳ {mi["why"]}</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;flex-wrap:wrap;gap:7px;margin:.5rem 0 .8rem">
          <div class="mpill">Accuracy <b>{mi['acc']}%</b></div>
          <div class="mpill">Precision <b>{mi['pre']}%</b></div>
          <div class="mpill">Recall <b>{mi['rec']}%</b></div>
          <div class="mpill">F1 Score <b>{mi['f1']}%</b></div>
          <div class="mpill">ROC-AUC <b>{mi['auc']}%</b></div>
          <div class="mpill">CV F1 <b>{mi['cv_m']}% ±{mi['cv_s']}%</b></div>
        </div>""", unsafe_allow_html=True)

        cm_d    = np.array(mi['cm'])
        has_fi  = name in ['Random Forest','Extra Trees']
        cols_mc = st.columns(3 if has_fi else 2)

        with cols_mc[0]:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            ax.axis('off')
            ax2 = fig.add_axes([0.1, 0.15, 0.8, 0.65])
            auc_v = mi['auc']/100
            t_fpr = np.linspace(0, 1, 100)
            t_tpr = np.clip(t_fpr ** (1/(2*auc_v-0.5+0.01)), 0, 1)
            ax2.plot(t_fpr, t_tpr, color=mi['color'], lw=2.5)
            ax2.fill_between(t_fpr, t_tpr, alpha=0.15, color=mi['color'])
            ax2.plot([0,1],[0,1],'#444',linestyle='--',lw=1)
            ax2.text(0.5, 0.3, f"AUC\n{mi['auc']}%", ha='center', va='center',
                     fontsize=16, fontweight='bold', color=mi['color'],
                     transform=ax2.transAxes)
            ax2.set_xlabel('FPR',fontsize=8); ax2.set_ylabel('TPR',fontsize=8)
            ax2.tick_params(labelsize=7); ax2.set_facecolor('#0f0f22')
            ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_color('#1e1e3a'); ax2.spines['bottom'].set_color('#1e1e3a')
            ax.set_title('ROC Curve', fontsize=10, fontweight='bold')
            render_fig(fig)

        with cols_mc[1]:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            ax.imshow(cm_d, cmap='Blues', aspect='auto')
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(['Pred No','Pred Yes'], fontsize=9)
            ax.set_yticklabels(['Act No','Act Yes'],   fontsize=9)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm_d[i,j], ha='center', va='center',
                            fontsize=18, fontweight='bold',
                            color='white' if cm_d[i,j] > cm_d.max()/2 else '#aaa')
            ax.set_title('Confusion Matrix', fontsize=10, fontweight='bold')
            plt.tight_layout(); render_fig(fig)

        if has_fi:
            with cols_mc[2]:
                try:
                    df_fi = pd.read_csv('diabetes_cleaned_v2.csv')
                    Xfi   = df_fi.drop('diabetes_diagnosis_enc',axis=1)
                    yfi   = df_fi['diabetes_diagnosis_enc']
                    Xtr_fi,_,ytr_fi,_ = train_test_split(Xfi,yfi,test_size=.2,random_state=42,stratify=yfi)
                    m_fi = (RandomForestClassifier(n_estimators=100,max_depth=20,min_samples_split=10,
                                                    max_features='sqrt',class_weight='balanced',random_state=42)
                            if name=='Random Forest' else
                            ExtraTreesClassifier(n_estimators=300,max_depth=10,max_features='log2',
                                                  class_weight='balanced',random_state=42))
                    m_fi.fit(Xtr_fi, ytr_fi)
                    fi2      = pd.Series(m_fi.feature_importances_, index=Xfi.columns)
                    fi2.index= [FEAT_LBL.get(f,f) for f in fi2.index]
                    top10    = fi2.sort_values(ascending=True).tail(10)
                    fig, ax  = plt.subplots(figsize=(5.5, 3.5))
                    ax.barh(top10.index, top10.values*100, color=mi['color']+'cc', edgecolor='none', height=.65)
                    for i2, (idx2, val2) in enumerate(top10.items()):
                        ax.text(val2*100+.2, i2, f'{val2*100:.1f}%', va='center', fontsize=8)
                    ax.set_xlabel('Importance (%)', fontsize=9)
                    ax.set_title('Top 10 Feature Importances', fontsize=10, fontweight='bold')
                    plt.tight_layout(); render_fig(fig)
                except Exception:
                    st.caption("Feature importance chart requires diabetes_cleaned_v2.csv")
        st.markdown('</div>', unsafe_allow_html=True)
