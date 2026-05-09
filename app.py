"""
Insurance Fraud Detection System
=================================
Streamlit app — matches exactly the preprocessing pipeline in the notebook:
  - Column lowercasing + strip
  - Ordinal encoding (9 columns)
  - get_dummies(drop_first=True) on all remaining categoricals
  - StandardScaler transform
  - Cost-sensitive Weighted LR with optimised threshold
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM STYLING ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }
    .risk-high {
        background: #fef2f2; border-left: 5px solid #dc2626;
        padding: 1.2rem 1.5rem; border-radius: 8px; margin-top: 1rem;
    }
    .risk-low {
        background: #f0fdf4; border-left: 5px solid #16a34a;
        padding: 1.2rem 1.5rem; border-radius: 8px; margin-top: 1rem;
    }
    .risk-medium {
        background: #fffbeb; border-left: 5px solid #d97706;
        padding: 1.2rem 1.5rem; border-radius: 8px; margin-top: 1rem;
    }
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; }
    .metric-card .label { font-size: 0.8rem; color: #64748b; margin-top: 2px; }
    .section-label {
        font-size: 0.75rem; font-weight: 600; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .flag-item {
        background: #fef9c3; border: 1px solid #fde047;
        border-radius: 6px; padding: 0.5rem 0.8rem;
        margin: 0.3rem 0; font-size: 0.88rem;
    }
    .stButton > button {
        width: 100%; background: #1e3a5f; color: white;
        border: none; border-radius: 8px; padding: 0.7rem;
        font-size: 1rem; font-weight: 600; cursor: pointer;
    }
    .stButton > button:hover { background: #2d6a9f; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD ARTIFACTS ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    required = ["model.pkl", "scaler.pkl", "features.pkl", "threshold.pkl"]
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        st.error(f"Missing model files: {', '.join(missing)}\n\n"
                 "Run your notebook's final cell to generate model.pkl, "
                 "scaler.pkl, features.pkl, and threshold.pkl.")
        st.stop()
    model     = joblib.load("model.pkl")
    scaler    = joblib.load("scaler.pkl")
    features  = joblib.load("features.pkl")
    threshold = joblib.load("threshold.pkl")
    return model, scaler, features, threshold

model, scaler, features, threshold = load_artifacts()

# ─── AGE → AGE GROUP HELPER ──────────────────────────────────────────────────
def age_to_group(age: int) -> str:
    """Map raw age to the ageofpolicyholder bucket the model was trained on."""
    if age <= 17: return "16 – 17"
    if age <= 20: return "18 – 20"
    if age <= 25: return "21 – 25"
    if age <= 30: return "26 – 30"
    if age <= 35: return "31 – 35"
    if age <= 40: return "36 – 40"
    if age <= 50: return "41 – 50"
    if age <= 65: return "51 – 65"
    return "Over 65"

# ─── ORDINAL MAPS — must match notebook exactly ───────────────────────────────
ORDINAL_MAPS = {
    "vehicleprice": {
        "Less than $20,000":  ("less than 20000", 1),
        "$20,000 – $29,999":  ("20000 to 29000",  2),
        "$30,000 – $39,999":  ("30000 to 39000",  3),
        "$40,000 – $59,999":  ("40000 to 59000",  4),
        "$60,000 – $69,999":  ("60000 to 69000",  5),
        "More than $69,999":  ("more than 69000", 6),
    },
    "pastnumberofclaims": {
        "None":         ("none",        0),
        "1 claim":      ("1",           1),
        "2 – 4 claims": ("2 to 4",      2),
        "More than 4":  ("more than 4", 3),
    },
    "ageofvehicle": {
        "Brand new":   ("new",         0),
        "2 years old": ("2 years",     2),
        "3 years old": ("3 years",     3),
        "4 years old": ("4 years",     4),
        "5 years old": ("5 years",     5),
        "6 years old": ("6 years",     6),
        "7 years old": ("7 years",     7),
        "More than 7": ("more than 7", 8),
    },
    "numberofcars": {
        "1 vehicle":   ("1 vehicle",  1),
        "2 vehicles":  ("2 vehicles", 2),
        "3 to 4":      ("3 to 4",     3),
        "5 to 8":      ("5 to 8",     5),
        "More than 8": ("more than 8",9),
    },
    "days_policy_accident": {
        "Same day (none)":   ("none",         0),
        "1 to 7 days":       ("1 to 7",       1),
        "8 to 15 days":      ("8 to 15",      2),
        "15 to 30 days":     ("15 to 30",     3),
        "More than 30 days": ("more than 30", 4),
    },
    "days_policy_claim": {
        "Same day (none)":   ("none",         0),
        "8 to 15 days":      ("8 to 15",      1),
        "15 to 30 days":     ("15 to 30",     2),
        "More than 30 days": ("more than 30", 3),
    },
    "numberofsuppliments": {
        "None":        ("none",        0),
        "1 to 2":      ("1 to 2",      1),
        "3 to 5":      ("3 to 5",      2),
        "More than 5": ("more than 5", 3),
    },
    "addresschange_claim": {
        "No change":      ("no change",      0),
        "Under 6 months": ("under 6 months", 1),
        "1 year":         ("1 year",         2),
        "2 to 3 years":   ("2 to 3 years",   3),
        "4 to 8 years":   ("4 to 8 years",   4),
    },
    "ageofpolicyholder": {
        "16 – 17": ("16 to 17", 1),
        "18 – 20": ("18 to 20", 2),
        "21 – 25": ("21 to 25", 3),
        "26 – 30": ("26 to 30", 4),
        "31 – 35": ("31 to 35", 5),
        "36 – 40": ("36 to 40", 6),
        "41 – 50": ("41 to 50", 7),
        "51 – 65": ("51 to 65", 8),
        "Over 65": ("over 65",  9),
    },
}

# ─── CATEGORICAL COLUMNS (one-hot after drop_first=True) ─────────────────────
# FIX 1: key is 'policytype' not 'policetype' — matches notebook column name
# FIX 2: 'Mecedes' not 'Mecury' — verified against insurance_claims.csv
CAT_COLUMNS = {
    "month":            ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"],
    "dayofweek":        ["Monday","Tuesday","Wednesday","Thursday","Friday",
                         "Saturday","Sunday"],
    "make":             ["Accura","BMW","Chevrolet","Dodge","Ferrari","Ford",
                         "Honda","Jaguar","Lexus","Mazda","Mecedes","Mercury",
                         "Nisson","Pontiac","Porche","Saab","Saturn","Toyota","VW"],
    "accidentarea":     ["Rural","Urban"],
    "dayofweekclaimed": ["Monday","Tuesday","Wednesday","Thursday","Friday",
                         "Saturday","Sunday"],
    "monthclaimed":     ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"],
    "sex":              ["Female","Male"],
    "maritalstatus":    ["Divorced","Married","Single","Widow"],
    "fault":            ["Policy Holder","Third Party"],
    "policytype":       ["Sedan - All Perils","Sedan - Collision","Sedan - Liability",
                         "Sport - All Perils","Sport - Collision","Sport - Liability",
                         "Utility - All Perils","Utility - Collision","Utility - Liability"],
    "vehiclecategory":  ["Sedan","Sport","Utility"],
    "policereportfiled":["No","Yes"],
    "witnesspresent":   ["No","Yes"],
    "agenttype":        ["External","Internal"],
    "basepolicy":       ["All Perils","Collision","Liability"],
}

# ─── PREPROCESSING — mirrors notebook exactly ─────────────────────────────────
def build_feature_vector(inputs: dict) -> pd.DataFrame:
    row = {}
    for col in ["age","weekofmonth","weekofmonthclaimed","deductible","driverrating","year"]:
        row[col] = inputs[col]
    for col, mapping in ORDINAL_MAPS.items():
        _, val = mapping[inputs[col]]
        row[col] = val
    for col, categories in CAT_COLUMNS.items():
        chosen  = inputs[col]
        dropped = sorted(categories)[0]
        for cat in categories:
            if cat != dropped:
                row[f"{col}_{cat}"] = 1 if chosen == cat else 0
    df = pd.DataFrame([row])
    df = df.reindex(columns=features, fill_value=0)
    return df

# ─── FRAUD RISK FLAGS ─────────────────────────────────────────────────────────
def get_risk_flags(inputs: dict) -> list[str]:
    flags = []
    _, days_pa_val = ORDINAL_MAPS["days_policy_accident"][inputs["days_policy_accident"]]
    _, days_pc_val = ORDINAL_MAPS["days_policy_claim"][inputs["days_policy_claim"]]
    _, addr_val    = ORDINAL_MAPS["addresschange_claim"][inputs["addresschange_claim"]]
    _, vp_val      = ORDINAL_MAPS["vehicleprice"][inputs["vehicleprice"]]
    _, past_val    = ORDINAL_MAPS["pastnumberofclaims"][inputs["pastnumberofclaims"]]
    _, supp_val    = ORDINAL_MAPS["numberofsuppliments"][inputs["numberofsuppliments"]]
    if inputs["fault"] == "Third Party":
        flags.append("🚩 Fault attributed to third party — common fraud pattern")
    if days_pa_val == 0:
        flags.append("🚩 Accident reported same day as policy start")
    if days_pc_val == 0:
        flags.append("🚩 Claim filed same day as policy start")
    if addr_val >= 1:
        flags.append(f"🚩 Address changed {inputs['addresschange_claim']} before claim — potential identity fraud indicator")
    if past_val >= 2:
        flags.append(f"🚩 {inputs['pastnumberofclaims']} prior claims — elevated repeat-claimant risk")
    if supp_val >= 2:
        flags.append(f"🚩 {inputs['numberofsuppliments']} supplemental claims — unusual volume")
    if vp_val >= 5:
        flags.append("🚩 High-value vehicle (>$60K) — higher financial incentive for fraud")
    if inputs["policereportfiled"] == "No" and inputs["witnesspresent"] == "No":
        flags.append("🚩 No police report AND no witnesses — claim unverifiable")
    if inputs["agenttype"] == "External":
        flags.append("⚠️ External agent — slightly elevated fraud rate in dataset")
    if inputs["accidentarea"] == "Rural":
        flags.append("⚠️ Rural accident area — harder to verify independently")
    return flags

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ Insurance Fraud Detection System</h1>
    <p>Cost-sensitive ML model · Optimised for fraud recall · For claims adjuster use</p>
</div>
""", unsafe_allow_html=True)
st.markdown(
    f"**Model:** Cost-Sensitive Logistic Regression &nbsp;|&nbsp; "
    f"**Decision Threshold:** {threshold:.3f} &nbsp;|&nbsp; "
    f"**Trained recall:** ≥ 0.86 on fraud class"
)
st.divider()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Claim Information")
    st.caption("Fill in all fields before clicking Assess Claim.")

    st.markdown("#### Policy Details")
    month      = st.selectbox("Month of Incident",  CAT_COLUMNS["month"])
    year       = st.selectbox("Policy Year",        [1994, 1995, 1996])
    basepolicy = st.selectbox("Base Policy Type",   CAT_COLUMNS["basepolicy"])
    policytype = st.selectbox("Policy Type",        CAT_COLUMNS["policytype"])
    agenttype  = st.selectbox("Agent Type",         CAT_COLUMNS["agenttype"])
    deductible = st.select_slider("Deductible ($)", options=[300, 400, 500, 700])

    st.markdown("#### Policyholder")
    sex           = st.selectbox("Sex",            CAT_COLUMNS["sex"])
    maritalstatus = st.selectbox("Marital Status", CAT_COLUMNS["maritalstatus"])
    driverrating  = st.select_slider("Driver Rating", options=[1, 2, 3, 4],
                                     help="1 = worst, 4 = best")
    pastnumberofclaims = st.selectbox("Past Claims",
                                      list(ORDINAL_MAPS["pastnumberofclaims"].keys()))
    # FIX 3: single age slider — age group derived automatically, no redundant dropdown
    age = st.slider("Policyholder Age", 16, 80, 35)
    ageofpolicyholder = age_to_group(age)
    st.caption(f"Age group (used by model): **{ageofpolicyholder}**")

    st.markdown("#### Vehicle")
    make            = st.selectbox("Vehicle Make",     CAT_COLUMNS["make"])
    vehiclecategory = st.selectbox("Vehicle Category", CAT_COLUMNS["vehiclecategory"])
    vehicleprice    = st.selectbox("Vehicle Price",    list(ORDINAL_MAPS["vehicleprice"].keys()))
    ageofvehicle    = st.selectbox("Age of Vehicle",   list(ORDINAL_MAPS["ageofvehicle"].keys()))
    numberofcars    = st.selectbox("Number of Cars Insured",
                                   list(ORDINAL_MAPS["numberofcars"].keys()))

    st.markdown("#### Accident Details")
    accidentarea      = st.selectbox("Accident Area",       CAT_COLUMNS["accidentarea"])
    fault             = st.selectbox("Fault",               CAT_COLUMNS["fault"])
    dayofweek         = st.selectbox("Day of Accident",     CAT_COLUMNS["dayofweek"])
    weekofmonth       = st.select_slider("Week of Month (Accident)", options=[1, 2, 3, 4, 5])
    policereportfiled = st.selectbox("Police Report Filed?", CAT_COLUMNS["policereportfiled"])
    witnesspresent    = st.selectbox("Witness Present?",    CAT_COLUMNS["witnesspresent"])

    st.markdown("#### Claim Filing")
    monthclaimed       = st.selectbox("Month Claimed", CAT_COLUMNS["monthclaimed"])
    dayofweekclaimed   = st.selectbox("Day Claimed",   CAT_COLUMNS["dayofweekclaimed"])
    weekofmonthclaimed = st.select_slider("Week of Month (Claim)", options=[1, 2, 3, 4, 5])
    days_policy_accident = st.selectbox("Days Between Policy Start & Accident",
                                        list(ORDINAL_MAPS["days_policy_accident"].keys()))
    days_policy_claim    = st.selectbox("Days Between Policy Start & Claim",
                                        list(ORDINAL_MAPS["days_policy_claim"].keys()))
    addresschange_claim  = st.selectbox("Address Changed Before Claim?",
                                        list(ORDINAL_MAPS["addresschange_claim"].keys()))
    numberofsuppliments  = st.selectbox("Supplemental Claims Filed",
                                        list(ORDINAL_MAPS["numberofsuppliments"].keys()))
    st.markdown("---")
    assess_btn = st.button("🔍  Assess Claim for Fraud", use_container_width=True)

# ─── MAIN PANEL ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown("### Claim Summary")
    summary = {
        "Policyholder Age / Group": f"{age} / {ageofpolicyholder}",
        "Sex":                      sex,
        "Vehicle":                  f"{year} {make} {vehiclecategory} ({ageofvehicle})",
        "Vehicle Price":            vehicleprice,
        "Accident Area / Fault":    f"{accidentarea} / {fault}",
        "Policy Type":              policytype,
        "Past Claims":              pastnumberofclaims,
        "Police Report Filed":      policereportfiled,
        "Witness Present":          witnesspresent,
        "Address Change":           addresschange_claim,
        "Supplemental Claims":      numberofsuppliments,
    }
    for k, v in summary.items():
        c1, c2 = st.columns([0.45, 0.55])
        c1.markdown(f"<div class='section-label'>{k}</div>", unsafe_allow_html=True)
        c2.markdown(f"**{v}**")

with col_right:
    st.markdown("### Risk Assessment")
    if not assess_btn:
        st.info("👈 Complete the claim form on the left and click "
                "**Assess Claim for Fraud** to run the model.")
    else:
        inputs = dict(
            age=age, weekofmonth=weekofmonth, weekofmonthclaimed=weekofmonthclaimed,
            deductible=deductible, driverrating=driverrating, year=year,
            month=month, dayofweek=dayofweek, make=make, accidentarea=accidentarea,
            dayofweekclaimed=dayofweekclaimed, monthclaimed=monthclaimed,
            sex=sex, maritalstatus=maritalstatus, fault=fault,
            policytype=policytype, vehiclecategory=vehiclecategory,
            policereportfiled=policereportfiled, witnesspresent=witnesspresent,
            agenttype=agenttype, basepolicy=basepolicy,
            vehicleprice=vehicleprice, pastnumberofclaims=pastnumberofclaims,
            ageofvehicle=ageofvehicle, numberofcars=numberofcars,
            days_policy_accident=days_policy_accident,
            days_policy_claim=days_policy_claim,
            numberofsuppliments=numberofsuppliments,
            addresschange_claim=addresschange_claim,
            ageofpolicyholder=ageofpolicyholder,
        )

        try:
            df_input   = build_feature_vector(inputs)
            df_scaled  = scaler.transform(df_input)
            proba      = model.predict_proba(df_scaled)[0, 1]
            prediction = int(proba >= threshold)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        pct         = proba * 100
        gauge_color = "#dc2626" if prediction == 1 else (
                      "#d97706" if proba > 0.3 else "#16a34a")

        st.markdown(f"""
        <div style="text-align:center; margin-bottom: 1rem;">
            <div style="font-size:3.5rem; font-weight:800; color:{gauge_color};">
                {pct:.1f}%
            </div>
            <div style="font-size:0.9rem; color:#64748b; margin-top:-0.3rem;">
                Fraud Probability Score
            </div>
        </div>""", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("""<div class="risk-high">
                <strong>⚠️ HIGH FRAUD RISK — Flag for Investigation</strong><br>
                <span style="font-size:0.88rem;">Model predicts this claim is fraudulent.
                Recommend manual review by a senior adjuster before processing payment.
                </span></div>""", unsafe_allow_html=True)
        elif proba > 0.3:
            st.markdown("""<div class="risk-medium">
                <strong>⚠️ ELEVATED RISK — Review Recommended</strong><br>
                <span style="font-size:0.88rem;">Below fraud threshold but above baseline
                risk. Consider requesting additional documentation before approval.
                </span></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="risk-low">
                <strong>✅ LOW FRAUD RISK — Proceed Normally</strong><br>
                <span style="font-size:0.88rem;">Model does not flag this claim as
                fraudulent. Standard processing workflow applies.
                </span></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"""<div class="metric-card">
            <div class="value" style="color:{gauge_color}">{pct:.1f}%</div>
            <div class="label">Fraud Score</div></div>""", unsafe_allow_html=True)
        m2.markdown(f"""<div class="metric-card">
            <div class="value">{threshold:.3f}</div>
            <div class="label">Decision Threshold</div></div>""", unsafe_allow_html=True)
        m3.markdown(f"""<div class="metric-card">
            <div class="value" style="color:{'#dc2626' if prediction==1 else '#16a34a'}">
                {"FRAUD" if prediction==1 else "LEGIT"}</div>
            <div class="label">Model Verdict</div></div>""", unsafe_allow_html=True)

        flags = get_risk_flags(inputs)
        if flags:
            st.markdown("<br>**Risk Signals Detected**", unsafe_allow_html=True)
            for flag in flags:
                st.markdown(f'<div class="flag-item">{flag}</div>', unsafe_allow_html=True)
        else:
            st.success("No specific risk signals detected.")

        st.markdown("<br>**Fraud Score vs. Decision Threshold**", unsafe_allow_html=True)
        st.progress(min(proba, 1.0),
                    text=f"Score: {proba:.3f} (threshold = {threshold:.3f})")

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📝 Add Adjuster Notes"):
            st.text_area("Notes for case file:",
                placeholder="Document any additional observations, phone call outcomes, "
                            "document requests, or escalation decisions here...",
                height=100)
            if st.button("Save Notes"):
                st.success("Notes saved to case file. (Connect to your CMS to persist.)")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Model:** Cost-Sensitive Logistic Regression (class_weight={0:1, 1:10}) | "
    f"**Threshold:** {threshold:.3f} optimised to minimise C_FN=10 × FN + C_FP=1 × FP | "
    "**Dataset:** Insurance Fraud Oracle (15,420 records, 6% fraud) | "
    "⚠️ This tool assists — it does not replace — human adjuster judgment."
)
