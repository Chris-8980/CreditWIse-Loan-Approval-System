import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="CreditWise Loan Approval System",
    page_icon="💳",
    layout="wide"
)


# -------------------------------------------------
# CUSTOM DARK UI
# -------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

/* Main text */
h1, h2, h3 {
    color: white;
}

p, label, .stMarkdown {
    color: #E6E6E6;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
}

/* Sidebar title and text */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    color: black !important;
}

/* Navigation radio options */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    color: black !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] p {
    color: black !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #1B1F26;
    border: 1px solid #333842;
    padding: 15px;
    border-radius: 10px;
}

div[data-testid="stMetricLabel"] {
    color: #B8BEC8;
}

div[data-testid="stMetricValue"] {
    color: white;
}

</style>
""", unsafe_allow_html=True)
st.title("💳 CreditWise Loan Approval System")
st.markdown("### Intelligent Loan Approval Prediction using Machine Learning")

# -------------------------------------------------
# LOAD & PREPROCESS DATA
# -------------------------------------------------
@st.cache_data
def preprocess():

    df = pd.read_csv("loan_approval_data.csv")

    # Missing values
    cat_cols = df.select_dtypes(include="object").columns
    num_cols = df.select_dtypes(include="number").columns

    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    # Remove ID
    df = df.drop("Applicant_ID", axis=1)

    # Label Encoding
    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

    # One Hot Encoding
    cols = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category"
    ]

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

    encoded = ohe.fit_transform(df[cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(cols),
        index=df.index
    )

    df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)

    # Feature Engineering
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2

    X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    y = df["Loan_Approved"]

    return df, X, y, ohe


df, X, y, ohe = preprocess()

# -------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------
@st.cache_resource
def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }

    metrics = {}

    for name, model in models.items():

        model.fit(X_train_s, y_train)

        pred = model.predict(X_test_s)

        metrics[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "CM": confusion_matrix(y_test, pred)
        }

    return models, scaler, metrics


models, scaler, metrics = train_models(X, y)

best_overall = max(metrics, key=lambda x: metrics[x]["F1"])
best_precision = max(metrics, key=lambda x: metrics[x]["Precision"])

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Loan Prediction", "Model Analysis"]
)

# =================================================
# LOAN PREDICTION
# =================================================
if page == "Loan Prediction":

    st.header("🔮 Loan Approval Prediction")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("Age", 18, 80, 30)

        income = st.number_input(
            "Applicant Income (Monthly)",
            0.0,
            500000.0,
            50000.0
        )

        co_income = st.number_input(
            "Coapplicant Income",
            0.0,
            500000.0,
            10000.0
        )

        loan_amount = st.number_input(
                        "Loan Amount (₹)",
                        min_value=1000.0,
                        max_value=10000000.0,
                        value=150000.0,
                        step=1000.0
        )
        

        savings = st.number_input(
                        "Savings (₹)",
                        min_value=0.0,
                        max_value=10000000.0,
                        value=50000.0,
                        step=1000.0
        )
        

        dti = st.slider("DTI Ratio", 0.0, 1.0, 0.30)

    with col2:

        credit = st.slider("Credit Score", 300, 900, 700)

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        marital = st.selectbox(
            "Marital Status",
            ["Single", "Married"]
        )

        employment = st.selectbox(
            "Employment",
            ["Employed", "Self-Employed", "Unemployed"]
        )

        purpose = st.selectbox(
            "Loan Purpose",
            ["Home", "Business", "Education", "Personal"]
        )

        area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

        employer = st.selectbox(
            "Employer Category",
            ["Private", "Government", "Other"]
        )

    if st.button("Predict Loan Approval", use_container_width=True):

        user = pd.DataFrame({
            "Age":[age],
            "Applicant_Income":[income],
            "Coapplicant_Income":[co_income],
            "Loan_Amount":[loan_amount],
            "Savings":[savings],
            "DTI_Ratio":[dti],
            "Credit_Score":[credit],
            "Education_Level":[1 if education=="Graduate" else 0],
            "Employment_Status":[employment],
            "Marital_Status":[marital],
            "Loan_Purpose":[purpose],
            "Property_Area":[area],
            "Gender":[gender],
            "Employer_Category":[employer]
        })

        # Feature engineering
        user["DTI_Ratio_sq"] = user["DTI_Ratio"]**2
        user["Credit_Score_sq"] = user["Credit_Score"]**2

        # One-hot encode
        cat = user[[
            "Employment_Status",
            "Marital_Status",
            "Loan_Purpose",
            "Property_Area",
            "Gender",
            "Employer_Category"
        ]]

        enc = ohe.transform(cat)

        enc_df = pd.DataFrame(
            enc,
            columns=ohe.get_feature_names_out(),
            index=user.index
        )

        user = pd.concat(
            [
                user.drop(columns=[
                    "Employment_Status",
                    "Marital_Status",
                    "Loan_Purpose",
                    "Property_Area",
                    "Gender",
                    "Employer_Category",
                    "Credit_Score",
                    "DTI_Ratio"
                ]),
                enc_df
            ],
            axis=1
        )

        user = user.reindex(columns=X.columns, fill_value=0)

        scaled = scaler.transform(user)

        model = models[best_overall]

        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        st.divider()

        if pred == 1:
            st.success("## ✅ Loan Approved")
        else:
            st.error("## ❌ Loan Rejected")

        st.metric(
            "Approval Probability",
            f"{prob*100:.2f}%"
        )

        st.info(f"Prediction generated using **{best_overall}**")

# =================================================
# MODEL ANALYSIS
# =================================================
else:

    st.header("📊 Model Performance Analysis")

    metric_df = pd.DataFrame({
        "Model": metrics.keys(),
        "Accuracy":[metrics[m]["Accuracy"] for m in metrics],
        "Precision":[metrics[m]["Precision"] for m in metrics],
        "Recall":[metrics[m]["Recall"] for m in metrics],
        "F1 Score":[metrics[m]["F1"] for m in metrics]
    })

    st.subheader("All Three Models")

    st.dataframe(
        metric_df.style.format({
            "Accuracy":"{:.3f}",
            "Precision":"{:.3f}",
            "Recall":"{:.3f}",
            "F1 Score":"{:.3f}"
        }),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.success("🏆 Best Overall Model")
        st.metric("Model", best_overall)
        st.metric("F1 Score", f"{metrics[best_overall]['F1']:.3f}")
        st.metric("Accuracy", f"{metrics[best_overall]['Accuracy']:.3f}")

    with col2:
        st.warning("🎯 Best Precision Model")
        st.metric("Model", best_precision)
        st.metric("Precision", f"{metrics[best_precision]['Precision']:.3f}")
        st.metric("Recall", f"{metrics[best_precision]['Recall']:.3f}")

    st.divider()

    st.subheader("Metric Comparison")

    st.bar_chart(
        metric_df.set_index("Model")[[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]]
    )

    st.divider()

    st.subheader(f"Confusion Matrix — {best_overall}")

    fig, ax = plt.subplots(figsize=(4,3))

    sns.heatmap(
        metrics[best_overall]["CM"],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.divider()

    st.subheader("Correlation Heatmap")

    fig2, ax2 = plt.subplots(figsize=(10,6))

    sns.heatmap(
        df.corr(numeric_only=True),
        cmap="coolwarm",
        ax=ax2
    )

    st.pyplot(fig2)