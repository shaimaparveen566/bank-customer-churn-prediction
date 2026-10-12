
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -------------------- PAGE CONFIG --------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"

st.set_page_config(
    page_title="ChurnIQ | Customer Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------- CUSTOM STYLING --------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F7FB;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #102A43, #176B87);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
    }

    .hero h1,
    .hero p {
        color: #FFFFFF !important;
    }

    /* Readable text on the light background */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp p,
    .stApp label,
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricValue"] {
        color: #243447;
    }

    /* Keep form labels dark and readable */
    .stApp [data-testid="stWidgetLabel"] p {
        color: #243447 !important;
    }

    /* White metric cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE5EF;
        padding: 1rem;
        border-radius: 12px;
    }

    /* White form container */
    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        padding: 1.4rem;
        border: 1px solid #DCE5EF;
        border-radius: 16px;
    }

    /* Teal submit button */
    div[data-testid="stFormSubmitButton"] button {
        background-color: #176B87;
        color: #FFFFFF !important;
        border: none;
        border-radius: 9px;
        min-height: 45px;
        font-weight: 600;
    }

    div[data-testid="stFormSubmitButton"] button p {
        color: #FFFFFF !important;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #102A43;
        border: none;
    }

    .footnote {
        color: #64748B !important;
        font-size: 0.85rem;
        text-align: center;
        padding-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error(
        "Could not load the trained model. "
        "Please check that churn_model.joblib exists "
        "and that the training environment is compatible."
    )
    st.stop()

# -------------------- HEADER --------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏦 ChurnIQ</h1>
        <p>Bank Customer Analytics &nbsp; • &nbsp;
        Machine Learning Risk Assessment</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "Understand customer retention risk using a trained "
    "machine-learning model. Enter a customer's details below "
    "to generate an estimate."
)

# -------------------- TOP INFORMATION --------------------
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Model", "Random Forest")

with metric2:
    st.metric("Training records", "8,000")

with metric3:
    st.metric("Test accuracy", "85.9%")

st.caption(
    "These model metrics come from our current experiment and "
    "are not a guarantee of future performance."
)

st.divider()

# -------------------- CUSTOMER INPUT FORM --------------------
st.markdown(
    '<div class="section-label">Customer assessment</div>',
    unsafe_allow_html=True,
)
st.header("Customer Profile")
st.write("Enter the available information to estimate churn risk.")

with st.form("customer_form"):
    st.subheader("Personal & Account Details")

    left, middle, right = st.columns(3)

    with left:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650,
            help="Customer's credit score.",
        )
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
        )
        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"],
        )
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

    with middle:
        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=10,
            value=5,
        )
        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            format="%.2f",
        )
        num_products = st.selectbox(
            "Number of Products",
            [1, 2, 3, 4],
        )

    with right:
        salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=60000.0,
            step=1000.0,
            format="%.2f",
        )
        has_card = st.selectbox(
            "Has Credit Card?",
            ["Yes", "No"],
        )
        active_member = st.selectbox(
            "Active Member?",
            ["Yes", "No"],
        )

    submitted = st.form_submit_button(
        "Calculate Churn Risk",
        use_container_width=True,
        type="primary",
    )

# -------------------- PREDICTION --------------------
if submitted:
    customer = pd.DataFrame(
        [{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": 1 if has_card == "Yes" else 0,
            "IsActiveMember": 1 if active_member == "Yes" else 0,
            "EstimatedSalary": salary,
        }]
    )

    prediction = int(model.predict(customer)[0])
    probabilities = model.predict_proba(customer)[0]
    churn_probability = float(probabilities[1])

    st.divider()
    st.markdown(
        '<div class="section-label">Analysis result</div>',
        unsafe_allow_html=True,
    )
    st.header("Churn Risk Assessment")

    result_col, probability_col = st.columns([1, 1])

    with result_col:
        if prediction == 1:
            st.error("⚠️ Model prediction: Potential churn")
            st.write(
                "The model classifies this customer's profile "
                "as likely to churn."
            )
        else:
            st.success("✅ Model prediction: Likely to stay")
            st.write(
                "The model classifies this customer's profile "
                "as likely to remain with the bank."
            )

    with probability_col:
        st.metric(
            "Estimated churn probability",
            f"{churn_probability:.1%}",
        )
        st.progress(churn_probability)

        if churn_probability < 0.30:
            st.caption("Lower model-estimated churn probability")
        elif churn_probability < 0.60:
            st.caption("Moderate model-estimated churn probability")
        else:
            st.caption("Higher model-estimated churn probability")

    with st.expander("View submitted customer details"):
        st.dataframe(customer, use_container_width=True)

    st.info(
        "This result is a model estimate based on historical data. "
        "It should not be used as the sole basis for decisions "
        "about an individual customer."
    )


# -------------------- MODEL INSIGHTS --------------------
st.divider()

st.markdown(
    '<div class="section-label">Model explainability</div>',
    unsafe_allow_html=True,
)

st.header("What Influences Churn Predictions?")
st.write(
    "Explore which customer attributes the Random Forest "
    "uses most when making its predictions."
)

# Get the trained Random Forest from the saved pipeline
classifier = model.named_steps["classifier"]

# Get feature names after preprocessing
preprocessor = model.named_steps["preprocessor"]
feature_names = preprocessor.get_feature_names_out()

# Extract importance scores
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": classifier.feature_importances_,
})

# Clean up feature names for display
importance_df["Feature"] = (
    importance_df["Feature"]
    .str.replace("categorical__", "", regex=False)
    .str.replace("numerical__", "", regex=False)
    .str.replace("Geography_", "Geography: ", regex=False)
    .str.replace("Gender_", "Gender: ", regex=False)
)

# Display the 10 most important features
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False,
).head(10)

left, right = st.columns([3, 2])

with left:
    st.subheader("Top 10 Features")
    chart_data = importance_df.set_index("Feature")
    st.bar_chart(
        chart_data,
        x_label="Customer Feature",
        y_label="Importance Score",
        horizontal=True,
    )

with right:
    st.subheader("Importance Scores")
    display_df = importance_df.copy()
    display_df["Importance"] = display_df["Importance"].map(
        lambda value: f"{value:.4f}"
    )
    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Higher importance means the feature contributed more to "
    "the Random Forest's decisions overall. These scores do not "
    "show whether a feature increases or decreases churn, and "
    "they do not establish cause and effect."
)


# -------------------- MODEL EVALUATION --------------------
st.divider()

st.markdown(
    '<div class="section-label">Model performance</div>',
    unsafe_allow_html=True,
)

st.header("How Well Does the Model Predict Churn?")
st.write(
    "The confusion matrix compares the model's predictions "
    "with the actual outcomes in the test dataset."
)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def get_test_results():
    df = pd.read_csv(BASE_DIR / "data" / "Churn_Modelling.csv")

    X = df.drop(
        columns=["RowNumber", "CustomerId", "Surname", "Exited"]
    )
    y = df["Exited"]

    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    predictions = model.predict(X_test)

    return y_test, predictions


y_test, predictions = get_test_results()

cm = confusion_matrix(y_test, predictions)

fig, ax = plt.subplots(figsize=(7, 4.5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=1,
    linecolor="white",
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"],
    ax=ax,
)

ax.set_xlabel("Predicted Outcome")
ax.set_ylabel("Actual Outcome")
ax.set_title("Random Forest — Confusion Matrix")

st.pyplot(fig)
plt.close(fig)

tn, fp, fn, tp = cm.ravel()

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Correctly Stayed", f"{tn:,}")

with metric2:
    st.metric("False Churn Alerts", f"{fp:,}")

with metric3:
    st.metric("Missed Churners", f"{fn:,}")

with metric4:
    st.metric("Correctly Detected Churn", f"{tp:,}")

st.caption(
    "Rows represent actual outcomes; columns represent model "
    "predictions. A false churn alert means the model predicted "
    "churn for a customer who actually stayed. A missed churner "
    "is a customer who left but was predicted to stay."
)


# -------------------- FOOTER --------------------
st.markdown(
    """
    <div class="footnote">
        ChurnIQ · Educational Machine Learning Project ·
        Predictions are estimates, not guarantees.
    </div>
    """,
    unsafe_allow_html=True,
)
