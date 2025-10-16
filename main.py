import streamlit as st
import pandas as pd
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(page_title="TKA TJA XAI", page_icon="🦿", layout="wide")


# --- LOAD MODELS AND DATA ---
# Using st.cache_data to avoid reloading on every interaction
@st.cache_data
def load_data_and_models():
    rf_model = joblib.load("models/tha/rf_classifier_CA.pkl")
    xgb_model = joblib.load("models/tha/xgb_classifier_CA.pkl")
    lr_model = joblib.load("models/tha/lr_classifier_CA.pkl")
    df = pd.read_csv("data/undersampled_CDARREST_Hip_21_22_23(2).csv")
    X = df.drop(columns=["CDARREST", "CDMI"])
    y = df["CDARREST"]
    return rf_model, xgb_model, lr_model, X, y


rf_model, xgb_model, lr_model, X, y = load_data_and_models()


# --- HEADER ---
col_logo, col_qr_code = st.columns([1, 1])
with col_logo:
    st.image("https://pitthexai.github.io/assets/img/Pitthexai_logo.png", width=200)
with col_qr_code:
    st.markdown(
        """
        <div style="text-align: right;">
            <img src="https://pitthexai.github.io/images/qr-code.png" width="80">
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- SIDEBAR ---
st.sidebar.title("Controls")
model_choice = st.sidebar.selectbox(
    "Choose Model", ["Random Forest", "XGBoost", "Logistic Regression"]
)

if model_choice == "Random Forest":
    model = rf_model
elif model_choice == "XGBoost":
    model = xgb_model
else:
    model = lr_model

class_names = ["CDMI", "Cardiac Arrest"]
class_name = st.sidebar.selectbox("Select class", class_names)
class_id = class_names.index(class_name)
row_id = st.sidebar.number_input("Select patient row (index)", 0, len(X) - 1, 0)

# Get instance
instance = pd.DataFrame([X.iloc[row_id, :]], columns=X.columns)

# Prediction
proba = model.predict_proba(instance)[0, class_id]
st.sidebar.metric(
    label=f"Predicted Probability ({class_names[class_id]})", value=f"{proba:.2f}"
)


# --- MAIN PANEL ---
st.write("### Selected Patient Data")
st.dataframe(instance)


# Create tabs for different explanations
tab1, tab2, tab3 = st.tabs(
    ["Global SHAP Summary", "Local SHAP Explanation", "LIME Explanation"]
)


# --- SHAP EXPLANATIONS ---
# This part is computationally expensive, so it's good to cache it.
@st.cache_data
def get_shap_values(_model, _X):
    explainer = (
        shap.TreeExplainer(_model)
        if model_choice != "Logistic Regression"
        else shap.Explainer(_model, _X)
    )
    shap_values = explainer.shap_values(_X)
    return explainer, shap_values


explainer, shap_values = get_shap_values(model, X)


if shap_values.ndim == 3:
    # tree models (RandomForest, sometimes XGB with TreeExplainer)
    values_for_class = shap_values[:, :, class_id]
    shap_row_for_class = shap_values[row_id, :, class_id]
    expected_value_for_class = explainer.expected_value[class_id]
else:
    # XGB (Explainer) or Logistic Regression
    values_for_class = shap_values
    shap_row_for_class = shap_values[row_id, :]
    expected_value_for_class = explainer.expected_value


# --- TAB 1: GLOBAL SHAP ---
with tab1:
    st.write("#### Global Feature Importance (SHAP)")

    fig, ax = plt.subplots()
    shap.summary_plot(
        values_for_class,
        X,
        show=False,
        plot_type="bar",
        color_bar=True,
        plot_size=(15, 5),
    )

    st.pyplot(fig, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# --- TAB 2: LOCAL SHAP ---
with tab2:
    st.write("#### Local Explanation for Selected Patient (SHAP)")

    # The force_plot function has a figsize parameter
    fig = shap.force_plot(
        expected_value_for_class,
        shap_row_for_class,
        X.iloc[row_id, :],
        matplotlib=True,
        show=False,
        figsize=(12, 5),  # Set figure size directly
    )
    plt.tight_layout()
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)


# --- TAB 3: LIME EXPLANATION ---
with tab3:
    st.write("#### Local Explanation for Selected Patient (LIME)")

    # Setup LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=["No Arrest", "Arrest"],
        mode="classification",
    )

    # Get LIME explanation for the instance
    lime_exp = lime_explainer.explain_instance(
        instance.values[0], model.predict_proba, num_features=10, labels=(class_id,)
    )

    # Generate the LIME plot
    fig = lime_exp.as_pyplot_figure(label=class_id)

    # Resize and adjust the layout to match SHAP
    fig.set_size_inches(10, 3)  # same as SHAP tab
    plt.subplots_adjust(left=0.3)  # give room for feature labels
    plt.tight_layout()

    st.pyplot(fig, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
