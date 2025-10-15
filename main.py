import streamlit as st
import pandas as pd
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt

# Load models
rf_model = joblib.load("models/tha/rf_classifier_CA.pkl")
xgb_model = joblib.load("models/tha/xgb_classifier_CA.pkl")
lr_model = joblib.load("models/tha/lr_classifier_CA.pkl")

# Load dataset (for feature names and LIME background)
df = pd.read_csv("data/undersampled_CDARREST_Hip_21_22_23(2).csv")
X = df.drop(columns=["CDARREST", "CDMI"])
y = df["CDARREST"]

# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Choose Model", ["Random Forest", "XGBoost", "Logistic Regression"]
)

if model_choice == "Random Forest":
    model = rf_model
elif model_choice == "XGBoost":
    model = xgb_model
else:
    model = lr_model

# Sidebar for instance selection
class_names = ["CDMI", "Cardiac Arrest"]
# class_id = st.sidebar.number_input("Select class (id)", 0, 1, 0)
class_name = st.sidebar.selectbox("Select class", class_names)
class_id = class_names.index(class_name)
row_id = st.sidebar.number_input("Select patient row (index)", 0, len(X) - 1, 0)

# Get instance
instance = pd.DataFrame([X.iloc[row_id, :]], columns=X.columns)

col_logo, _, col_qr_code = st.columns([1, 5, 1])

with col_logo:
    st.image("https://pitthexai.github.io/assets/img/Pitthexai_logo.png", width=300)

with col_qr_code:
    st.image("https://pitthexai.github.io/images/qr-code.png", width=120)

st.set_page_config(page_title="TKA TJA XAI", page_icon="🦿", layout="wide")


st.write("### Selected Patient Data")
st.write(instance)

# Prediction
proba = model.predict_proba(instance)[0, class_id]

st.sidebar.text(f"Predicted Probability ({class_names[class_id]}): {proba:.2f}")


# SHAP Explanation
explainer = (
    shap.TreeExplainer(model)
    if model_choice != "Logistic Regression"
    else shap.Explainer(model, X)
)
shap_values = explainer.shap_values(X)  # type: ignore

if shap_values.ndim == 3:
    # tree models (RandomForest, sometimes XGB with TreeExplainer)
    values = shap_values[:, :, class_id]
    shap_row = shap_values[row_id, :, class_id]
    expected_value = explainer.expected_value[class_id]  # type: ignore
else:
    # XGB (Explainer) or Logistic Regression
    values = shap_values  # already 2D
    shap_row = shap_values[row_id, :]
    expected_value = explainer.expected_value  # type: ignore

col1, col2, col3 = st.columns(3)

with col1:
    st.write("### Global SHAP Summary")
    fig, ax = plt.subplots()
    shap.summary_plot(values, X, show=False, plot_type="bar")  # or "dot"
    st.pyplot(fig)

# Local SHAP explanation
with col2:
    st.write("### Local SHAP (Selected Patient)")

    fig = shap.force_plot(
        expected_value,  # type: ignore
        shap_row,
        X.iloc[row_id, :],
        matplotlib=True,
        show=False,
    )
    st.pyplot(fig)

# LIME Explanation
with col3:
    st.write("### LIME Explanation")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=["No Arrest", "Arrest"],
        mode="classification",
    )

    lime_exp = lime_explainer.explain_instance(
        instance.values[0], model.predict_proba, num_features=10
    )

    fig = lime_exp.as_pyplot_figure()
    st.pyplot(fig)
