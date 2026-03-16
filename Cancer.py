# ==============================
# Breast Cancer Detection App
# Built with Streamlit
# ==============================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------
# Title
# ------------------------------
st.title("🩺 Breast Cancer Detection System")
st.markdown("Machine Learning powered web app for predicting *Breast Cancer (Malignant or Benign)*")

# ------------------------------
# Load Dataset
# ------------------------------
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# ------------------------------
# Train Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Train Model
# ------------------------------
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# ------------------------------
# Accuracy
# ------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Data Exploration", "Model Performance", "Prediction"]
)

# ==============================
# HOME PAGE
# ==============================
if page == "Home":

    st.subheader("Project Overview")

    st.write("""
    This web application predicts whether a breast tumor is *Malignant (Cancerous)* or *Benign (Non-Cancerous)* using Machine Learning.
    
    The model is trained on the *Breast Cancer Wisconsin Dataset*.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Samples", X.shape[0])
    col2.metric("Features", X.shape[1])
    col3.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Breast_cancer_cells.jpg/640px-Breast_cancer_cells.jpg",
        caption="Breast Cancer Cells"
    )

# ==============================
# DATA EXPLORATION
# ==============================
elif page == "Data Exploration":

    st.subheader("Dataset")

    df = pd.concat([X, y], axis=1)
    df.rename(columns={0: "target"}, inplace=True)

    st.write(df.head())

    st.subheader("Class Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x=y)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig2, ax2 = plt.subplots(figsize=(12,8))
    sns.heatmap(X.corr(), cmap="coolwarm")
    st.pyplot(fig2)

# ==============================
# MODEL PERFORMANCE
# ==============================
elif page == "Model Performance":

    st.subheader("Model Accuracy")

    st.success(f"Accuracy: {accuracy*100:.2f}%")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    st.pyplot(fig3)

    st.subheader("Classification Report")

    report = classification_report(y_test, y_pred)
    st.text(report)

# ==============================
# PREDICTION PAGE
# ==============================
elif page == "Prediction":

    st.subheader("Enter Tumor Measurements")

    input_data = []

    for feature in X.columns:
        value = st.number_input(feature, float(X[feature].min()), float(X[feature].max()))
        input_data.append(value)

    input_data = np.array(input_data).reshape(1, -1)

    if st.button("Predict"):

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        if prediction[0] == 0:
            st.error("Prediction: Malignant (Cancerous)")
        else:
            st.success("Prediction: Benign (Non-Cancerous)")

        st.subheader("Prediction Probability")

        st.write(f"Malignant: {probability[0][0]*100:.2f}%")
        st.write(f"Benign: {probability[0][1]*100:.2f}%")