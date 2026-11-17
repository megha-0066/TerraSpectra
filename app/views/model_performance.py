import streamlit as st

from app.components.header import render_header, render_hero

def render():

    render_header(
        "Model Performance",
        "Evaluation metrics and machine learning validation."
    )

    render_hero(
        "MODEL EVALUATION",
        "Measure. Validate. Improve.",
        "Model metrics will be populated after integration of the trained model."
    )

    st.info(
        "Model evaluation is pending. Actual test-set metrics are not available yet."
    )

    columns = st.columns(4)

    for col, name in zip(
        columns,
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ):

        with col:
            st.metric(name, "Pending")
