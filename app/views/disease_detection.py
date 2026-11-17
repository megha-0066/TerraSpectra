import streamlit as st

from app.components.header import render_header, render_hero

def render():

    render_header(
        "Crop Disease Detection",
        "Upload supported hyperspectral data for analysis."
    )

    render_hero(
        "AI-POWERED ANALYSIS",
        "Analyse Your Crop Samples",
        "Upload a supported dataset to begin the analysis workflow."
    )

    uploaded = st.file_uploader(
        "Choose a hyperspectral data file",
        type=["csv", "npy", "npz", "tif", "tiff"]
    )

    if uploaded:

        st.success(f"File received: {uploaded.name}")

        st.write(
            "File size:",
            round(uploaded.size / 1024, 2),
            "KB"
        )

        st.info(
            "Upload successful. Model integration is pending."
        )

    else:

        st.info(
            "Upload a supported hyperspectral dataset to begin."
        )
