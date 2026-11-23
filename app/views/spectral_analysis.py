import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from app.components.header import render_header, render_hero


def render():

    render_header(
        "Spectral Analysis",
        "Explore spectral patterns from uploaded hyperspectral data."
    )

    render_hero(
        "SPECTRAL INTELLIGENCE",
        "Understand What the Eye Cannot See.",
        "Explore VNIR and SWIR measurements from your uploaded dataset."
    )

    df = st.session_state.get("hyperspectral_df")

    if df is None:
        st.warning(
            "No dataset available. Upload a CSV first in Disease Detection."
        )
        return

    st.success(
        f"Using dataset: {st.session_state.get('hyperspectral_filename', 'Uploaded CSV')}"
    )

    spectral_columns = [
        col for col in df.columns
        if str(col).upper().startswith(("VNIR_", "SWIR_"))
        and pd.to_numeric(df[col], errors="coerce").notna().any()
    ]

    if not spectral_columns:
        st.warning("No numeric VNIR or SWIR columns found.")
        return

    vnir_columns = [
        col for col in spectral_columns
        if str(col).upper().startswith("VNIR_")
    ]

    swir_columns = [
        col for col in spectral_columns
        if str(col).upper().startswith("SWIR_")
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric("Spectral Features", len(spectral_columns))
    col2.metric("VNIR Features", len(vnir_columns))
    col3.metric("SWIR Features", len(swir_columns))

    band = st.selectbox(
        "Select spectral region",
        ["All", "VNIR", "SWIR"]
    )

    if band == "VNIR":
        selected_columns = vnir_columns
    elif band == "SWIR":
        selected_columns = swir_columns
    else:
        selected_columns = spectral_columns

    if not selected_columns:
        st.warning("No columns available for this region.")
        return

    selected_columns = sorted(selected_columns)

    chosen_columns = st.multiselect(
        "Select spectral columns",
        selected_columns,
        default=selected_columns[:min(30, len(selected_columns))]
    )

    if not chosen_columns:
        st.info("Select at least one spectral column.")
        return

    numeric_data = df[chosen_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    average_values = numeric_data.mean()

    def get_wavelength(column):
        numbers = re.findall(r"\d+(?:\.\d+)?", str(column))
        return float(numbers[-1]) if numbers else None

    wavelengths = [
        get_wavelength(column) for column in chosen_columns
    ]

    if all(value is not None for value in wavelengths):
        plot_data = pd.DataFrame({
            "Wavelength": wavelengths,
            "Reflectance": average_values.values
        }).sort_values("Wavelength")

        x_values = plot_data["Wavelength"]
        y_values = plot_data["Reflectance"]
        x_title = "Wavelength (nm)"
    else:
        x_values = chosen_columns
        y_values = average_values.values
        x_title = "Spectral Feature"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines+markers",
        name="Mean Spectral Signature",
        line=dict(color="#c5f36a", width=3),
        marker=dict(size=6)
    ))

    fig.update_layout(
        paper_bgcolor="#14231a",
        plot_bgcolor="#14231a",
        font=dict(color="#b5c7b8"),
        height=450,
        xaxis_title=x_title,
        yaxis_title="Mean Reflectance / Feature Value",
        xaxis=dict(gridcolor="#293d30"),
        yaxis=dict(gridcolor="#293d30"),
        margin=dict(l=30, r=20, t=30, b=40)
    )

    st.markdown("## Dataset-Based Spectral Signature")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Mean values calculated from the uploaded dataset. "
        "This is not an ML disease prediction."
    )

    st.markdown("## Spectral Statistics")

    st.dataframe(
        numeric_data.describe().round(4),
        use_container_width=True
    )
