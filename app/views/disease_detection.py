import streamlit as st
import pandas as pd
from io import BytesIO

from app.components.header import render_header, render_hero


def load_csv(uploaded_file):
    """Read CSV files with common encoding support."""
    uploaded_file.seek(0)

    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=None, engine="python", encoding="cp1252")


def render():

    render_header(
        "Crop Disease Detection",
        "Upload and analyse your hyperspectral crop dataset."
    )

    render_hero(
        "DATASET ANALYSIS",
        "Analyse Your Crop Samples",
        "Upload a CSV dataset to explore its structure, quality and spectral information."
    )

    uploaded = st.file_uploader(
        "Upload hyperspectral CSV dataset",
        type=["csv"],
        help="Upload a CSV file containing crop or spectral measurements."
    )

    if uploaded is None:
        st.info("Upload a CSV dataset to begin the analysis.")
        return

    st.success(f"File uploaded successfully: {uploaded.name}")

    st.caption(
        f"File size: {uploaded.size / 1024:.2f} KB"
    )

    try:
        df = load_csv(uploaded)

        if df.empty:
            st.warning("The uploaded CSV file contains no records.")
            return

        st.markdown("## Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Records", f"{df.shape[0]:,}")

        with col2:
            st.metric("Total Columns", f"{df.shape[1]:,}")

        with col3:
            st.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

        st.markdown("---")

        st.markdown("## Dataset Preview")

        preview_rows = st.slider(
            "Select number of rows to preview",
            min_value=5,
            max_value=min(50, len(df)),
            value=min(10, len(df))
        )

        st.dataframe(
            df.head(preview_rows),
            use_container_width=True
        )

        st.markdown("---")

        st.markdown("## Data Quality Analysis")

        missing_data = df.isnull().sum()

        missing_data = missing_data[missing_data > 0]

        if missing_data.empty:
            st.success("No missing values detected in the dataset.")
        else:
            st.warning(
                f"Missing values detected in {len(missing_data)} columns."
            )

            missing_df = pd.DataFrame({
                "Column": missing_data.index,
                "Missing Values": missing_data.values,
                "Missing Percentage": (
                    missing_data.values / len(df) * 100
                ).round(2)
            })

            st.dataframe(
                missing_df,
                use_container_width=True
            )

        st.markdown("---")

        st.markdown("## Spectral Data Identification")

        spectral_columns = [
            column for column in df.columns
            if str(column).upper().startswith(("VNIR_", "SWIR_"))
        ]

        if spectral_columns:
            st.success(
                f"Detected {len(spectral_columns)} spectral columns."
            )

            st.write("Detected spectral columns:")
            st.write(spectral_columns[:20])

        else:
            st.info(
                "No columns with VNIR_ or SWIR_ prefixes were detected. "
                "The dataset may use a different naming convention."
            )

        st.markdown("---")

        st.markdown("## Numerical Data Statistics")

        numeric_df = df.select_dtypes(include="number")

        if not numeric_df.empty:
            st.dataframe(
                numeric_df.describe().round(3),
                use_container_width=True
            )
        else:
            st.info("No numerical columns available for statistical analysis.")

        st.markdown("---")

        st.markdown("## Dataset Information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.write("**Column Names**")
            st.write(list(df.columns))

        with info_col2:
            st.write("**Data Types**")
            st.write(
                df.dtypes.astype(str).to_dict()
            )

        st.success(
            "Dataset successfully loaded and analysed. "
            "Disease prediction model integration is still pending."
        )

    except Exception as error:
        st.error(f"Unable to process the uploaded CSV file: {error}")
