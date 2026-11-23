import streamlit as st
import pandas as pd

from app.components.header import render_header, render_hero


def load_csv(uploaded_file):
    uploaded_file.seek(0)

    try:
        return pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python",
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python",
            encoding="cp1252"
        )


def render():

    render_header(
        "Crop Disease Detection",
        "Upload and analyse your hyperspectral crop dataset."
    )

    render_hero(
        "DATASET ANALYSIS",
        "Analyse Your Crop Samples",
        "Upload a CSV dataset to explore its structure and spectral information."
    )

    uploaded = st.file_uploader(
        "Upload hyperspectral CSV dataset",
        type=["csv"],
        key="hyperspectral_csv"
    )

    if uploaded is None:
        st.info("Upload a CSV dataset to begin the analysis.")

        if "hyperspectral_df" in st.session_state:
            st.caption(
                "A previously uploaded dataset is available in this session."
            )
        return

    try:
        df = load_csv(uploaded)

        if df.empty:
            st.warning("The uploaded CSV contains no records.")
            return

        st.session_state["hyperspectral_df"] = df
        st.session_state["hyperspectral_filename"] = uploaded.name

        st.success(f"Dataset loaded: {uploaded.name}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", f"{df.shape[0]:,}")
        col2.metric("Total Columns", f"{df.shape[1]:,}")
        col3.metric(
            "Missing Values",
            f"{int(df.isna().sum().sum()):,}"
        )

        st.markdown("---")
        st.markdown("## Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("## Data Quality Analysis")

        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            st.success("No missing values detected.")
        else:
            missing_df = pd.DataFrame({
                "Column": missing.index,
                "Missing Values": missing.values,
                "Missing Percentage": (
                    missing.values / len(df) * 100
                ).round(2)
            })

            st.dataframe(
                missing_df,
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("## Spectral Data Identification")

        spectral_columns = [
            col for col in df.columns
            if str(col).upper().startswith(("VNIR_", "SWIR_"))
        ]

        if spectral_columns:
            st.success(
                f"Detected {len(spectral_columns)} spectral columns."
            )
            st.write(spectral_columns[:20])
        else:
            st.info(
                "No VNIR_ or SWIR_ prefixed columns detected."
            )

        st.markdown("---")
        st.markdown("## Numerical Statistics")

        numeric_df = df.select_dtypes(include="number")

        if not numeric_df.empty:
            st.dataframe(
                numeric_df.describe().round(3),
                use_container_width=True
            )
        else:
            st.info("No numerical columns available.")

        st.success(
            "Dataset analysis completed. "
            "Disease prediction model integration is pending."
        )

    except Exception as error:
        st.error(f"Unable to process CSV: {error}")
