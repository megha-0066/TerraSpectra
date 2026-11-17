import streamlit as st

from app.components.header import render_header, render_hero

def render():

    render_header(
        "About TerraSpectra",
        "Technology, purpose and project information."
    )

    render_hero(
        "OUR MISSION",
        "Smarter Agriculture Through Spectral Intelligence.",
        "TerraSpectra explores hyperspectral imaging and artificial "
        "intelligence for crop health analysis and disease classification."
    )

    st.markdown("### Technology Stack")

    columns = st.columns(3)

    technologies = [
        ("Python", "Core programming"),
        ("Streamlit", "Dashboard framework"),
        ("Plotly", "Interactive visualizations")
    ]

    for col, (name, description) in zip(columns, technologies):
        with col:
            st.markdown(f"**{name}**")
            st.caption(description)

    st.markdown("### Team Structure")

    st.markdown("""
    - Member 1: Data Research Engineer
    - Member 2: Data Scientist
    - Member 3: ML Engineer
    - Member 4: Dashboard and Integration Developer
    """)

    st.warning(
        "Future forecasting requires suitable historical, environmental "
        "or time-series data."
    )
