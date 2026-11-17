import streamlit as st

from app.components.header import render_header, render_hero
from app.components.metric_cards import render_metric
from app.components.charts import spectral_chart, health_distribution

def render():

    render_header(
        "Crop Intelligence Dashboard",
        "Your central workspace for hyperspectral crop health analytics."
    )

    render_hero(
        "WELCOME TO TERRASPECTRA",
        "See Beyond the Visible. Understand Crop Health.",
        "Explore hyperspectral signatures, analyse crop health patterns "
        "and use artificial intelligence to support agricultural disease classification."
    )

    st.markdown("### Platform Overview")

    columns = st.columns(4)

    metrics = [
        ("🌾", "TOTAL SAMPLES", "1,250", "Illustrative demo data"),
        ("🌱", "HEALTHY SAMPLES", "875", "Illustrative demo data"),
        ("⚠️", "DISEASED SAMPLES", "375", "Illustrative demo data"),
        ("🎯", "MODEL ACCURACY", "92.4%", "Illustrative demo value")
    ]

    for col, data in zip(columns, metrics):
        with col:
            render_metric(*data)

    st.markdown("### Spectral Intelligence")

    left, right = st.columns([1.7, 1])

    with left:
        st.markdown("**Spectral Signature Analysis**")
        st.plotly_chart(
            spectral_chart(),
            use_container_width=True
        )

    with right:
        st.markdown("**Crop Health Distribution**")
        st.plotly_chart(
            health_distribution(),
            use_container_width=True
        )

    st.markdown("### Workspace Activity")

    st.info("Frontend architecture and dashboard development in progress.")

    st.caption(
        "All dashboard figures currently use illustrative demo data."
    )
