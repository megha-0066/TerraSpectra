import streamlit as st

from app.components.header import render_header, render_hero
from app.components.charts import spectral_chart

def render():

    render_header(
        "Spectral Analysis",
        "Explore wavelength patterns and spectral signatures."
    )

    render_hero(
        "SPECTRAL INTELLIGENCE",
        "Understand What the Eye Cannot See.",
        "Study reflectance patterns across different wavelength bands."
    )

    st.plotly_chart(
        spectral_chart(),
        use_container_width=True
    )

    st.caption(
        "Illustrative data only. Actual signatures will come from the verified dataset."
    )
