import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT))
import streamlit as st
from pathlib import Path

from app.components.sidebar import render_sidebar

from app.views import overview
from app.views import disease_detection
from app.views import spectral_analysis
from app.views import model_performance
from app.views import about

st.set_page_config(
    page_title="TerraSpectra | Crop Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

css_path = Path(__file__).parent / "assets" / "styles.css"

with open(css_path, encoding="utf-8") as css_file:
    st.markdown(
        f"<style>{css_file.read()}</style>",
        unsafe_allow_html=True
    )

page = render_sidebar()

pages = {
    "Overview": overview.render,
    "Disease Detection": disease_detection.render,
    "Spectral Analysis": spectral_analysis.render,
    "Model Performance": model_performance.render,
    "About Project": about.render
}

pages[page]()

st.markdown("---")

st.caption(
    "TERRASPECTRA | AI-POWERED AGRICULTURAL INTELLIGENCE | VERSION 1.0"
)


