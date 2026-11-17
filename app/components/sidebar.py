import streamlit as st

def render_sidebar():

    with st.sidebar:

        st.markdown("""
        <div class="brand">🌱 TerraSpectra</div>
        <div class="brand-sub">AGRICULTURAL INTELLIGENCE</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-label">MAIN MENU</div>',
                    unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            [
                "Overview",
                "Disease Detection",
                "Spectral Analysis",
                "Model Performance",
                "About Project"
            ],
            label_visibility="collapsed"
        )

        st.divider()

        st.markdown('<div class="sidebar-label">SYSTEM STATUS</div>',
                    unsafe_allow_html=True)

        st.success("Application Online")

        st.caption("AI Model: Integration Pending")
        st.caption("Version 1.0.0")

        st.divider()

        st.caption("Developed by TerraSpectra Team")

    return page
