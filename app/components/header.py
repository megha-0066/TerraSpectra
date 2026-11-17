import streamlit as st

def render_header(title, subtitle):

    st.markdown(
        f'<div class="page-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )

def render_hero(tag, title, description):

    st.markdown(f"""
    <div class="hero">
        <div class="hero-tag">{tag}</div>
        <div class="hero-title">{title}</div>
        <div class="hero-text">{description}</div>
    </div>
    """, unsafe_allow_html=True)
