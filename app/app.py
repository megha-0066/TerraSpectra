import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ==========================================
# TERRASPECTRA - MAIN APPLICATION
# ==========================================

st.set_page_config(
    page_title="TerraSpectra | Crop Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --background: #0b1510;
    --surface: #14231a;
    --surface-light: #1a2d21;
    --border: #293d30;
    --green: #c5f36a;
    --text: #edf4ed;
    --muted: #91a495;
}

.stApp {
    background: var(--background);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: #101e16;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

.brand {
    font-family: 'Outfit', sans-serif;
    font-size: 29px;
    font-weight: 800;
    color: var(--green);
    letter-spacing: -1px;
}

.brand-sub {
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 2px;
    margin-top: -5px;
}

.sidebar-label {
    color: #758b7b;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 30px;
}

.page-title {
    font-family: 'Outfit', sans-serif;
    font-size: 37px;
    font-weight: 700;
    letter-spacing: -1px;
    color: var(--text);
}

.page-subtitle {
    color: var(--muted);
    font-size: 14px;
    margin-top: -8px;
    margin-bottom: 28px;
}

.hero {
    background: linear-gradient(110deg, #1b3826, #17271d 65%, #263b25);
    border: 1px solid #36533a;
    border-radius: 18px;
    padding: 28px 30px;
    margin-bottom: 25px;
}

.hero-tag {
    color: var(--green);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 29px;
    font-weight: 700;
    color: #f2f8ed;
    margin-top: 10px;
}

.hero-text {
    color: #b4c8b7;
    font-size: 14px;
    max-width: 650px;
    line-height: 1.7;
}

.metric-card {
    background: linear-gradient(145deg, #192b20, #142119);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
    min-height: 145px;
}

.metric-icon {
    font-size: 20px;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    margin-top: 13px;
}

.metric-value {
    color: var(--green);
    font-family: 'Outfit', sans-serif;
    font-size: 31px;
    font-weight: 700;
    margin-top: 3px;
}

.metric-note {
    color: #829789;
    font-size: 11px;
    margin-top: 5px;
}

.section-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 21px;
    font-weight: 600;
    color: #edf4ed;
    margin-top: 28px;
    margin-bottom: 12px;
}

.panel {
    background: #14231a;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
}

.panel-title {
    font-family: 'Outfit', sans-serif;
    font-size: 17px;
    font-weight: 600;
    color: #edf4ed;
}

.panel-subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-top: 4px;
}

.status {
    display: inline-block;
    background: #253e29;
    color: #c5f36a;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}

.activity {
    padding: 13px 0;
    border-bottom: 1px solid var(--border);
}

.activity:last-child {
    border-bottom: none;
}

.activity-name {
    color: #e5eee5;
    font-size: 13px;
    font-weight: 600;
}

.activity-time {
    color: var(--muted);
    font-size: 11px;
}

.footer {
    border-top: 1px solid var(--border);
    padding-top: 18px;
    margin-top: 35px;
    color: #74887a;
    font-size: 11px;
}

div[data-testid="stFileUploader"] {
    background: #14231a;
    border: 1px dashed #45644b;
    border-radius: 14px;
    padding: 15px;
}

.stButton button {
    background: var(--green);
    color: #102016;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    padding: 9px 20px;
}

.stButton button:hover {
    background: #dcff99;
    color: #102016;
}

div[data-testid="stMetric"] {
    background: #192b20;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown('<div class="brand">🌱 TerraSpectra</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="brand-sub">AGRICULTURAL INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-label">MAIN MENU</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="sidebar-label">SYSTEM STATUS</div>', unsafe_allow_html=True)

    st.markdown(
        '<span class="status">● Application Online</span>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption("AI Model: Integration Pending")

    st.caption("Version 1.0.0")

    st.divider()

    st.caption("Developed by TerraSpectra Team")


# ==========================================
# REUSABLE COMPONENTS
# ==========================================

def page_header(title, subtitle):

    st.markdown(
        f'<div class="page-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def metric_card(icon, label, value, note):

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def chart_layout(fig, height=340):

    fig.update_layout(
        paper_bgcolor="#14231a",
        plot_bgcolor="#14231a",
        font=dict(color="#b5c7b8"),
        height=height,
        margin=dict(l=20, r=20, t=30, b=25),
        xaxis=dict(
            gridcolor="#293d30",
            zerolinecolor="#293d30"
        ),
        yaxis=dict(
            gridcolor="#293d30",
            zerolinecolor="#293d30"
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )

    return fig


# ==========================================
# OVERVIEW PAGE
# ==========================================

if page == "Overview":

    page_header(
        "Crop Intelligence Dashboard",
        "Your central workspace for hyperspectral crop health analytics."
    )

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">WELCOME TO TERRASPECTRA</div>
        <div class="hero-title">See Beyond the Visible. Understand Crop Health.</div>
        <div class="hero-text">
            Explore hyperspectral signatures, analyse crop health patterns
            and use artificial intelligence to support agricultural
            disease classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI SECTION

    st.markdown(
        '<div class="section-heading">Platform Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("🌾", "TOTAL SAMPLES", "1,250", "Illustrative demo data")

    with c2:
        metric_card("🌱", "HEALTHY SAMPLES", "875", "Illustrative demo data")

    with c3:
        metric_card("⚠️", "DISEASED SAMPLES", "375", "Illustrative demo data")

    with c4:
        metric_card("🎯", "MODEL ACCURACY", "92.4%", "Illustrative demo value")

    # CHARTS

    st.markdown(
        '<div class="section-heading">Spectral Intelligence</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.7, 1])

    wavelengths = np.linspace(400, 1000, 120)

    healthy = (
        0.2
        + 0.5 / (1 + np.exp(-(wavelengths - 700) / 25))
        + 0.015 * np.sin(wavelengths / 20)
    )

    stressed = (
        0.2
        + 0.35 / (1 + np.exp(-(wavelengths - 710) / 30))
        + 0.015 * np.cos(wavelengths / 22)
    )

    with left:

        st.markdown("""
        <div class="panel">
            <div class="panel-title">Spectral Signature Analysis</div>
            <div class="panel-subtitle">
                Illustrative reflectance patterns across wavelengths
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=wavelengths,
            y=healthy,
            name="Healthy Crop",
            line=dict(color="#c5f36a", width=3)
        ))

        fig.add_trace(go.Scatter(
            x=wavelengths,
            y=stressed,
            name="Stressed Crop",
            line=dict(color="#e8a45b", width=3)
        ))

        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Reflectance",
            legend=dict(orientation="h", y=1.15)
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    with right:

        st.markdown("""
        <div class="panel">
            <div class="panel-title">Crop Health Distribution</div>
            <div class="panel-subtitle">
                Illustrative sample distribution
            </div>
        </div>
        """, unsafe_allow_html=True)

        pie = go.Figure(data=[go.Pie(
            labels=["Healthy", "Diseased"],
            values=[875, 375],
            hole=0.72,
            marker=dict(
                colors=["#c5f36a", "#e8a45b"],
                line=dict(color="#14231a", width=4)
            ),
            textinfo="label+percent"
        )])

        pie.update_layout(
            showlegend=False,
            annotations=[dict(
                text="1,250<br>Samples",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=17, color="#edf4ed")
            )]
        )

        st.plotly_chart(
            chart_layout(pie),
            use_container_width=True
        )

    # ACTIVITY

    st.markdown(
        '<div class="section-heading">Workspace Activity</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns([1.4, 1])

    with a:

        st.markdown("""
        <div class="panel">
            <div class="panel-title">Project Progress</div>
            <br>
            <div class="activity">
                <div class="activity-name">Dashboard Environment Setup</div>
                <div class="activity-time">Completed</div>
            </div>
            <div class="activity">
                <div class="activity-name">Frontend Architecture</div>
                <div class="activity-time">In Development</div>
            </div>
            <div class="activity">
                <div class="activity-name">ML Model Integration</div>
                <div class="activity-time">Awaiting Model Module</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with b:

        st.markdown("""
        <div class="panel">
            <div class="panel-title">System Information</div>
            <br>
            <div class="activity">
                <div class="activity-name">Application</div>
                <div class="activity-time">TerraSpectra</div>
            </div>
            <div class="activity">
                <div class="activity-name">Framework</div>
                <div class="activity-time">Python + Streamlit</div>
            </div>
            <div class="activity">
                <div class="activity-name">Visualization</div>
                <div class="activity-time">Plotly</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# DISEASE DETECTION PAGE
# ==========================================

elif page == "Disease Detection":

    page_header(
        "Crop Disease Detection",
        "Upload supported hyperspectral data for analysis."
    )

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">AI-POWERED ANALYSIS</div>
        <div class="hero-title">Analyse Your Crop Samples</div>
        <div class="hero-text">
            Upload a supported dataset to begin the analysis workflow.
            Prediction functionality will be connected to the verified
            preprocessing and ML pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a hyperspectral data file",
        type=["csv", "npy", "npz", "tif", "tiff"]
    )

    if uploaded:

        st.success(f"File received: {uploaded.name}")

        st.write("File size:", round(uploaded.size / 1024, 2), "KB")

        st.info(
            "Upload successful. Model-based prediction is not enabled yet."
        )


# ==========================================
# SPECTRAL ANALYSIS PAGE
# ==========================================

elif page == "Spectral Analysis":

    page_header(
        "Spectral Analysis",
        "Explore wavelength patterns and spectral signatures."
    )

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">SPECTRAL INTELLIGENCE</div>
        <div class="hero-title">Understand What the Eye Cannot See.</div>
        <div class="hero-text">
            Hyperspectral analysis studies reflectance across multiple
            wavelength bands to support crop health investigation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    wavelengths = np.linspace(400, 1000, 120)

    reflectance = (
        0.2
        + 0.5 / (1 + np.exp(-(wavelengths - 700) / 25))
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=reflectance,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#c5f36a", width=3),
        name="Illustrative Signature"
    ))

    fig.update_layout(
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance"
    )

    st.plotly_chart(
        chart_layout(fig, 400),
        use_container_width=True
    )

    st.caption(
        "Illustrative data only. Actual spectral signatures will come from the verified dataset."
    )


# ==========================================
# MODEL PERFORMANCE PAGE
# ==========================================

elif page == "Model Performance":

    page_header(
        "Model Performance",
        "Evaluation metrics and machine learning validation."
    )

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">MODEL EVALUATION</div>
        <div class="hero-title">Measure. Validate. Improve.</div>
        <div class="hero-text">
            Performance metrics will be populated after the ML engineer
            provides the trained model and evaluation artifacts.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Model evaluation is pending. No actual accuracy or classification metrics are available yet."
    )

    metrics = st.columns(4)

    for col, name in zip(
        metrics,
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ):

        with col:

            st.metric(
                name,
                "Pending",
                help="Will be populated using actual test-set results."
            )


# ==========================================
# ABOUT PAGE
# ==========================================

elif page == "About Project":

    page_header(
        "About TerraSpectra",
        "Technology, purpose and project information."
    )

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">OUR MISSION</div>
        <div class="hero-title">Smarter Agriculture Through Spectral Intelligence.</div>
        <div class="hero-text">
            TerraSpectra explores the use of hyperspectral imaging
            and artificial intelligence for crop health analysis
            and disease classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Technology Stack")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Python**")
        st.caption("Core programming")

    with c2:
        st.markdown("**Streamlit**")
        st.caption("Dashboard framework")

    with c3:
        st.markdown("**Plotly**")
        st.caption("Interactive visualizations")

    st.markdown("### Team Structure")

    st.markdown("""
    - Member 1: Data Research Engineer
    - Member 2: Data Scientist
    - Member 3: ML Engineer
    - Member 4: Dashboard and Integration Developer
    """)

    st.warning(
        "Future disease forecasting requires suitable historical, "
        "environmental or time-series data. Static image classification "
        "alone does not establish a future forecast."
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    TERRASPECTRA · AI-POWERED AGRICULTURAL INTELLIGENCE
    <br><br>
    Hyperspectral Crop Disease Forecasting · Development Version 1.0
</div>
""", unsafe_allow_html=True)

