import numpy as np
import plotly.graph_objects as go

def spectral_chart():

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
        paper_bgcolor="#14231a",
        plot_bgcolor="#14231a",
        font=dict(color="#b5c7b8"),
        height=350,
        margin=dict(l=20, r=20, t=30, b=25),
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
        xaxis=dict(gridcolor="#293d30"),
        yaxis=dict(gridcolor="#293d30")
    )

    return fig


def health_distribution():

    fig = go.Figure(data=[go.Pie(
        labels=["Healthy", "Diseased"],
        values=[875, 375],
        hole=0.72,
        marker=dict(colors=["#c5f36a", "#e8a45b"])
    )])

    fig.update_layout(
        paper_bgcolor="#14231a",
        font=dict(color="#b5c7b8"),
        height=350,
        margin=dict(l=15, r=15, t=30, b=20)
    )

    return fig
