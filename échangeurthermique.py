"""
Echangeurs de Chaleur - Methode e-NTU
App Streamlit complete avec schema 2D, visualisation 3D Plotly,
page de theorie avec LaTeX et rapport PDF telechargeable.
"""

import io
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import streamlit as st
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Echangeurs e-NTU",
    layout="wide",
)

# ─────────────────────────────────────────────
# PHYSICS
# ─────────────────────────────────────────────

FLOW_TYPES = [
    "Contre-courant",
    "Co-courant",
    "Croise non melange",
    "Croise Cmin melange",
    "Calandre (1 passe) / Tubes (2 passes)",
]


def epsilon_parallel(NTU, Cr):
    return (1.0 - np.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)


def epsilon_counter(NTU, Cr):
    if np.isclose(Cr, 1.0):
        return NTU / (1.0 + NTU)
    a = NTU * (1.0 - Cr)
    return (1.0 - np.exp(-a)) / (1.0 - Cr * np.exp(-a))


def epsilon_crossflow_unmixed(NTU, Cr):
    return 1.0 - np.exp(
        (np.exp(-Cr * NTU ** 0.22) - 1.0) / (Cr * NTU ** (-0.78) + 1e-12)
    )


def epsilon_crossflow_mixed_Cmin(NTU, Cr):
    return (1.0 / Cr) * (1.0 - np.exp(-Cr * (1.0 - np.exp(-NTU))))


def epsilon_shell_tube(NTU, Cr):
    sq = np.sqrt(1.0 + Cr ** 2)
    num = 1.0 + np.exp(-NTU * sq)
    den = 1.0 - np.exp(-NTU * sq)
    if abs(den) < 1e-12:
        return 1.0
    return 2.0 / (1.0 + Cr + sq * num / den)


def compute_outputs(Th_in, Tc_in, mh, cph, mc, cpc, UA, flow_type):
    Ch = mh * cph
    Cc = mc * cpc
    if Ch == 0 or Cc == 0:
        return None
    Cmin = min(Ch, Cc)
    Cmax = max(Ch, Cc)
    Cr = Cmin / Cmax
    NTU = UA / Cmin

    if flow_type == "Co-courant":
        eps = epsilon_parallel(NTU, Cr)
    elif flow_type == "Contre-courant":
        eps = epsilon_counter(NTU, Cr)
    elif flow_type == "Croise non melange":
        eps = epsilon_crossflow_unmixed(NTU, Cr)
    elif flow_type == "Croise Cmin melange":
        eps = epsilon_crossflow_mixed_Cmin(NTU, Cr)
    elif flow_type == "Calandre (1 passe) / Tubes (2 passes)":
        eps = epsilon_shell_tube(NTU, Cr)
    else:
        eps = epsilon_counter(NTU, Cr)

    eps = min(eps, 1.0)
    Qmax = Cmin * (Th_in - Tc_in)
    Q = eps * Qmax
    Th_out = Th_in - Q / Ch
    Tc_out = Tc_in + Q / Cc

    return {
        "Ch": Ch, "Cc": Cc, "Cmin": Cmin, "Cmax": Cmax,
        "Cr": Cr, "NTU": NTU, "epsilon": eps,
        "Qmax": Qmax, "Q": Q,
        "Th_out": Th_out, "Tc_out": Tc_out,
        "flow_type": flow_type,
    }


def solve_UA_for_target(Th_in, Tc_in, mh, cph, mc, cpc, flow_type,
                        target="Tc_out", target_value=40.0):
    UA_low, UA_high = 1e-6, 1e8

    def f(UA):
        r = compute_outputs(Th_in, Tc_in, mh, cph, mc, cpc, UA, flow_type)
        if r is None:
            return np.nan
        return r[target] - target_value

    f_low, f_high = f(UA_low), f(UA_high)
    if np.isnan(f_low) or np.isnan(f_high):
        return None
    if f_low * f_high > 0:
        return None

    UA_mid = UA_low
    for _ in range(100):
        UA_mid = 0.5 * (UA_low + UA_high)
        f_mid = f(UA_mid)
        if abs(f_mid) < 1e-8:
            break
        if f_low * f_mid <= 0:
            UA_high = UA_mid
            f_high = f_mid
        else:
            UA_low = UA_mid
            f_low = f_mid

    return UA_mid


# ─────────────────────────────────────────────
# 2D SCHEMA DRAWING
# ─────────────────────────────────────────────

def draw_schema(res, UA, Th_in, Tc_in):
    flow_type = res["flow_type"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    HOT = "#e74c3c"
    COLD = "#2980b9"
    BODY = "#ecf0f1"
    BORDER = "#2c3e50"
    GREEN = "#27ae60"

    box = FancyBboxPatch(
        (2.5, 1.5), 5, 2,
        boxstyle="round,pad=0.15",
        linewidth=2, edgecolor=BORDER, facecolor=BODY, zorder=2,
    )
    ax.add_patch(box)

    for y_line in [2.0, 2.5, 3.0, 3.5]:
        ax.plot([2.65, 7.35], [y_line, y_line],
                color="#bdc3c7", linewidth=0.8, linestyle="--", zorder=3)

    if flow_type in ("Contre-courant", "Calandre (1 passe) / Tubes (2 passes)"):
        ax.annotate("", xy=(2.5, 3.2), xytext=(0.3, 3.2),
                    arrowprops=dict(arrowstyle="-|>", color=HOT, lw=2.5), zorder=5)
        ax.plot([0.3, 7.7], [3.2, 3.2], color=HOT, lw=2.5, zorder=4)
        ax.annotate("", xy=(9.7, 3.2), xytext=(7.5, 3.2),
                    arrowprops=dict(arrowstyle="-|>", color=HOT, lw=2.5), zorder=5)
        ax.annotate("", xy=(7.5, 2.0), xytext=(9.7, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color=COLD, lw=2.5), zorder=5)
        ax.plot([0.3, 9.7], [2.0, 2.0], color=COLD, lw=2.5, zorder=4)
        ax.annotate("", xy=(0.3, 2.0), xytext=(2.5, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color=COLD, lw=2.5), zorder=5)
        ax.text(0.1, 3.5, f"The\n{Th_in:.1f}C", ha="center", va="center",
                color=HOT, fontsize=10, fontweight="bold")
        ax.text(9.9, 3.5, f"Tfe\n{res['Th_out']:.2f}C", ha="center", va="center",
                color=HOT, fontsize=10, fontweight="bold")
        ax.text(9.9, 1.6, f"Tce\n{Tc_in:.1f}C", ha="center", va="center",
                color=COLD, fontsize=10, fontweight="bold")
        ax.text(0.1, 1.6, f"Tcs\n{res['Tc_out']:.2f}C", ha="center", va="center",
                color=COLD, fontsize=10, fontweight="bold")

    elif flow_type == "Co-courant":
        ax.annotate("", xy=(2.5, 3.2), xytext=(0.3, 3.2),
                    arrowprops=dict(arrowstyle="-|>", color=HOT, lw=2.5), zorder=5)
        ax.plot([0.3, 7.7], [3.2, 3.2], color=HOT, lw=2.5, zorder=4)
        ax.annotate("", xy=(9.7, 3.2), xytext=(7.5, 3.2),
                    arrowprops=dict(arrowstyle="-|>", color=HOT, lw=2.5), zorder=5)
        ax.annotate("", xy=(2.5, 2.0), xytext=(0.3, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color=COLD, lw=2.5), zorder=5)
        ax.plot([0.3, 7.7], [2.0, 2.0], color=COLD, lw=2.5, zorder=4)
        ax.annotate("", xy=(9.7, 2.0), xytext=(7.5, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color=COLD, lw=2.5), zorder=5)
        ax.text(0.1, 3.55, f"The\n{Th_in:.1f}C", ha="center", va="center",
                color=HOT, fontsize=10, fontweight="bold")
        ax.text(9.9, 3.55, f"Tfe\n{res['Th_out']:.2f}C", ha="center", va="center",
                color=HOT, fontsize=10, fontweight="bold")
        ax.text(0.1, 1.55, f"Tce\n{Tc_in:.1f}C", ha="center", va="center",
                color=COLD, fontsize=10, fontweight="bold")
        ax.text(9.9, 1.55, f"Tcs\n{res['Tc_out']:.2f}C", ha="center", va="center",
                color=COLD, fontsize=10, fontweight="bold")

    else:  # Croise
        ax.annotate("", xy=(2.5, 2.6), xytext=(0.3, 2.6),
                    arrowprops=dict(arrowstyle="-|>", color=HOT, lw=2.5), zorder=5)
        ax.plot([0.3, 7.7], [2.6, 2.6], color=HOT, lw=2.5, zorder=4)
        ax.annotate("", xy=(9.7, 2.6), xytext=(7.5, 2.6),
                    arrowprops=dict(arrowstyle="-|>", color=HOT, lw=2.5), zorder=5)
        ax.annotate("", xy=(5.0, 1.5), xytext=(5.0, 0.3),
                    arrowprops=dict(arrowstyle="-|>", color=COLD, lw=2.5), zorder=5)
        ax.plot([5.0, 5.0], [0.3, 3.7], color=COLD, lw=2.5, zorder=4)
        ax.annotate("", xy=(5.0, 4.7), xytext=(5.0, 3.5),
                    arrowprops=dict(arrowstyle="-|>", color=COLD, lw=2.5), zorder=5)
        ax.text(0.1, 3.0, f"The\n{Th_in:.1f}C", ha="center", va="center",
                color=HOT, fontsize=10, fontweight="bold")
        ax.text(9.9, 3.0, f"Tfe\n{res['Th_out']:.2f}C", ha="center", va="center",
                color=HOT, fontsize=10, fontweight="bold")
        ax.text(5.0, 0.1, f"Tce  {Tc_in:.1f}C", ha="center", va="center",
                color=COLD, fontsize=10, fontweight="bold")
        ax.text(5.0, 4.9, f"Tcs  {res['Tc_out']:.2f}C", ha="center", va="center",
                color=COLD, fontsize=10, fontweight="bold")

    ax.text(5.0, 4.6, flow_type, ha="center", va="center",
            fontsize=13, fontweight="bold", color=BORDER)
    info = (
        f"Q = {res['Q']:,.1f} W   |   e = {res['epsilon']:.4f}"
        f"   |   NTU = {res['NTU']:.4f}   |   UA = {UA:,.1f} W/K"
    )
    ax.text(5.0, 0.3, info, ha="center", va="center",
            fontsize=9, color=GREEN, fontweight="bold")

    plt.tight_layout()
    return fig


def draw_temperature_profile(res, Th_in, Tc_in):
    flow_type = res["flow_type"]
    Th_out = res["Th_out"]
    Tc_out = res["Tc_out"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(0, 1, 200)

    Th_x = Th_out + (Th_in - Th_out) * (1 - x)
    if flow_type == "Co-courant":
        Tc_x = Tc_in + (Tc_out - Tc_in) * x
        cold_label = "Fluide froid (->)"
    else:
        Tc_x = Tc_in + (Tc_out - Tc_in) * (1 - x)
        cold_label = "Fluide froid (<-)"

    ax.plot(x, Th_x, color="#e74c3c", lw=2.5, label="Fluide chaud")
    ax.plot(x, Tc_x, color="#2980b9", lw=2.5, label=cold_label,
            linestyle="--" if flow_type != "Co-courant" else "-")
    ax.fill_between(x, Tc_x, Th_x, alpha=0.08, color="#e74c3c")
    ax.set_xlabel("Position relative dans l'echangeur (0 -> 1)", fontsize=11)
    ax.set_ylabel("Temperature (C)", fontsize=11)
    ax.set_title(f"Profil de temperatures - {flow_type}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(-0.02, 1.02)
    plt.tight_layout()
    return fig


def draw_eps_vs_NTU(res):
    Cr = res["Cr"]
    NTU_range = np.linspace(0, 8, 400)

    fig, ax = plt.subplots(figsize=(8, 4))
    styles_map = {
        "Contre-courant":         (epsilon_counter,             "#8e44ad", "-"),
        "Co-courant":             (epsilon_parallel,            "#e67e22", "--"),
        "Croise non melange":     (epsilon_crossflow_unmixed,   "#27ae60", "-."),
        "Calandre (1P/2P)":       (epsilon_shell_tube,         "#2980b9", ":"),
    }
    for label, (fn, col, ls) in styles_map.items():
        eps_vals = [fn(n, Cr) for n in NTU_range]
        ax.plot(NTU_range, eps_vals, color=col, lw=2, linestyle=ls, label=label)

    ax.plot(res["NTU"], res["epsilon"], "ko", markersize=9, zorder=5,
            label=f"Point de fonctionnement (e={res['epsilon']:.3f})")
    ax.set_xlabel("NTU", fontsize=11)
    ax.set_ylabel("e (efficacite)", fontsize=11)
    ax.set_title(f"e-NTU pour R = Cr = {Cr:.3f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 3D PLOTLY VISUALIZATIONS
# ─────────────────────────────────────────────

def _tube_line(x_pts, y_pts, z_pts, color, width=6, name=""):
    return go.Scatter3d(
        x=x_pts, y=y_pts, z=z_pts,
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        hoverinfo="name",
    )


def _arrow_cone(x, y, z, u, v, w, color, name=""):
    return go.Cone(
        x=[x], y=[y], z=[z],
        u=[u], v=[v], w=[w],
        colorscale=[[0, color], [1, color]],
        showscale=False,
        sizemode="absolute",
        sizeref=0.18,
        anchor="tail",
        name=name,
        hoverinfo="name",
        showlegend=False,
    )


def _temp_label(x, y, z, text, color):
    return go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode="text",
        text=[text],
        textfont=dict(color=color, size=13, family="Arial Black"),
        hoverinfo="skip",
        showlegend=False,
    )


def _gradient_segments(xs, ys, zs, T_vals, T_min_all, T_max_all):
    traces = []
    for seg in range(len(xs) - 1):
        nval = np.clip((T_vals[seg] - T_min_all) / max(T_max_all - T_min_all, 1), 0, 1)
        r = int(min(255, nval * 2 * 255))
        b = int(min(255, (1 - nval) * 2 * 255))
        g = int(255 - abs(nval - 0.5) * 2 * 255)
        traces.append(go.Scatter3d(
            x=[xs[seg], xs[seg + 1]],
            y=[ys[seg], ys[seg + 1]],
            z=[zs[seg], zs[seg + 1]],
            mode="lines",
            line=dict(color=f"rgb({r},{g},{b})", width=5),
            showlegend=False,
            hoverinfo="skip",
        ))
    return traces


def build_3d_counterflow(res, Th_in, Tc_in, params):
    L = params.get("length", 3.0)
    Ro = params.get("shell_radius", 0.5)
    n_tubes = params.get("n_tubes", 6)
    Th_out = res["Th_out"]
    Tc_out = res["Tc_out"]
    flow_type = res["flow_type"]
    T_min_all = min(Tc_in, Tc_out)
    T_max_all = max(Th_in, Th_out)

    traces = []

    # Shell surface
    theta = np.linspace(0, 2 * np.pi, 60)
    x_ring = np.linspace(0, L, 2)
    T_grid, X_grid = np.meshgrid(theta, x_ring)
    Ys = Ro * np.cos(T_grid)
    Zs = Ro * np.sin(T_grid)
    traces.append(go.Surface(
        x=X_grid, y=Ys, z=Zs,
        colorscale=[[0, "rgba(100,160,220,0.12)"], [1, "rgba(100,160,220,0.12)"]],
        showscale=False, opacity=0.18,
        name="Calandre (shell)", hoverinfo="name", showlegend=True,
    ))

    # End plates
    for x_pos in [0, L]:
        th2 = np.linspace(0, 2 * np.pi, 60)
        r2 = np.linspace(0, Ro, 5)
        TH, RR = np.meshgrid(th2, r2)
        traces.append(go.Surface(
            x=np.full_like(RR, x_pos), y=RR * np.cos(TH), z=RR * np.sin(TH),
            colorscale=[[0, "rgba(80,80,120,0.35)"], [1, "rgba(80,80,120,0.35)"]],
            showscale=False, opacity=0.35, showlegend=False, hoverinfo="skip",
        ))

    # Internal tubes with thermal gradient
    cols = max(1, n_tubes // 2)
    y_positions = np.linspace(-0.25, 0.25, cols)
    z_positions = [-0.12, 0.12] if n_tubes > 1 else [0.0]
    xs_tube = np.linspace(0, L, 30)
    t_frac = xs_tube / L
    for zp in z_positions:
        for yp in y_positions:
            Th_x = Th_in + (Th_out - Th_in) * t_frac
            if flow_type == "Contre-courant":
                Tc_x = Tc_out + (Tc_in - Tc_out) * t_frac
            else:
                Tc_x = Tc_in + (Tc_out - Tc_in) * t_frac
            T_wall = (Th_x + Tc_x) / 2
            traces += _gradient_segments(
                xs_tube, np.full(30, yp), np.full(30, zp),
                T_wall, T_min_all, T_max_all,
            )

    # Flow arrows hot
    traces.append(_tube_line([-0.6, 0], [0, 0], [0, 0], "#e74c3c", 8, "Fluide chaud (entree)"))
    traces.append(_arrow_cone(0, 0, 0, 0.5, 0, 0, "#e74c3c", "-> chaud"))
    traces.append(_tube_line([L, L + 0.6], [0, 0], [0, 0], "#c0392b", 8, "Fluide chaud (sortie)"))
    traces.append(_arrow_cone(L, 0, 0, 0.5, 0, 0, "#c0392b"))

    # Flow arrows cold
    if flow_type == "Contre-courant":
        traces.append(_tube_line([L + 0.6, L], [0, 0], [0.55, 0.55], "#2980b9", 8, "Fluide froid (entree)"))
        traces.append(_arrow_cone(L, 0, 0.55, -0.5, 0, 0, "#2980b9", "<- froid"))
        traces.append(_tube_line([0, -0.6], [0, 0], [0.55, 0.55], "#1a6fa0", 8, "Fluide froid (sortie)"))
        traces.append(_arrow_cone(0.5, 0, 0.55, -0.5, 0, 0, "#1a6fa0"))
    else:
        traces.append(_tube_line([-0.6, 0], [0, 0], [0.55, 0.55], "#2980b9", 8, "Fluide froid (entree)"))
        traces.append(_arrow_cone(0, 0, 0.55, 0.5, 0, 0, "#2980b9", "-> froid"))
        traces.append(_tube_line([L, L + 0.6], [0, 0], [0.55, 0.55], "#1a6fa0", 8, "Fluide froid (sortie)"))
        traces.append(_arrow_cone(L, 0, 0.55, 0.5, 0, 0, "#1a6fa0"))

    # Labels
    traces += [
        _temp_label(-0.7, 0, 0,    f"The={Th_in:.1f}C",  "#e74c3c"),
        _temp_label(L + 0.1, 0, 0, f"Tfe={Th_out:.2f}C", "#c0392b"),
    ]
    if flow_type == "Contre-courant":
        traces += [
            _temp_label(L + 0.1, 0, 0.65, f"Tce={Tc_in:.1f}C",   "#2980b9"),
            _temp_label(-0.7,    0, 0.65, f"Tcs={Tc_out:.2f}C", "#1a6fa0"),
        ]
    else:
        traces += [
            _temp_label(-0.7,    0, 0.65, f"Tce={Tc_in:.1f}C",   "#2980b9"),
            _temp_label(L + 0.1, 0, 0.65, f"Tcs={Tc_out:.2f}C", "#1a6fa0"),
        ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"Visualisation 3D - {flow_type}<br>"
                 f"<sup>Q={res['Q']:,.0f} W | e={res['epsilon']:.4f} | NTU={res['NTU']:.4f}</sup>",
            x=0.5, font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(title="Longueur (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Z (m)"),
            bgcolor="rgba(240,244,250,1)",
            camera=dict(eye=dict(x=1.6, y=1.1, z=0.9)),
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#aaa", borderwidth=1, font=dict(size=11)),
        margin=dict(l=0, r=0, t=80, b=0),
        height=620,
        paper_bgcolor="rgba(245,248,252,1)",
    )
    return fig


def build_3d_crossflow(res, Th_in, Tc_in, params):
    L = params.get("length", 2.5)
    W = params.get("width", 2.5)
    H = params.get("height", 1.2)
    n_ch = params.get("n_channels", 7)
    Th_out = res["Th_out"]
    Tc_out = res["Tc_out"]
    T_min_all = min(Tc_in, Tc_out)
    T_max_all = max(Th_in, Th_out)

    traces = []
    channel_h = H / (2 * n_ch + 1)

    for i in range(n_ch):
        z_hot  = (2 * i + 1) * channel_h
        z_cold = (2 * i + 2) * channel_h
        plate_z = z_cold - channel_h / 2

        Xp = [[0, L], [0, L]]
        Yp = [[0, 0], [W, W]]
        Zp = [[plate_z, plate_z], [plate_z, plate_z]]
        traces.append(go.Surface(
            x=Xp, y=Yp, z=Zp,
            colorscale=[[0, "rgba(80,90,110,0.55)"], [1, "rgba(80,90,110,0.55)"]],
            showscale=False, opacity=0.55, showlegend=False, hoverinfo="skip",
        ))

        t_frac = np.linspace(0, 1, 20)
        Th_x = Th_in + (Th_out - Th_in) * t_frac
        Tc_x = Tc_in + (Tc_out - Tc_in) * t_frac
        xs = t_frac * L
        ys_cold = t_frac * W

        traces += _gradient_segments(xs, np.full(20, W / 2), np.full(20, z_hot),
                                     Th_x, T_min_all, T_max_all)
        traces += _gradient_segments(np.full(20, L / 2), ys_cold, np.full(20, z_cold),
                                     Tc_x, T_min_all, T_max_all)

    # Outer box
    box_x = [0, L, L, 0, 0, 0, L, L, 0, 0, L, L, 0, 0, L, L]
    box_y = [0, 0, W, W, 0, 0, 0, W, W, 0, 0, 0, 0, 0, W, W]
    box_z = [0, 0, 0, 0, 0, H, H, H, H, H, 0, H, 0, H, 0, H]
    traces.append(go.Scatter3d(
        x=box_x, y=box_y, z=box_z,
        mode="lines", line=dict(color="#2c3e50", width=3),
        name="Boitier echangeur", hoverinfo="name",
    ))

    traces.append(_arrow_cone(-0.3, W / 2, H * 0.25, 0.5, 0, 0, "#e74c3c", "-> Chaud (x)"))
    traces.append(_arrow_cone(L / 2, -0.3, H * 0.75, 0, 0.5, 0, "#2980b9", "-> Froid (y)"))
    traces += [
        _temp_label(-0.5, W / 2, H * 0.25, f"The={Th_in:.1f}C",    "#e74c3c"),
        _temp_label(L + 0.1, W / 2, H * 0.25, f"Tfe={Th_out:.2f}C","#c0392b"),
        _temp_label(L / 2, -0.5, H * 0.75, f"Tce={Tc_in:.1f}C",    "#2980b9"),
        _temp_label(L / 2, W + 0.1, H * 0.75, f"Tcs={Tc_out:.2f}C","#1a6fa0"),
    ]
    traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                               line=dict(color="#e74c3c", width=6), name="Fluide chaud"))
    traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                               line=dict(color="#2980b9", width=6), name="Fluide froid"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"Visualisation 3D - {res['flow_type']}<br>"
                 f"<sup>Q={res['Q']:,.0f} W | e={res['epsilon']:.4f} | NTU={res['NTU']:.4f}</sup>",
            x=0.5, font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(title="X - Fluide chaud (m)"),
            yaxis=dict(title="Y - Fluide froid (m)"),
            zaxis=dict(title="Z - Hauteur (m)"),
            bgcolor="rgba(240,244,250,1)",
            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.2)),
            aspectmode="cube",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#aaa", borderwidth=1, font=dict(size=11)),
        margin=dict(l=0, r=0, t=80, b=0),
        height=620,
        paper_bgcolor="rgba(245,248,252,1)",
    )
    return fig


def build_3d_shell_tube(res, Th_in, Tc_in, params):
    L = params.get("length", 3.0)
    Ro = params.get("shell_radius", 0.55)
    n_t = params.get("n_tubes", 8)
    Th_out = res["Th_out"]
    Tc_out = res["Tc_out"]
    T_min_all = min(Tc_in, Tc_out)
    T_max_all = max(Th_in, Th_out)

    traces = []

    # Shell
    theta = np.linspace(0, 2 * np.pi, 60)
    x_ring = np.linspace(0, L, 2)
    T_grid, X_grid = np.meshgrid(theta, x_ring)
    traces.append(go.Surface(
        x=X_grid, y=Ro * np.cos(T_grid), z=Ro * np.sin(T_grid),
        colorscale=[[0, "rgba(180,200,230,0.13)"], [1, "rgba(180,200,230,0.13)"]],
        showscale=False, opacity=0.18,
        name="Calandre", hoverinfo="name", showlegend=True,
    ))

    # Baffle
    th_baffle = np.linspace(0, np.pi, 40)
    traces.append(go.Scatter3d(
        x=np.full(40, L / 2), y=Ro * np.cos(th_baffle), z=Ro * np.sin(th_baffle),
        mode="lines", line=dict(color="rgba(80,80,80,0.6)", width=4),
        name="Chicane", hoverinfo="name",
    ))

    # 2-pass tubes
    cols = max(1, n_t // 2)
    y_positions = np.linspace(-Ro * 0.55, Ro * 0.55, cols)
    xs_p1 = np.linspace(0, L, 30)
    xs_p2 = np.linspace(L, 0, 30)
    t1 = np.linspace(0, 1, 30)
    t2 = np.linspace(1, 0, 30)

    for yp in y_positions:
        Th_p1 = Th_in  + (((Th_in + Th_out) / 2) - Th_in)  * t1
        Th_p2 = ((Th_in + Th_out) / 2) + (Th_out - ((Th_in + Th_out) / 2)) * t2
        traces += _gradient_segments(xs_p1, np.full(30, yp), np.full(30, -0.12),
                                     Th_p1, T_min_all, T_max_all)
        traces += _gradient_segments(xs_p2, np.full(30, yp), np.full(30, 0.12),
                                     Th_p2, T_min_all, T_max_all)

    # Shell-side cold flow (S-path)
    xs_cold = [0,   L / 2, L / 2, L,   L]
    ys_cold = [0,   0,     0,     0,   0]
    zs_cold = [Ro * 0.65, Ro * 0.65, -Ro * 0.65, -Ro * 0.65, -Ro * 0.65]
    traces.append(_tube_line(xs_cold, ys_cold, zs_cold, "#2980b9", 7, "Fluide froid (calandre)"))
    traces.append(_arrow_cone(0, 0, Ro * 0.65, 0.4, 0, 0, "#2980b9"))
    traces.append(_arrow_cone(L / 2, 0, -Ro * 0.4, 0, 0, -0.4, "#2980b9"))

    traces += [
        _temp_label(-0.5, 0, 0,   f"The={Th_in:.1f}C",  "#e74c3c"),
        _temp_label(L + 0.2, 0, 0, f"Tfe={Th_out:.2f}C","#c0392b"),
        _temp_label(0,  0, Ro * 0.85, f"Tce={Tc_in:.1f}C",  "#2980b9"),
        _temp_label(L,  0, -Ro * 0.85, f"Tcs={Tc_out:.2f}C","#1a6fa0"),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text="Visualisation 3D - Calandre 1P / Tubes 2P<br>"
                 f"<sup>Q={res['Q']:,.0f} W | e={res['epsilon']:.4f} | NTU={res['NTU']:.4f}</sup>",
            x=0.5, font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(title="Longueur (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Z (m)"),
            bgcolor="rgba(240,244,250,1)",
            camera=dict(eye=dict(x=1.8, y=1.2, z=1.0)),
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#aaa", borderwidth=1, font=dict(size=11)),
        margin=dict(l=0, r=0, t=80, b=0),
        height=650,
        paper_bgcolor="rgba(245,248,252,1)",
    )
    return fig


def build_3d_visualization(res, Th_in, Tc_in, params):
    ft = res["flow_type"]
    if ft in ("Contre-courant", "Co-courant"):
        return build_3d_counterflow(res, Th_in, Tc_in, params)
    elif ft in ("Croise non melange", "Croise Cmin melange"):
        return build_3d_crossflow(res, Th_in, Tc_in, params)
    elif ft == "Calandre (1 passe) / Tubes (2 passes)":
        return build_3d_shell_tube(res, Th_in, Tc_in, params)
    return build_3d_counterflow(res, Th_in, Tc_in, params)


# ─────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────

def fig_to_image_buffer(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


def generate_pdf(res, UA, Th_in, Tc_in, mc, mh, cpc, cph,
                 fig_schema, fig_profile, fig_eps):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Rapport Echangeur e-NTU",
    )

    styles = getSampleStyleSheet()
    s_title   = ParagraphStyle("T",  parent=styles["Title"],
                               fontSize=18, spaceAfter=6,
                               textColor=colors.HexColor("#2c3e50"))
    s_h1      = ParagraphStyle("H1", parent=styles["Heading1"],
                               fontSize=13, spaceBefore=14, spaceAfter=4,
                               textColor=colors.HexColor("#2980b9"))
    s_body    = ParagraphStyle("B",  parent=styles["Normal"],
                               fontSize=10, leading=14)
    s_caption = ParagraphStyle("C",  parent=styles["Normal"],
                               fontSize=9, leading=11,
                               textColor=colors.grey, alignment=TA_CENTER)

    story = []
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    story.append(Paragraph("Rapport Echangeur de Chaleur", s_title))
    story.append(Paragraph(f"<font color='grey'>Methode e-NTU | {now}</font>", s_body))
    story.append(HRFlowable(width="100%", thickness=2,
                            color=colors.HexColor("#2980b9"), spaceAfter=10))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("1. Donnees d'entree", s_h1))
    input_data = [
        ["Parametre",                      "Fluide chaud",  "Fluide froid"],
        ["Temperature entree (C)",         f"{Th_in:.2f}",  f"{Tc_in:.2f}"],
        ["Debit massique (kg/s)",          f"{mh:.4f}",     f"{mc:.4f}"],
        ["Chaleur specifique cp (J/kg.K)", f"{cph:.2f}",    f"{cpc:.2f}"],
        ["Capacite C = m.cp (W/K)",        f"{res['Ch']:.2f}", f"{res['Cc']:.2f}"],
    ]
    t_input = Table(input_data, colWidths=[7 * cm, 5 * cm, 5 * cm])
    t_input.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_input)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        f"<b>Type :</b> {res['flow_type']} &nbsp;&nbsp; <b>UA :</b> {UA:,.2f} W/K",
        s_body,
    ))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("2. Parametres adimensionnels", s_h1))
    param_data = [
        ["Grandeur",   "Formule",           "Valeur"],
        ["C_min (W/K)","min(Ch, Cc)",       f"{res['Cmin']:.2f}"],
        ["C_max (W/K)","max(Ch, Cc)",       f"{res['Cmax']:.2f}"],
        ["R = Cr",     "C_min/C_max",       f"{res['Cr']:.4f}"],
        ["NTU",        "UA/C_min",          f"{res['NTU']:.4f}"],
        ["epsilon",    "f(NTU,Cr,config)",  f"{res['epsilon']:.4f}"],
        ["Q_max (W)",  "C_min*(The-Tce)",   f"{res['Qmax']:,.2f}"],
    ]
    t_param = Table(param_data, colWidths=[5 * cm, 6 * cm, 6 * cm])
    t_param.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#8e44ad")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN",         (2, 0), (2, -1),  "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_param)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("3. Resultats thermiques", s_h1))
    res_data = [
        ["Grandeur",                        "Valeur",             "Unite"],
        ["Flux thermique Q (Phi)",          f"{res['Q']:,.2f}",   "W"],
        ["Temperature sortie chaud Tfe",    f"{res['Th_out']:.2f}", "C"],
        ["Temperature sortie froid Tcs",    f"{res['Tc_out']:.2f}", "C"],
        ["Efficacite epsilon",              f"{res['epsilon']:.4f}", "-"],
        ["NTU",                             f"{res['NTU']:.4f}",  "-"],
        ["UA",                              f"{UA:,.2f}",         "W/K"],
    ]
    t_res = Table(res_data, colWidths=[8 * cm, 5 * cm, 4 * cm])
    t_res.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#27ae60")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_res)

    ok_hot  = res["Th_out"] > Tc_in
    ok_cold = res["Tc_out"] < Th_in
    if ok_hot and ok_cold:
        story.append(Paragraph(
            "<font color='green'>Configuration physiquement coherente.</font>", s_body))
    else:
        story.append(Paragraph(
            "<font color='red'>Attention : croisement de temperatures detecte.</font>", s_body))

    story.append(PageBreak())

    story.append(Paragraph("4. Schema de l'echangeur", s_h1))
    story.append(Spacer(1, 0.2 * cm))
    img1_buf = fig_to_image_buffer(fig_schema, dpi=150)
    story.append(RLImage(img1_buf, width=17 * cm, height=8 * cm))
    story.append(Paragraph(
        f"Figure 1 - Schema de l'echangeur {res['flow_type']}.", s_caption))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("5. Profil de temperatures", s_h1))
    img2_buf = fig_to_image_buffer(fig_profile, dpi=150)
    story.append(RLImage(img2_buf, width=16 * cm, height=8 * cm))
    story.append(Paragraph(
        "Figure 2 - Evolution des temperatures des deux fluides.", s_caption))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("6. Courbes e-NTU", s_h1))
    img3_buf = fig_to_image_buffer(fig_eps, dpi=150)
    story.append(RLImage(img3_buf, width=16 * cm, height=8 * cm))
    story.append(Paragraph(
        f"Figure 3 - Efficacite e en fonction de NTU pour R = {res['Cr']:.3f}.", s_caption))

    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Paragraph(
        f"Rapport genere par l'application Echangeurs e-NTU — {now}", s_caption))

    doc.build(story)
    buf.seek(0)
    return buf.read()  # return bytes, not BytesIO


# ─────────────────────────────────────────────
# THEORY PAGE
# ─────────────────────────────────────────────

def show_theory_page():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
    .stApp { background: #0b0f1a !important; }
    section[data-testid="stSidebar"] { background: #080c14 !important; }
    .step-card {
        background: #0f1623; border: 1px solid #1e2d45;
        border-left: 3px solid #f6ad55; border-radius: 6px;
        padding: 24px 28px; margin-bottom: 28px;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .step-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 10px;
        font-weight: 600; letter-spacing: 3px; color: #f6ad55;
        text-transform: uppercase; margin-bottom: 6px;
    }
    .step-title {
        font-size: 19px; font-weight: 500; color: #e2e8f0;
        margin-bottom: 10px; letter-spacing: -0.2px;
    }
    .step-body {
        font-size: 14px; color: #8899aa; line-height: 1.8; font-weight: 300;
    }
    .step-body b { color: #cbd5e0; font-weight: 500; }
    .chip-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }
    .chip {
        font-family: 'IBM Plex Mono', monospace; font-size: 11px;
        padding: 4px 12px; border-radius: 3px; font-weight: 600;
        letter-spacing: 0.3px; border: 1px solid;
    }
    .chip-ok   { color:#68d391; border-color:rgba(104,211,145,0.3); background:rgba(104,211,145,0.07); }
    .chip-warn { color:#f6ad55; border-color:rgba(246,173,85,0.3);  background:rgba(246,173,85,0.07); }
    .chip-err  { color:#fc8181; border-color:rgba(252,129,129,0.3); background:rgba(252,129,129,0.07); }
    .sec-div { height:1px; background:linear-gradient(to right,#1e2d45,transparent); margin:36px 0; }
    .cfg-table { width:100%; border-collapse:collapse; font-size:13px; margin-top:14px; }
    .cfg-table th {
        font-family:'IBM Plex Mono',monospace; font-size:9px; letter-spacing:2px;
        text-transform:uppercase; color:#4a5568; padding:8px 14px;
        border-bottom:1px solid #1e2d45; text-align:left;
    }
    .cfg-table td { padding:10px 14px; border-bottom:1px solid #111827; color:#718096; }
    .cfg-table td:first-child { color:#a0aec0; font-weight:500; }
    .cfg-table td:last-child  { color:#68d391; font-family:'IBM Plex Mono',monospace; font-size:12px; }
    .sum-table { width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace;
                 font-size:12px; margin-top:12px; }
    .sum-table th { background:#0d1117; color:#4a5568; font-size:9px; letter-spacing:2px;
                    padding:9px 14px; text-align:left; border-bottom:1px solid #1e2d45; }
    .sum-table td { padding:9px 14px; border-bottom:1px solid #111827; color:#a0aec0; }
    .sum-table td:first-child  { color:#63b3ed; }
    .sum-table td:nth-child(2) { color:#68d391; font-size:13px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="border-bottom:1px solid #1e2d45; padding-bottom:20px; margin-bottom:36px;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:3px;
                    color:#4a5568; text-transform:uppercase; margin-bottom:8px;">
            THERMIQUE · GENIE THERMIQUE · TRANSFERT DE CHALEUR
        </div>
        <h2 style="font-family:'IBM Plex Sans',sans-serif; font-size:28px; font-weight:300;
                   color:#f7fafc; margin:0; letter-spacing:-0.5px;">
            Methode <span style="color:#f6ad55; font-weight:600;">e &ndash; NTU</span> &mdash;
            Theorie et Methodologie
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1 ──
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Etape 01</div>
        <div class="step-title">Bilan enthalpique — Capacites calorifiques</div>
        <div class="step-body">
            Chaque fluide transporte une puissance thermique proportionnelle a son
            <b>debit massique</b> et sa <b>chaleur specifique</b>.
            On definit la capacite calorifique C (W/K) de chaque fluide.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"C_h = \dot{m}_h \cdot c_{p,h} \quad \text{[W/K]}, \qquad C_c = \dot{m}_c \cdot c_{p,c} \quad \text{[W/K]}")
    st.latex(r"\Phi = C_h\,(T_{h,e} - T_{h,s}) = C_c\,(T_{c,s} - T_{c,e})")

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # ── Step 2 ──
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Etape 02</div>
        <div class="step-title">Identification de C_min, C_max et rapport R</div>
        <div class="step-body">
            Le fluide dont la capacite est la plus faible subit la plus grande variation de
            temperature — c'est le <b>fluide limitant</b>. Le rapport
            <b>R = C_min / C_max</b> est toujours compris entre 0 et 1.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"C_{\min} = \min(C_h, C_c), \qquad C_{\max} = \max(C_h, C_c)")
    st.latex(r"R = \frac{C_{\min}}{C_{\max}}, \qquad 0 < R \leq 1")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="step-card" style="border-left-color:#68d391;">
            <div class="step-label" style="color:#68d391;">Cas R vers 0</div>
            <div class="step-body">Un fluide a capacite tres grande (condensation/evaporation).
            Sa temperature reste quasi constante. e tend vers 1 - exp(-NTU).</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="step-card" style="border-left-color:#fc8181;">
            <div class="step-label" style="color:#fc8181;">Cas R = 1</div>
            <div class="step-body">Les deux fluides ont la meme capacite.
            En contre-courant : e = NTU/(1+NTU).
            En co-courant, la limite est e = 0.5.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # ── Step 3 ──
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Etape 03</div>
        <div class="step-title">Flux maximal theorique et Efficacite e</div>
        <div class="step-body">
            Le flux <b>Phi_max</b> serait echange dans un echangeur de longueur infinie.
            L'<b>efficacite e</b> est le rapport entre le flux reel et ce maximum.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"\Phi_{\max} = C_{\min}\,\bigl(T_{h,e} - T_{c,e}\bigr)")
    st.latex(r"\varepsilon = \frac{\Phi}{\Phi_{\max}} \in [0,\,1]")
    st.latex(r"T_{h,s} = T_{h,e} - \frac{\varepsilon\,\Phi_{\max}}{C_h}, \qquad T_{c,s} = T_{c,e} + \frac{\varepsilon\,\Phi_{\max}}{C_c}")

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # ── Step 4 ──
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Etape 04</div>
        <div class="step-title">Nombre de Transfert d'Unites — NTU</div>
        <div class="step-body">
            Le NTU quantifie la <b>capacite d'echange thermique</b> de l'echangeur
            par rapport au fluide limitant. Il depend du coefficient global <b>U</b>
            et de la surface <b>A</b>, regroupes dans le produit <b>UA</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"\mathrm{NTU} = \frac{UA}{C_{\min}}")
    st.latex(r"\frac{1}{UA} = \frac{1}{h_h A} + R_{\text{paroi}} + \frac{1}{h_c A}")

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("""
        <div class="step-card" style="border-left-color:#b794f4;">
            <div class="step-label" style="color:#b794f4;">NTU faible (&lt; 0.5)</div>
            <div class="step-body">Echangeur sous-dimensionne. Faible transfert thermique.</div>
        </div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown("""
        <div class="step-card" style="border-left-color:#68d391;">
            <div class="step-label" style="color:#68d391;">NTU eleve (&gt; 3)</div>
            <div class="step-body">Echangeur bien dimensionne. Au-dela de NTU=5, le gain marginal est faible.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # ── Step 5 ──
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Etape 05</div>
        <div class="step-title">Formules analytiques e = f(NTU, R) par configuration</div>
        <div class="step-body">
            Chaque configuration possede sa propre relation entre e, NTU et R,
            derivee de la resolution des equations differentielles du transfert.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Contre-courant** *(configuration la plus efficace)*")
    st.latex(r"""
    \varepsilon = \begin{cases}
        \dfrac{1 - e^{-\mathrm{NTU}(1-R)}}{1 - R\,e^{-\mathrm{NTU}(1-R)}} & R \neq 1 \\[10pt]
        \dfrac{\mathrm{NTU}}{1 + \mathrm{NTU}} & R = 1
    \end{cases}
    """)

    st.markdown("**Co-courant** *(parallele)*")
    st.latex(r"\varepsilon = \frac{1 - e^{-\mathrm{NTU}(1+R)}}{1 + R}")

    st.markdown("**Croise — fluides non melanges**")
    st.latex(r"\varepsilon \approx 1 - \exp\!\left(\frac{e^{-R\,\mathrm{NTU}^{0.22}}-1}{R\,\mathrm{NTU}^{-0.78}}\right)")

    st.markdown("**Calandre 1 passe / Tubes 2 passes**")
    st.latex(r"""
    \varepsilon = \frac{2}{1 + R + \sqrt{1+R^2}\;\dfrac{1+e^{-\mathrm{NTU}\sqrt{1+R^2}}}{1-e^{-\mathrm{NTU}\sqrt{1+R^2}}}}
    """)

    st.markdown("""
    <table class="cfg-table">
        <tr><th>Configuration</th><th>e max (NTU vers infini)</th><th>Avantage</th></tr>
        <tr><td>Contre-courant</td><td>e vers 1</td><td>Meilleur rendement possible</td></tr>
        <tr><td>Co-courant</td><td>e vers 1/(1+R)</td><td>Simple, controle T sortie</td></tr>
        <tr><td>Croise non melange</td><td>Entre co et contre</td><td>Compacite</td></tr>
        <tr><td>Calandre 1P/2P</td><td>Proche contre-courant</td><td>Standard industriel</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # ── Step 6 ──
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Etape 06</div>
        <div class="step-title">Dimensionnement inverse — Trouver UA pour une cible T</div>
        <div class="step-body">
            En mode dimensionnement, on fixe une <b>temperature de sortie cible</b> et on
            cherche le <b>UA necessaire</b> par resolution numerique par <b>dichotomie</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"\text{Trouver } UA^* \text{ tel que } f(UA^*) = T_{\text{sortie}}(UA^*) - T_{\text{cible}} = 0")
    st.latex(r"UA_{\text{mid}} = \frac{UA_{\text{low}} + UA_{\text{high}}}{2}, \quad \text{convergence : } |f(UA_{\text{mid}})| < 10^{-8}")

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # ── Step 7 — Algo summary ──
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Etape 07</div>
        <div class="step-title">Synthese — Algorithme complet de calcul</div>
    </div>
    <table class="sum-table">
        <tr><th>#</th><th>Calcul</th><th>Formule</th><th>Resultat</th></tr>
        <tr><td>1</td><td>Capacites calorifiques</td><td>C = m_point * cp</td><td>Ch, Cc [W/K]</td></tr>
        <tr><td>2</td><td>Fluide limitant</td><td>Cmin = min(Ch, Cc)</td><td>Cmin, Cmax [W/K]</td></tr>
        <tr><td>3</td><td>Rapport de capacites</td><td>R = Cmin/Cmax</td><td>R dans [0,1]</td></tr>
        <tr><td>4</td><td>Nombre de transfert</td><td>NTU = UA/Cmin</td><td>NTU [-]</td></tr>
        <tr><td>5</td><td>Efficacite</td><td>e = f(NTU, R, config)</td><td>e dans [0,1]</td></tr>
        <tr><td>6</td><td>Flux max theorique</td><td>Phi_max = Cmin*(The-Tce)</td><td>Phi_max [W]</td></tr>
        <tr><td>7</td><td>Flux echange reel</td><td>Phi = e * Phi_max</td><td>Phi [W]</td></tr>
        <tr><td>8</td><td>Temperatures de sortie</td><td>Tfe = The - Phi/Ch ; Tcs = Tce + Phi/Cc</td><td>Tfe, Tcs [C]</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    # ── Step 8 — Validity ──
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Etape 08</div>
        <div class="step-title">Hypotheses et Domaine de validite</div>
        <div class="step-body">La methode e-NTU repose sur des hypotheses simplificatrices.</div>
    </div>
    <div class="chip-row">
        <span class="chip chip-ok">Regime permanent</span>
        <span class="chip chip-ok">Proprietes constantes</span>
        <span class="chip chip-ok">Pas de pertes exterieures</span>
        <span class="chip chip-warn">Pas de changement de phase</span>
        <span class="chip chip-err">Non valide si T se croisent</span>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"T_{h,s} > T_{c,e} \qquad \text{et} \qquad T_{c,s} < T_{h,e}")

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step-card" style="border-left-color:#4a5568;">
        <div class="step-label" style="color:#718096;">References</div>
        <div class="step-body">
            Incropera et al. — <i>Fundamentals of Heat and Mass Transfer</i>, 7e ed., Wiley<br>
            Cengel Y.A. — <i>Transfert thermique</i>, De Boeck<br>
            Kays & London — <i>Compact Heat Exchangers</i>, McGraw-Hill
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

# ── Sidebar ──
with st.sidebar:
    st.header("Configuration")

    flow_type = st.selectbox("Type d'echangeur", FLOW_TYPES)

    st.divider()
    st.subheader("Fluide chaud")
    Th_in = st.number_input("The - Temperature entree (C)", value=80.0, step=1.0)
    mh    = st.number_input("mh  - Debit massique (kg/s)", value=1.0, min_value=0.001, step=0.1)
    cph   = st.number_input("cph - Chaleur specifique (J/kg.K)", value=4180.0, min_value=1.0, step=10.0)

    st.divider()
    st.subheader("Fluide froid")
    Tc_in = st.number_input("Tce - Temperature entree (C)", value=20.0, step=1.0)
    mc    = st.number_input("mc  - Debit massique (kg/s)", value=1.0, min_value=0.001, step=0.1)
    cpc   = st.number_input("cpc - Chaleur specifique (J/kg.K)", value=4180.0, min_value=1.0, step=10.0)

    st.divider()
    st.subheader("Geometrie 3D")
    length_3d = st.slider("Longueur echangeur (m)", 1.0, 6.0, 3.0, 0.1)
    if flow_type in ("Contre-courant", "Co-courant", "Calandre (1 passe) / Tubes (2 passes)"):
        n_tubes = st.slider("Nombre de tubes internes", 2, 16, 6, 2)
        shell_r = st.slider("Rayon calandre (m)", 0.2, 1.0, 0.5, 0.05)
        params_3d = {
            "length": length_3d, "shell_radius": shell_r,
            "tube_radius": 0.05, "n_tubes": n_tubes,
        }
    else:
        n_ch = st.slider("Nombre de canaux", 3, 12, 6, 1)
        params_3d = {
            "length": length_3d, "width": length_3d,
            "height": 1.2, "n_channels": n_ch,
        }
    st.divider()

# ── Main tabs ──
tab_calc, tab_theory = st.tabs(["Calculateur", "Theorie et Methodologie"])

with tab_theory:
    show_theory_page()

with tab_calc:
    st.markdown(
        "Calculez **Q (Phi)**, les **temperatures de sortie**, l'**efficacite** "
        "et le **NTU** pour differentes configurations d'echangeurs."
    )

    col_mode, _ = st.columns([3, 1])
    with col_mode:
        mode = st.radio(
            "Mode de calcul",
            ["Calcul direct (UA connu)", "Dimensionnement (cible de temperature)"],
            horizontal=True,
        )

    UA = None
    res = None

    if mode == "Calcul direct (UA connu)":
        UA = st.number_input("UA (W/K)", value=2000.0, min_value=0.0, step=100.0)
    else:
        c1, c2 = st.columns(2)
        with c1:
            target_choice = st.selectbox("Temperature cible", [
                "Tcs - Sortie froid", "Tfe - Sortie chaud",
            ])
        with c2:
            target_value = st.number_input("Valeur cible (C)", value=50.0, step=1.0)

        target_key = "Tc_out" if target_choice.startswith("Tcs") else "Th_out"

        if st.button("Calculer UA requis", type="primary"):
            UA_sol = solve_UA_for_target(
                Th_in, Tc_in, mh, cph, mc, cpc, flow_type,
                target=target_key, target_value=target_value,
            )
            if UA_sol is None:
                st.error("Aucune solution trouvee. Verifiez que la cible est physiquement atteignable.")
            else:
                UA = float(UA_sol)
                st.session_state["last_UA"] = UA
                st.success(f"UA requis = {UA:,.2f} W/K")

        if UA is None and "last_UA" in st.session_state:
            UA = st.session_state["last_UA"]
        if UA is not None:
            st.session_state["last_UA"] = UA

    # ── Calcul principal ──
    if UA is not None and UA > 0:
        if Th_in <= Tc_in:
            st.error("The doit etre superieure a Tce.")
        else:
            res = compute_outputs(Th_in, Tc_in, mh, cph, mc, cpc, UA, flow_type)
            if res is None:
                st.error("Debits ou capacites thermiques invalides.")
            else:
                # ── Metrics ──
                st.markdown("---")
                st.subheader("Resultats")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Q = Phi (W)",      f"{res['Q']:,.1f}")
                m2.metric("e (efficacite)",   f"{res['epsilon']:.4f}")
                m3.metric("NTU",              f"{res['NTU']:.4f}")
                m4.metric("Tfe (C)",          f"{res['Th_out']:.2f}")
                m5.metric("Tcs (C)",          f"{res['Tc_out']:.2f}")

                if res["Th_out"] < Tc_in or res["Tc_out"] > Th_in:
                    st.warning("Croisement de temperatures detecte.")
                else:
                    st.success("Configuration physiquement coherente.")

                with st.expander("Details complets (Cmin, Cmax, Cr, Qmax)"):
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Ch (W/K)",    f"{res['Ch']:.2f}")
                    d1.metric("Cc (W/K)",    f"{res['Cc']:.2f}")
                    d2.metric("Cmin (W/K)",  f"{res['Cmin']:.2f}")
                    d2.metric("Cmax (W/K)",  f"{res['Cmax']:.2f}")
                    d3.metric("R = Cr",      f"{res['Cr']:.4f}")
                    d3.metric("UA (W/K)",    f"{UA:,.2f}")
                    d4.metric("Qmax (W)",    f"{res['Qmax']:,.1f}")

                # ── Schema 2D ──
                st.markdown("---")
                st.subheader("Schema de l'echangeur")
                fig_schema = draw_schema(res, UA, Th_in, Tc_in)
                st.pyplot(fig_schema)
                plt.close(fig_schema)

                # ── Profil T ──
                st.subheader("Profil de temperatures")
                fig_profile = draw_temperature_profile(res, Th_in, Tc_in)
                st.pyplot(fig_profile)
                plt.close(fig_profile)

                # ── e-NTU ──
                st.subheader("Courbes e-NTU")
                fig_eps = draw_eps_vs_NTU(res)
                st.pyplot(fig_eps)
                plt.close(fig_eps)

                # ── 3D ──
                st.markdown("---")
                st.subheader("Visualisation 3D interactive")
                st.markdown(
                    "Modele 3D de l'echangeur avec gradient thermique. "
                    "Tournez, zoomez et explorez librement. "
                    "Le degrade bleu->rouge represente la montee en temperature."
                )
                with st.spinner("Generation du modele 3D..."):
                    fig_3d = build_3d_visualization(res, Th_in, Tc_in, params_3d)
                st.plotly_chart(fig_3d, use_container_width=True)

                # Simple color legend using plotly scatter (no heatmap colorbar issues)
                T_min_leg = min(Tc_in, res["Tc_out"])
                T_max_leg = max(Th_in, res["Th_out"])
                T_vals = np.linspace(T_min_leg, T_max_leg, 50)
                norm_vals = (T_vals - T_min_leg) / max(T_max_leg - T_min_leg, 1)
                r_vals = np.clip(norm_vals * 2, 0, 1)
                b_vals = np.clip((1 - norm_vals) * 2, 0, 1)
                g_vals = 1 - np.abs(norm_vals - 0.5) * 2

                col_leg1, col_leg2, col_leg3 = st.columns([1, 3, 1])
                with col_leg2:
                    leg_fig = go.Figure()
                    for i in range(len(T_vals) - 1):
                        r = int(r_vals[i] * 255)
                        g = int(g_vals[i] * 255)
                        b = int(b_vals[i] * 255)
                        leg_fig.add_trace(go.Scatter(
                            x=[T_vals[i], T_vals[i + 1]],
                            y=[0, 0],
                            mode="lines",
                            line=dict(color=f"rgb({r},{g},{b})", width=20),
                            showlegend=False,
                            hoverinfo="skip",
                        ))
                    leg_fig.update_layout(
                        height=80,
                        margin=dict(l=40, r=40, t=10, b=30),
                        xaxis=dict(
                            title="Temperature (C)",
                            tickvals=[T_min_leg,
                                      (T_min_leg + T_max_leg) / 2,
                                      T_max_leg],
                            ticktext=[f"{T_min_leg:.1f} C (froid)",
                                      f"{(T_min_leg + T_max_leg) / 2:.1f} C",
                                      f"{T_max_leg:.1f} C (chaud)"],
                            showgrid=False,
                        ),
                        yaxis=dict(visible=False),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(leg_fig, use_container_width=True)

                # ── PDF ──
                st.markdown("---")
                st.subheader("Rapport PDF")
                st.markdown(
                    "Generez un rapport complet avec toutes les donnees, "
                    "schemas et courbes."
                )
                if st.button("Generer le rapport PDF", type="primary"):
                    with st.spinner("Generation du rapport en cours..."):
                        f_schema  = draw_schema(res, UA, Th_in, Tc_in)
                        f_profile = draw_temperature_profile(res, Th_in, Tc_in)
                        f_eps     = draw_eps_vs_NTU(res)

                        pdf_bytes = generate_pdf(
                            res, UA, Th_in, Tc_in, mc, mh, cpc, cph,
                            f_schema, f_profile, f_eps,
                        )
                        plt.close("all")

                    # download_button must be OUTSIDE the spinner block
                    fname = (
                        "rapport_echangeur_"
                        + flow_type.replace(" ", "_").replace("/", "_")
                        + ".pdf"
                    )
                    st.download_button(
                        label="Telecharger le rapport PDF",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf",
                    )

    elif mode == "Calcul direct (UA connu)":
        st.info("Entrez un UA > 0 pour lancer le calcul.")