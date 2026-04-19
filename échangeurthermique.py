"""
Échangeurs de Chaleur – Méthode ε–NTU
App Streamlit complète avec schéma et rapport PDF téléchargeable.
"""

import io
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Échangeurs ε–NTU", layout="wide", page_icon="🌡️")

# ─────────────────────────────────────────────
# PHYSICS
# ─────────────────────────────────────────────

def epsilon_parallel(NTU, Cr):
    return (1.0 - np.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)

def epsilon_counter(NTU, Cr):
    if np.isclose(Cr, 1.0):
        return NTU / (1.0 + NTU)
    a = NTU * (1.0 - Cr)
    return (1.0 - np.exp(-a)) / (1.0 - Cr * np.exp(-a))

def epsilon_crossflow_unmixed(NTU, Cr):
    """Croisé – deux fluides non-mélangés (corrélation approchée)."""
    return 1.0 - np.exp((np.exp(-Cr * NTU**0.22) - 1.0) / (Cr * NTU**(-0.78) + 1e-12))

def epsilon_crossflow_mixed_Cmin(NTU, Cr):
    """Croisé – Cmin mélangé, Cmax non-mélangé."""
    return (1.0 / Cr) * (1.0 - np.exp(-Cr * (1.0 - np.exp(-NTU))))

def epsilon_shell_tube(NTU, Cr):
    """1 passe calandre, 2 passes tubes (formule exacte)."""
    sq = np.sqrt(1.0 + Cr**2)
    num = 1.0 + np.exp(-NTU * sq)
    den = 1.0 - np.exp(-NTU * sq)
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

    ft = flow_type
    if ft == "Co-courant":
        eps = epsilon_parallel(NTU, Cr)
    elif ft == "Contre-courant":
        eps = epsilon_counter(NTU, Cr)
    elif ft == "Croisé – non mélangé":
        eps = epsilon_crossflow_unmixed(NTU, Cr)
    elif ft == "Croisé – Cmin mélangé":
        eps = epsilon_crossflow_mixed_Cmin(NTU, Cr)
    elif ft == "Calandre (1 passe) / Tubes (2 passes)":
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
        "flow_type": flow_type
    }

def solve_UA_for_target(Th_in, Tc_in, mh, cph, mc, cpc, flow_type,
                         target="Tc_out", target_value=40.0):
    UA_low, UA_high = 1e-6, 1e8

    def f(UA):
        r = compute_outputs(Th_in, Tc_in, mh, cph, mc, cpc, UA, flow_type)
        if r is None:
            return np.nan
        return r[target] - target_value

    f_low = f(UA_low)
    f_high = f(UA_high)
    if np.isnan(f_low) or np.isnan(f_high):
        return None
    if f_low * f_high > 0:
        return None

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
# SCHEMA DRAWING
# ─────────────────────────────────────────────

def draw_schema(res, UA, Th_in, Tc_in):
    """Dessine le schéma de l'échangeur avec les températures et flux."""
    flow_type = res["flow_type"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    # ── Couleurs ──
    HOT   = "#e74c3c"
    COLD  = "#2980b9"
    BODY  = "#ecf0f1"
    BORDER= "#2c3e50"
    GREEN = "#27ae60"

    # ── Corps échangeur ──
    exchanger_box = FancyBboxPatch((2.5, 1.5), 5, 2,
                                   boxstyle="round,pad=0.15",
                                   linewidth=2, edgecolor=BORDER,
                                   facecolor=BODY, zorder=2)
    ax.add_patch(exchanger_box)

    # ── Lignes internes (tubes représentatifs) ──
    for y_line in [2.0, 2.5, 3.0, 3.5]:
        ax.plot([2.65, 7.35], [y_line, y_line],
                color="#bdc3c7", linewidth=0.8, linestyle="--", zorder=3)

    # ─────────────────────────────────────────
    # Schéma selon le type d'échangeur
    # ─────────────────────────────────────────
    if flow_type in ("Contre-courant", "Calandre (1 passe) / Tubes (2 passes)"):
        # Fluide chaud : gauche → droite (haut)
        ax.annotate("", xy=(2.5, 3.2), xytext=(0.3, 3.2),
                    arrowprops=dict(arrowstyle="-|>", color=HOT, lw=2.5), zorder=5)
        ax.plot([0.3, 7.7], [3.2, 3.2], color=HOT, lw=2.5, zorder=4)
        ax.annotate("", xy=(9.7, 3.2), xytext=(7.5, 3.2),
                    arrowprops=dict(arrowstyle="-|>", color=HOT, lw=2.5), zorder=5)

        # Fluide froid : droite → gauche (bas)
        ax.annotate("", xy=(7.5, 2.0), xytext=(9.7, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color=COLD, lw=2.5), zorder=5)
        ax.plot([0.3, 9.7], [2.0, 2.0], color=COLD, lw=2.5, zorder=4)
        ax.annotate("", xy=(0.3, 2.0), xytext=(2.5, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color=COLD, lw=2.5), zorder=5)

        # Températures
        ax.text(0.1, 3.5, f"The\n{Th_in:.1f}°C", ha="center", va="center",
                color=HOT, fontsize=10, fontweight="bold")
        ax.text(9.9, 3.5, f"Tfe\n{res['Th_out']:.2f}°C", ha="center", va="center",
                color=HOT, fontsize=10, fontweight="bold")
        ax.text(9.9, 1.6, f"Tce\n{Tc_in:.1f}°C", ha="center", va="center",
                color=COLD, fontsize=10, fontweight="bold")
        ax.text(0.1, 1.6, f"Tcs\n{res['Tc_out']:.2f}°C", ha="center", va="center",
                color=COLD, fontsize=10, fontweight="bold")

    elif flow_type == "Co-courant":
        # Les deux fluides gauche → droite
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

        ax.text(0.1, 3.55, f"The\n{Th_in:.1f}°C", ha="center", va="center",
                color=HOT, fontsize=10, fontweight="bold")
        ax.text(9.9, 3.55, f"Tfe\n{res['Th_out']:.2f}°C", ha="center", va="center",
                color=HOT, fontsize=10, fontweight="bold")
        ax.text(0.1, 1.55, f"Tce\n{Tc_in:.1f}°C", ha="center", va="center",
                color=COLD, fontsize=10, fontweight="bold")
        ax.text(9.9, 1.55, f"Tcs\n{res['Tc_out']:.2f}°C", ha="center", va="center",
                color=COLD, fontsize=10, fontweight="bold")

    elif flow_type in ("Croisé – non mélangé", "Croisé – Cmin mélangé"):
        # Fluide chaud : gauche → droite
        ax.annotate("", xy=(2.5, 2.6), xytext=(0.3, 2.6),
                    arrowprops=dict(arrowstyle="-|>", color=HOT, lw=2.5), zorder=5)
        ax.plot([0.3, 7.7], [2.6, 2.6], color=HOT, lw=2.5, zorder=4)
        ax.annotate("", xy=(9.7, 2.6), xytext=(7.5, 2.6),
                    arrowprops=dict(arrowstyle="-|>", color=HOT, lw=2.5), zorder=5)

        # Fluide froid : bas → haut (croisé)
        ax.annotate("", xy=(5.0, 1.5), xytext=(5.0, 0.3),
                    arrowprops=dict(arrowstyle="-|>", color=COLD, lw=2.5), zorder=5)
        ax.plot([5.0, 5.0], [0.3, 3.7], color=COLD, lw=2.5, zorder=4)
        ax.annotate("", xy=(5.0, 4.7), xytext=(5.0, 3.5),
                    arrowprops=dict(arrowstyle="-|>", color=COLD, lw=2.5), zorder=5)

        ax.text(0.1, 3.0, f"The\n{Th_in:.1f}°C", ha="center", va="center",
                color=HOT, fontsize=10, fontweight="bold")
        ax.text(9.9, 3.0, f"Tfe\n{res['Th_out']:.2f}°C", ha="center", va="center",
                color=HOT, fontsize=10, fontweight="bold")
        ax.text(5.0, 0.1, f"Tce  {Tc_in:.1f}°C", ha="center", va="center",
                color=COLD, fontsize=10, fontweight="bold")
        ax.text(5.0, 4.9, f"Tcs  {res['Tc_out']:.2f}°C", ha="center", va="center",
                color=COLD, fontsize=10, fontweight="bold")

    # ── Titre du schéma ──
    ax.text(5.0, 4.6, flow_type, ha="center", va="center",
            fontsize=13, fontweight="bold", color=BORDER)

    # ── Données clés ──
    info = (
        f"Q = {res['Q']:,.1f} W   |   ε = {res['epsilon']:.4f}"
        f"   |   NTU = {res['NTU']:.4f}   |   UA = {UA:,.1f} W/K"
    )
    ax.text(5.0, 0.3, info, ha="center", va="center",
            fontsize=9, color=GREEN, fontweight="bold")

    plt.tight_layout()
    return fig


def draw_temperature_profile(res, Th_in, Tc_in):
    """Profil de températures le long de l'échangeur."""
    flow_type = res["flow_type"]
    Th_out = res["Th_out"]
    Tc_out = res["Tc_out"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(0, 1, 200)

    if flow_type == "Co-courant":
        # profil exponentiel simplifié co-courant
        NTU, Cr = res["NTU"], res["Cr"]
        Ch, Cc = res["Ch"], res["Cc"]
        factor = np.exp(-NTU * (1 + Cr) * x)
        Th_x = Th_out + (Th_in - Th_out) * (1 - x)
        Tc_x = Tc_in + (Tc_out - Tc_in) * x
        x_lbl_cold = 1.02
        x_lbl_hot  = 1.02
    else:
        # contre-courant (et approx autres)
        Th_x = Th_out + (Th_in - Th_out) * (1 - x)
        Tc_x = Tc_in + (Tc_out - Tc_in) * (1 - x)  # froid va de droite à gauche
        x_lbl_cold = -0.04
        x_lbl_hot  = 1.02

    ax.plot(x, Th_x, color="#e74c3c", lw=2.5, label="Fluide chaud")
    ax.plot(x, Tc_x, color="#2980b9", lw=2.5,
            label="Fluide froid" + (" (→)" if flow_type == "Co-courant" else " (←)"),
            linestyle="--" if flow_type != "Co-courant" else "-")

    ax.fill_between(x, Tc_x, Th_x, alpha=0.08, color="#e74c3c")
    ax.set_xlabel("Position relative dans l'échangeur (0 → 1)", fontsize=11)
    ax.set_ylabel("Température (°C)", fontsize=11)
    ax.set_title(f"Profil de températures – {flow_type}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    return fig


def draw_eps_vs_NTU(res):
    """ε en fonction de NTU pour R=Cr fixé."""
    Cr = res["Cr"]
    NTU_range = np.linspace(0, 8, 400)

    fig, ax = plt.subplots(figsize=(8, 4))
    styles_map = {
        "Contre-courant":     (epsilon_counter,               "#8e44ad", "-"),
        "Co-courant":         (epsilon_parallel,               "#e67e22", "--"),
        "Croisé – non mélangé": (epsilon_crossflow_unmixed,   "#27ae60", "-."),
        "Calandre (1/2)":     (epsilon_shell_tube,            "#2980b9", ":"),
    }
    for label, (fn, col, ls) in styles_map.items():
        eps_vals = [fn(n, Cr) for n in NTU_range]
        ax.plot(NTU_range, eps_vals, color=col, lw=2, linestyle=ls, label=label)

    # Point de fonctionnement
    ax.plot(res["NTU"], res["epsilon"], "ko", markersize=9, zorder=5,
            label=f"Point de fonctionnement (ε={res['epsilon']:.3f})")

    ax.set_xlabel("NTU", fontsize=11)
    ax.set_ylabel("ε (efficacité)", fontsize=11)
    ax.set_title(f"ε–NTU pour R = Cr = {Cr:.3f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# PDF REPORT GENERATION
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
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Rapport Échangeur ε-NTU"
    )

    styles = getSampleStyleSheet()
    style_title   = ParagraphStyle("MyTitle",   parent=styles["Title"],
                                   fontSize=20, spaceAfter=6,
                                   textColor=colors.HexColor("#2c3e50"))
    style_h1      = ParagraphStyle("MyH1",      parent=styles["Heading1"],
                                   fontSize=14, spaceBefore=14, spaceAfter=4,
                                   textColor=colors.HexColor("#2980b9"))
    style_h2      = ParagraphStyle("MyH2",      parent=styles["Heading2"],
                                   fontSize=12, spaceBefore=8, spaceAfter=4,
                                   textColor=colors.HexColor("#8e44ad"))
    style_body    = ParagraphStyle("MyBody",    parent=styles["Normal"],
                                   fontSize=10, leading=14)
    style_caption = ParagraphStyle("MyCaption", parent=styles["Normal"],
                                   fontSize=9, leading=11,
                                   textColor=colors.grey, alignment=TA_CENTER)
    style_sub     = ParagraphStyle("MySub",     parent=styles["Normal"],
                                   fontSize=9, textColor=colors.HexColor("#555555"))

    story = []
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    # ── ENTÊTE ──
    story.append(Paragraph("Rapport d'Analyse – Échangeur de Chaleur", style_title))
    story.append(Paragraph(f"<font color='grey'>Méthode ε–NTU | Généré le {now}</font>",
                            style_body))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#2980b9"), spaceAfter=10))
    story.append(Spacer(1, 0.3*cm))

    # ── 1. DONNÉES D'ENTRÉE ──
    story.append(Paragraph("1. Données d'entrée", style_h1))

    input_data = [
        ["Paramètre", "Fluide chaud", "Fluide froid"],
        ["Température entrée (°C)", f"{Th_in:.2f}", f"{Tc_in:.2f}"],
        ["Débit massique (kg/s)",   f"{mh:.4f}",    f"{mc:.4f}"],
        ["Capacité thermique cp (J/kg·K)", f"{cph:.2f}", f"{cpc:.2f}"],
        ["Capacité calorifique C = m·cp (W/K)", f"{res['Ch']:.2f}", f"{res['Cc']:.2f}"],
    ]

    t_input = Table(input_data, colWidths=[7*cm, 5*cm, 5*cm])
    t_input.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  10),
        ("BACKGROUND",   (0, 1), (0, -1),  colors.HexColor("#ecf0f1")),
        ("FONTNAME",     (0, 1), (0, -1),  "Helvetica-Bold"),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7f9fc")]),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(t_input)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"<b>Type d'échangeur :</b> {res['flow_type']} &nbsp;&nbsp; "
        f"<b>UA :</b> {UA:,.2f} W/K",
        style_body))
    story.append(Spacer(1, 0.4*cm))

    # ── 2. PARAMÈTRES ADIMENSIONNELS ──
    story.append(Paragraph("2. Paramètres adimensionnels", style_h1))
    story.append(Paragraph(
        "La méthode ε–NTU repose sur deux grandeurs sans dimension :",
        style_body))
    story.append(Spacer(1, 0.2*cm))

    param_data = [
        ["Grandeur", "Formule", "Valeur calculée"],
        ["C_min (W/K)",   "min(Ch, Cc)",              f"{res['Cmin']:.2f}"],
        ["C_max (W/K)",   "max(Ch, Cc)",              f"{res['Cmax']:.2f}"],
        ["R = Cr",        "C_min / C_max",            f"{res['Cr']:.4f}"],
        ["NTU",           "UA / C_min",               f"{res['NTU']:.4f}"],
        ["epsilon (eps)", "f(NTU, Cr, config)",       f"{res['epsilon']:.4f}"],
        ["Q_max (W)",     "C_min × (The - Tce)",      f"{res['Qmax']:,.2f}"],
    ]

    t_param = Table(param_data, colWidths=[5*cm, 6.5*cm, 5.5*cm])
    t_param.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#8e44ad")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN",        (2, 0), (2, -1),  "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7f9fc")]),
        ("BACKGROUND",   (0, 1), (0, -1),  colors.HexColor("#ecf0f1")),
        ("FONTNAME",     (0, 1), (0, -1),  "Helvetica-Bold"),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(t_param)
    story.append(Spacer(1, 0.4*cm))

    # ── 3. RÉSULTATS ──
    story.append(Paragraph("3. Résultats thermiques", style_h1))
    res_data = [
        ["Grandeur",              "Valeur",                 "Unité"],
        ["Flux thermique échangé Q (Phi)", f"{res['Q']:,.2f}", "W"],
        ["Température sortie chaud Tfe",   f"{res['Th_out']:.2f}", "°C"],
        ["Température sortie froid Tcs",   f"{res['Tc_out']:.2f}", "°C"],
        ["Efficacité epsilon",              f"{res['epsilon']:.4f}",  "–"],
        ["NTU",                            f"{res['NTU']:.4f}",      "–"],
        ["UA",                             f"{UA:,.2f}",             "W/K"],
    ]
    t_res = Table(res_data, colWidths=[8*cm, 5*cm, 4*cm])
    t_res.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#27ae60")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7f9fc")]),
        ("BACKGROUND",   (0, 1), (0, -1),  colors.HexColor("#ecf0f1")),
        ("FONTNAME",     (0, 1), (0, -1),  "Helvetica-Bold"),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(t_res)

    # ── Vérification ──
    story.append(Spacer(1, 0.3*cm))
    ok_hot  = res["Th_out"] > Tc_in
    ok_cold = res["Tc_out"] < Th_in
    checks = []
    if ok_hot and ok_cold:
        checks.append("<font color='green'>✔ Configuration physiquement cohérente.</font>")
    else:
        if not ok_hot:
            checks.append("<font color='red'>✖ Tfe &lt; Tce – Croisement de températures !</font>")
        if not ok_cold:
            checks.append("<font color='red'>✖ Tcs &gt; The – Croisement de températures !</font>")
    for c in checks:
        story.append(Paragraph(c, style_body))

    story.append(PageBreak())

    # ── 4. SCHÉMA ──
    story.append(Paragraph("4. Schéma de l'échangeur", style_h1))
    story.append(Spacer(1, 0.2*cm))

    img_buf = fig_to_image_buffer(fig_schema, dpi=150)
    img = RLImage(img_buf, width=17*cm, height=8*cm)
    story.append(img)
    story.append(Paragraph(
        f"Figure 1 – Schéma de l'échangeur {res['flow_type']} avec les températures.",
        style_caption))
    story.append(Spacer(1, 0.5*cm))

    # ── 5. PROFIL DE TEMPÉRATURES ──
    story.append(Paragraph("5. Profil de températures", style_h1))
    img_buf2 = fig_to_image_buffer(fig_profile, dpi=150)
    img2 = RLImage(img_buf2, width=16*cm, height=8*cm)
    story.append(img2)
    story.append(Paragraph(
        "Figure 2 – Évolution des températures des deux fluides le long de l'échangeur.",
        style_caption))
    story.append(Spacer(1, 0.5*cm))

    # ── 6. ε–NTU ──
    story.append(Paragraph("6. Courbes ε–NTU", style_h1))
    img_buf3 = fig_to_image_buffer(fig_eps, dpi=150)
    img3 = RLImage(img_buf3, width=16*cm, height=8*cm)
    story.append(img3)
    story.append(Paragraph(
        f"Figure 3 – Efficacité ε en fonction de NTU pour R = Cr = {res['Cr']:.3f}. "
        "Le point noir indique le point de fonctionnement calculé.",
        style_caption))
    story.append(Spacer(1, 0.5*cm))

    # ── 7. RAPPELS THÉORIQUES ──
    story.append(Paragraph("7. Rappels des formules ε–NTU", style_h1))
    theory_rows = [
        ["Configuration",          "Formule ε"],
        ["Co-courant",             "ε = [1 – exp(–NTU(1+R))] / (1+R)"],
        ["Contre-courant (R≠1)",   "ε = [1 – exp(–NTU(1–R))] / [1 – R·exp(–NTU(1–R))]"],
        ["Contre-courant (R=1)",   "ε = NTU / (1 + NTU)"],
        ["Croisé – non mélangé",   "ε ≈ 1 – exp{[exp(–R·NTU^0.22)–1]/(R·NTU^(–0.78))}"],
        ["Calandre 1P / Tubes 2P", "ε = 2/[1+R+√(1+R²)·(1+exp(–NTU√(1+R²)))/(1–exp(–NTU√(1+R²)))]"],
    ]
    t_theory = Table(theory_rows, colWidths=[6*cm, 11*cm])
    t_theory.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7f9fc")]),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(t_theory)

    # ── PIED DE PAGE ──
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Paragraph(
        f"Rapport généré automatiquement par l'application Échangeurs ε–NTU — {now}",
        style_caption))

    doc.build(story)
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────
# 3D PLOTLY VISUALIZATIONS
# ─────────────────────────────────────────────

def _cylinder_mesh(cx, cy, cz, radius, length, axis="x", n=40, color="red", opacity=0.85, name=""):
    """Return a Plotly Mesh3d approximating a cylinder."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Circle vertices at both ends
    if axis == "x":
        x0 = np.full(n, cx);          x1 = np.full(n, cx + length)
        y0 = cy + radius * np.cos(theta); y1 = y0.copy()
        z0 = cz + radius * np.sin(theta); z1 = z0.copy()
    elif axis == "y":
        y0 = np.full(n, cy);          y1 = np.full(n, cy + length)
        x0 = cx + radius * np.cos(theta); x1 = x0.copy()
        z0 = cz + radius * np.sin(theta); z1 = z0.copy()
    else:  # z
        z0 = np.full(n, cz);          z1 = np.full(n, cz + length)
        x0 = cx + radius * np.cos(theta); x1 = x0.copy()
        y0 = cy + radius * np.sin(theta); y1 = y0.copy()

    # All vertices: end-cap 0 + end-cap 1 + center0 + center1
    xs = np.concatenate([x0, x1, [cx if axis != "x" else cx],
                                  [cx if axis != "x" else cx + length]])
    ys = np.concatenate([y0, y1, [cy if axis != "y" else cy],
                                  [cy if axis != "y" else cy + length]])
    zs = np.concatenate([z0, z1, [cz if axis != "z" else cz],
                                  [cz if axis != "z" else cz + length]])

    # Side triangles
    i_side, j_side, k_side = [], [], []
    for t in range(n):
        t1 = (t + 1) % n
        i_side += [t,     t,     n + t]
        j_side += [n + t, t1,    n + t1]
        k_side += [t1,    n + t, t1]

    # End-cap triangles
    c0, c1 = 2 * n, 2 * n + 1
    i_cap, j_cap, k_cap = [], [], []
    for t in range(n):
        t1 = (t + 1) % n
        i_cap += [c0, c1]
        j_cap += [t,  n + t]
        k_cap += [t1, n + t1]

    i_all = i_side + i_cap
    j_all = j_side + j_cap
    k_all = k_side + k_cap

    return go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i_all, j=j_all, k=k_all,
        color=color, opacity=opacity,
        name=name,
        hoverinfo="name",
        showlegend=True,
        flatshading=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5),
        lightposition=dict(x=100, y=200, z=150),
    )


def _tube_line(x_pts, y_pts, z_pts, color, width=6, name="", dash="solid"):
    """Thin colored line for fluid flow path annotation."""
    return go.Scatter3d(
        x=x_pts, y=y_pts, z=z_pts,
        mode="lines",
        line=dict(color=color, width=width, dash=dash),
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


def _color_gradient_tubes(n_tubes, L, radius, Th_in, Th_out, Tc_in, Tc_out, flow_type):
    """Draw individual internal tubes with temperature-based color gradient."""
    traces = []
    # Arrange tubes in a 2-row grid inside the shell
    cols = max(1, n_tubes // 2)
    y_positions = np.linspace(-0.25, 0.25, cols)
    z_positions = [-0.12, 0.12] if n_tubes > 1 else [0.0]

    for row, zp in enumerate(z_positions):
        for col, yp in enumerate(y_positions):
            tube_idx = row * cols + col
            if tube_idx >= n_tubes:
                break
            # x along the tube
            xs = np.linspace(0, L, 30)
            t_frac = xs / L

            # hot fluid temp along tube
            Th_x = Th_in + (Th_out - Th_in) * t_frac
            # cold fluid (reversed if counter-flow)
            if flow_type == "Contre-courant":
                Tc_x = Tc_out + (Tc_in - Tc_out) * t_frac
            else:
                Tc_x = Tc_in + (Tc_out - Tc_in) * t_frac

            # mix color: hot inside tube → wall temperature ≈ average
            T_wall = (Th_x + Tc_x) / 2
            T_min_all = min(Tc_in, Tc_out)
            T_max_all = max(Th_in, Th_out)
            norm = np.clip((T_wall - T_min_all) / max(T_max_all - T_min_all, 1), 0, 1)

            # colorscale: blue → yellow → red
            for seg in range(len(xs) - 1):
                n_val = norm[seg]
                r = int(min(255, n_val * 2 * 255))
                b = int(min(255, (1 - n_val) * 2 * 255))
                g = int(255 - abs(n_val - 0.5) * 2 * 255)
                seg_color = f"rgb({r},{g},{b})"
                traces.append(go.Scatter3d(
                    x=[xs[seg], xs[seg+1]],
                    y=[yp, yp],
                    z=[zp, zp],
                    mode="lines",
                    line=dict(color=seg_color, width=5),
                    showlegend=False,
                    hoverinfo="skip",
                ))
    return traces


def build_3d_counterflow(res, Th_in, Tc_in, params):
    """3D model: shell-and-tube or plate counter-flow exchanger."""
    L  = params.get("length", 3.0)
    Ro = params.get("shell_radius", 0.5)
    Ri = params.get("tube_radius", 0.08)
    n_tubes = params.get("n_tubes", 6)

    Th_out = res["Th_out"]
    Tc_out = res["Tc_out"]
    flow_type = res["flow_type"]

    traces = []

    # ── Outer shell (transparent) ──
    theta = np.linspace(0, 2 * np.pi, 60)
    x_ring = np.linspace(0, L, 2)
    T_grid, X_grid = np.meshgrid(theta, x_ring)
    Ys = Ro * np.cos(T_grid)
    Zs = Ro * np.sin(T_grid)
    traces.append(go.Surface(
        x=X_grid, y=Ys, z=Zs,
        colorscale=[[0, "rgba(100,160,220,0.12)"], [1, "rgba(100,160,220,0.12)"]],
        showscale=False, opacity=0.18,
        name="Calandre (shell)",
        hoverinfo="name", showlegend=True,
        contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
    ))

    # ── End plates ──
    for x_pos in [0, L]:
        th2 = np.linspace(0, 2 * np.pi, 60)
        r2  = np.linspace(0, Ro, 5)
        TH, RR = np.meshgrid(th2, r2)
        Yp = RR * np.cos(TH)
        Zp = RR * np.sin(TH)
        Xp = np.full_like(Yp, x_pos)
        traces.append(go.Surface(
            x=Xp, y=Yp, z=Zp,
            colorscale=[[0, "rgba(80,80,120,0.35)"], [1, "rgba(80,80,120,0.35)"]],
            showscale=False, opacity=0.35,
            showlegend=False, hoverinfo="skip",
        ))

    # ── Internal tubes with thermal gradient ──
    traces += _color_gradient_tubes(n_tubes, L, Ri, Th_in, Th_out, Tc_in, Tc_out, flow_type)

    # ── Hot fluid arrows (left → right, inside tubes) ──
    traces.append(_tube_line([-0.6, 0], [0, 0], [0, 0], "#e74c3c", 8, "Fluide chaud (entrée)"))
    traces.append(_arrow_cone(0, 0, 0, 0.5, 0, 0, "#e74c3c", "→ chaud"))
    traces.append(_tube_line([L, L + 0.6], [0, 0], [0, 0], "#c0392b", 8, "Fluide chaud (sortie)"))
    traces.append(_arrow_cone(L, 0, 0, 0.5, 0, 0, "#c0392b"))

    # ── Cold fluid arrows – direction depends on flow type ──
    if flow_type == "Contre-courant":
        # right → left (shell side)
        traces.append(_tube_line([L + 0.6, L], [0, 0], [0.55, 0.55], "#2980b9", 8, "Fluide froid (entrée)"))
        traces.append(_arrow_cone(L, 0, 0.55, -0.5, 0, 0, "#2980b9", "← froid"))
        traces.append(_tube_line([0, -0.6], [0, 0], [0.55, 0.55], "#1a6fa0", 8, "Fluide froid (sortie)"))
        traces.append(_arrow_cone(0.5, 0, 0.55, -0.5, 0, 0, "#1a6fa0"))
    else:
        # left → right
        traces.append(_tube_line([-0.6, 0], [0, 0], [0.55, 0.55], "#2980b9", 8, "Fluide froid (entrée)"))
        traces.append(_arrow_cone(0, 0, 0.55, 0.5, 0, 0, "#2980b9", "→ froid"))
        traces.append(_tube_line([L, L + 0.6], [0, 0], [0.55, 0.55], "#1a6fa0", 8, "Fluide froid (sortie)"))
        traces.append(_arrow_cone(L, 0, 0.55, 0.5, 0, 0, "#1a6fa0"))

    # ── Temperature labels ──
    traces += [
        _temp_label(-0.7, 0, 0,    f"The={Th_in:.1f}°C",  "#e74c3c"),
        _temp_label(L+0.1, 0, 0,   f"Tfe={Th_out:.2f}°C", "#c0392b"),
        _temp_label(L+0.1, 0, 0.65,f"Tce={Tc_in:.1f}°C",  "#2980b9") if flow_type == "Contre-courant"
            else _temp_label(-0.7, 0, 0.65, f"Tce={Tc_in:.1f}°C", "#2980b9"),
        _temp_label(-0.7, 0, 0.65, f"Tcs={Tc_out:.2f}°C", "#1a6fa0") if flow_type == "Contre-courant"
            else _temp_label(L+0.1, 0, 0.65, f"Tcs={Tc_out:.2f}°C", "#1a6fa0"),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"<b>Visualisation 3D – {flow_type}</b><br>"
                 f"<sup>Q={res['Q']:,.0f} W | ε={res['epsilon']:.4f} | NTU={res['NTU']:.4f}</sup>",
            x=0.5, font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(title="Longueur (m)", showgrid=True, gridcolor="#ddd"),
            yaxis=dict(title="Y (m)",        showgrid=True, gridcolor="#ddd"),
            zaxis=dict(title="Z (m)",        showgrid=True, gridcolor="#ddd"),
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
    """3D plate cross-flow heat exchanger."""
    L    = params.get("length", 2.5)
    W    = params.get("width",  2.5)
    H    = params.get("height", 1.2)
    n_ch = params.get("n_channels", 7)   # number of channel pairs

    Th_out = res["Th_out"]
    Tc_out = res["Tc_out"]

    traces = []
    channel_h = H / (2 * n_ch + 1)

    for i in range(n_ch):
        z_hot  = (2 * i + 1) * channel_h           # hot channel z-center
        z_cold = (2 * i + 2) * channel_h            # cold channel z-center
        plate_z = (2 * i + 2) * channel_h - channel_h / 2

        # Plate between channels
        Xp = [[0, L], [0, L]]
        Yp = [[0, 0], [W, W]]
        Zp = [[plate_z, plate_z], [plate_z, plate_z]]
        traces.append(go.Surface(
            x=Xp, y=Yp, z=Zp,
            colorscale=[[0, "rgba(80,90,110,0.55)"], [1, "rgba(80,90,110,0.55)"]],
            showscale=False, opacity=0.55,
            showlegend=False, hoverinfo="skip",
        ))

        # Hot fluid channel: flows in x-direction (left→right)
        t_frac = np.linspace(0, 1, 20)
        Th_x = Th_in + (Th_out - Th_in) * t_frac
        Tc_x = Tc_in + (Tc_out - Tc_in) * np.linspace(0, 1, 20)  # y-direction

        for seg in range(len(t_frac) - 1):
            nval = (Th_x[seg] - min(Tc_in, Tc_out)) / max(Th_in - min(Tc_in, Tc_out), 1)
            nval = np.clip(nval, 0, 1)
            r = int(min(255, nval * 2 * 255))
            b = int(min(255, (1 - nval) * 2 * 255))
            g = int(255 - abs(nval - 0.5) * 2 * 255)
            seg_color = f"rgb({r},{g},{b})"
            x_seg = [t_frac[seg] * L, t_frac[seg+1] * L]
            traces.append(go.Scatter3d(
                x=x_seg, y=[W/2, W/2], z=[z_hot, z_hot],
                mode="lines", line=dict(color=seg_color, width=6),
                showlegend=False, hoverinfo="skip"
            ))

        # Cold fluid channel: flows in y-direction (front→back)
        for seg in range(len(t_frac) - 1):
            nval = (Tc_x[seg] - min(Tc_in, Tc_out)) / max(Th_in - min(Tc_in, Tc_out), 1)
            nval = np.clip(nval, 0, 1)
            r = int(min(255, nval * 2 * 255))
            b = int(min(255, (1 - nval) * 2 * 255))
            g = int(255 - abs(nval - 0.5) * 2 * 255)
            seg_color = f"rgb({r},{g},{b})"
            y_seg = [t_frac[seg] * W, t_frac[seg+1] * W]
            traces.append(go.Scatter3d(
                x=[L/2, L/2], y=y_seg, z=[z_cold, z_cold],
                mode="lines", line=dict(color=seg_color, width=6),
                showlegend=False, hoverinfo="skip"
            ))

    # ── Outer box wireframe ──
    box_x = [0,L,L,0,0, 0,L,L,0,0, L,L, 0,0, L,L]
    box_y = [0,0,W,W,0, 0,0,W,W,0, 0,0, 0,0, W,W]
    box_z = [0,0,0,0,0, H,H,H,H,H, 0,H, 0,H, 0,H]
    traces.append(go.Scatter3d(
        x=box_x, y=box_y, z=box_z,
        mode="lines", line=dict(color="#2c3e50", width=3),
        name="Boîtier échangeur", hoverinfo="name"
    ))

    # ── Flow arrows ──
    traces.append(_arrow_cone(-0.3, W/2, H*0.25, 0.5, 0, 0, "#e74c3c", "→ Chaud (x)"))
    traces.append(_arrow_cone(L,    W/2, H*0.25, 0.5, 0, 0, "#c0392b"))
    traces.append(_arrow_cone(L/2, -0.3, H*0.75, 0, 0.5, 0, "#2980b9", "→ Froid (y)"))
    traces.append(_arrow_cone(L/2,  W,   H*0.75, 0, 0.5, 0, "#1a6fa0"))

    # ── Labels ──
    traces += [
        _temp_label(-0.5, W/2, H*0.25, f"The={Th_in:.1f}°C",  "#e74c3c"),
        _temp_label(L+0.1, W/2, H*0.25, f"Tfe={Th_out:.2f}°C", "#c0392b"),
        _temp_label(L/2, -0.5, H*0.75, f"Tce={Tc_in:.1f}°C",  "#2980b9"),
        _temp_label(L/2,  W+0.1, H*0.75, f"Tcs={Tc_out:.2f}°C", "#1a6fa0"),
    ]

    # ── Legend proxies ──
    traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                               line=dict(color="#e74c3c", width=6), name="Fluide chaud"))
    traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                               line=dict(color="#2980b9", width=6), name="Fluide froid"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"<b>Visualisation 3D – {res['flow_type']}</b><br>"
                 f"<sup>Q={res['Q']:,.0f} W | ε={res['epsilon']:.4f} | NTU={res['NTU']:.4f}</sup>",
            x=0.5, font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(title="X – Fluide chaud (m)"),
            yaxis=dict(title="Y – Fluide froid (m)"),
            zaxis=dict(title="Z – Hauteur (m)"),
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
    """Shell & tube – 1 shell pass / 2 tube passes."""
    L     = params.get("length", 3.0)
    Ro    = params.get("shell_radius", 0.55)
    n_t   = params.get("n_tubes", 8)

    Th_out = res["Th_out"]
    Tc_out = res["Tc_out"]
    traces = []

    # ── Shell surface (half transparent) ──
    theta = np.linspace(0, 2 * np.pi, 60)
    x_ring = np.linspace(0, L, 2)
    T_grid, X_grid = np.meshgrid(theta, x_ring)
    Ys = Ro * np.cos(T_grid)
    Zs = Ro * np.sin(T_grid)
    traces.append(go.Surface(
        x=X_grid, y=Ys, z=Zs,
        colorscale=[[0, "rgba(180,200,230,0.13)"], [1, "rgba(180,200,230,0.13)"]],
        showscale=False, opacity=0.18,
        name="Calandre", hoverinfo="name", showlegend=True,
    ))

    # Central baffle (half-moon) at mid-length
    th_baffle = np.linspace(0, np.pi, 40)
    xb = np.full(40, L / 2)
    traces.append(go.Scatter3d(
        x=xb, y=Ro * np.cos(th_baffle), z=Ro * np.sin(th_baffle),
        mode="lines", line=dict(color="rgba(80,80,80,0.6)", width=4),
        name="Chicane", hoverinfo="name"
    ))

    # ── 2-pass tubes: pass-1 left→right (bottom half), pass-2 right→left (top) ──
    cols = n_t // 2
    for c in range(cols):
        yp = np.linspace(-Ro * 0.55, Ro * 0.55, cols)[c]

        for zp, x_start, x_end, pass_name, T_start, T_end in [
            (-0.12, 0,   L,   "Passe 1", Th_in,  (Th_in+Th_out)/2),
            ( 0.12, L,   0,   "Passe 2", (Th_in+Th_out)/2, Th_out),
        ]:
            xs = np.linspace(x_start, x_end, 30)
            t_frac = np.linspace(0, 1, 30)
            T_x = T_start + (T_end - T_start) * t_frac

            for seg in range(len(xs) - 1):
                nval = np.clip((T_x[seg] - min(Tc_in, Tc_out)) /
                               max(Th_in - min(Tc_in, Tc_out), 1), 0, 1)
                r = int(min(255, nval * 2 * 255))
                b = int(min(255, (1 - nval) * 2 * 255))
                g = int(255 - abs(nval - 0.5) * 2 * 255)
                traces.append(go.Scatter3d(
                    x=[xs[seg], xs[seg+1]], y=[yp, yp], z=[zp, zp],
                    mode="lines", line=dict(color=f"rgb({r},{g},{b})", width=5),
                    showlegend=False, hoverinfo="skip"
                ))

    # ── Shell-side (cold) flow – S-shaped path around baffles ──
    xs_cold = [0, L/2, L/2, L, L]
    ys_cold = [0, 0, 0, 0, 0]
    zs_cold = [Ro*0.65, Ro*0.65, -Ro*0.65, -Ro*0.65, -Ro*0.65]
    traces.append(_tube_line(xs_cold, ys_cold, zs_cold, "#2980b9", 7, "Fluide froid (calandre)"))
    traces.append(_arrow_cone(0, 0, Ro*0.65, 0.4, 0, 0, "#2980b9"))
    traces.append(_arrow_cone(L/2, 0, -Ro*0.4, 0, 0, -0.4, "#2980b9"))

    # ── Header boxes ──
    for x_h, clr in [(-0.15, "#c0392b"), (L+0.05, "#c0392b")]:
        traces.append(go.Mesh3d(
            x=[x_h, x_h+0.15, x_h+0.15, x_h, x_h, x_h+0.15, x_h+0.15, x_h],
            y=[-Ro, -Ro, Ro, Ro, -Ro, -Ro, Ro, Ro],
            z=[-Ro, -Ro, -Ro, -Ro, Ro, Ro, Ro, Ro],
            i=[0,0,0,1,1,5], j=[1,2,4,2,5,6], k=[2,3,5,6,6,7],
            color=clr, opacity=0.25, showlegend=False, hoverinfo="skip"
        ))

    # ── Labels ──
    traces += [
        _temp_label(-0.5, 0, 0, f"The={Th_in:.1f}°C", "#e74c3c"),
        _temp_label(L+0.2, 0, 0, f"Tfe={Th_out:.2f}°C", "#c0392b"),
        _temp_label(0,  0, Ro*0.85, f"Tce={Tc_in:.1f}°C", "#2980b9"),
        _temp_label(L,  0, -Ro*0.85, f"Tcs={Tc_out:.2f}°C", "#1a6fa0"),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"<b>Visualisation 3D – Calandre 1P / Tubes 2P</b><br>"
                 f"<sup>Q={res['Q']:,.0f} W | ε={res['epsilon']:.4f} | NTU={res['NTU']:.4f}</sup>",
            x=0.5, font=dict(size=16)
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
    """Route to the correct 3D builder based on flow_type."""
    ft = res["flow_type"]
    if ft in ("Contre-courant", "Co-courant"):
        return build_3d_counterflow(res, Th_in, Tc_in, params)
    elif ft in ("Croisé – non mélangé", "Croisé – Cmin mélangé"):
        return build_3d_crossflow(res, Th_in, Tc_in, params)
    elif ft == "Calandre (1 passe) / Tubes (2 passes)":
        return build_3d_shell_tube(res, Th_in, Tc_in, params)
    else:
        return build_3d_counterflow(res, Th_in, Tc_in, params)


# ─────────────────────────────────────────────
# THEORY PAGE
# ─────────────────────────────────────────────

def show_theory_page():
    # ── Global dark engineering CSS ──
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    /* Dark background for entire app */
    .stApp { background: #0b0f1a !important; }
    section[data-testid="stSidebar"] { background: #080c14 !important; }
    .block-container { padding-top: 2rem !important; }

    /* Step card */
    .step-card {
        background: #0f1623;
        border: 1px solid #1e2d45;
        border-left: 3px solid #f6ad55;
        border-radius: 6px;
        padding: 28px 32px;
        margin-bottom: 32px;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .step-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 3px;
        color: #f6ad55;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .step-title {
        font-size: 20px;
        font-weight: 500;
        color: #e2e8f0;
        margin-bottom: 12px;
        font-family: 'IBM Plex Sans', sans-serif;
        letter-spacing: -0.2px;
    }
    .step-body {
        font-size: 14px;
        color: #8899aa;
        line-height: 1.8;
        font-weight: 300;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .step-body b { color: #cbd5e0; font-weight: 500; }

    /* Inline highlight */
    .hl { color: #f6ad55; font-family: 'IBM Plex Mono', monospace; font-size: 13px; }

    /* Config table */
    .cfg-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 13px;
        margin-top: 16px;
    }
    .cfg-table th {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #4a5568;
        padding: 8px 14px;
        border-bottom: 1px solid #1e2d45;
        text-align: left;
    }
    .cfg-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #111827;
        color: #718096;
        vertical-align: top;
        line-height: 1.55;
    }
    .cfg-table td:first-child { color: #a0aec0; font-weight: 500; }
    .cfg-table td:last-child  { color: #68d391; font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
    .cfg-table tr:hover td    { background: rgba(255,255,255,0.015); }

    /* Validity chips */
    .chip-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }
    .chip {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        padding: 4px 12px;
        border-radius: 3px;
        font-weight: 600;
        letter-spacing: 0.3px;
        border: 1px solid;
    }
    .chip-ok  { color: #68d391; border-color: rgba(104,211,145,0.3); background: rgba(104,211,145,0.07); }
    .chip-warn{ color: #f6ad55; border-color: rgba(246,173,85,0.3);  background: rgba(246,173,85,0.07); }
    .chip-err { color: #fc8181; border-color: rgba(252,129,129,0.3); background: rgba(252,129,129,0.07); }

    /* Summary table */
    .sum-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        margin-top: 12px;
    }
    .sum-table th {
        background: #0d1117;
        color: #4a5568;
        font-size: 9px;
        letter-spacing: 2px;
        padding: 9px 14px;
        text-align: left;
        border-bottom: 1px solid #1e2d45;
    }
    .sum-table td {
        padding: 9px 14px;
        border-bottom: 1px solid #111827;
        color: #a0aec0;
    }
    .sum-table td:first-child { color: #63b3ed; }
    .sum-table td:nth-child(2){ color: #68d391; font-size: 13px; }
    .sum-table tr:hover td    { background: rgba(255,255,255,0.015); }

    /* Section divider */
    .sec-divider {
        height: 1px;
        background: linear-gradient(to right, #1e2d45, transparent);
        margin: 40px 0;
    }
    /* Header */
    .theory-header {
        border-bottom: 1px solid #1e2d45;
        padding-bottom: 24px;
        margin-bottom: 40px;
    }
    .theory-header h2 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 30px;
        font-weight: 300;
        color: #f7fafc;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .theory-header h2 span { color: #f6ad55; font-weight: 600; }
    .theory-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #4a5568;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div class="theory-header">
        <div class="theory-sub">THERMIQUE · GÉNIE THERMIQUE · TRANSFERT DE CHALEUR</div>
        <h2>Méthode <span>ε – NTU</span><br>Théorie & Méthodologie</h2>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # STEP 1 — Bilan enthalpique
    # ══════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Étape 01</div>
        <div class="step-title">Bilan enthalpique — Capacités calorifiques</div>
        <div class="step-body">
            Point de départ : chaque fluide transporte une puissance thermique
            proportionnelle à son <b>débit massique</b> et sa <b>chaleur spécifique</b>.
            On définit la <b>capacité calorifique</b> C (W/K) pour chaque fluide.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"C_h = \dot{m}_h \cdot c_{p,h} \qquad \text{[W/K]}")
    st.latex(r"C_c = \dot{m}_c \cdot c_{p,c} \qquad \text{[W/K]}")
    st.latex(r"\Phi = C_h\,(T_{h,e} - T_{h,s}) = C_c\,(T_{c,s} - T_{c,e})")

    st.markdown("""
    <div class="step-card" style="border-left-color:#63b3ed; margin-top:20px;">
        <div class="step-label" style="color:#63b3ed;">Notations</div>
        <div class="step-body">
            <b>T<sub>h,e</sub></b> : température entrée fluide chaud &nbsp;·&nbsp;
            <b>T<sub>h,s</sub></b> : sortie chaud (T<sub>fe</sub>)<br>
            <b>T<sub>c,e</sub></b> : température entrée fluide froid &nbsp;·&nbsp;
            <b>T<sub>c,s</sub></b> : sortie froid (T<sub>cs</sub>)<br>
            <b>Φ</b> : flux thermique échangé (W) &nbsp;·&nbsp;
            <b>ṁ</b> : débit massique (kg/s) &nbsp;·&nbsp;
            <b>c<sub>p</sub></b> : chaleur massique (J/kg·K)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # STEP 2 — Cmin, Cmax, R
    # ══════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Étape 02</div>
        <div class="step-title">Identification de C<sub>min</sub>, C<sub>max</sub> et rapport R</div>
        <div class="step-body">
            Le fluide dont la capacité calorifique est la plus faible subit la plus grande
            variation de température — il est le <b>fluide limitant</b>.
            Le rapport <b>R = C<sub>min</sub>/C<sub>max</sub></b> (aussi noté C*) est
            toujours compris entre 0 et 1 par définition.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"C_{\min} = \min(C_h,\, C_c), \qquad C_{\max} = \max(C_h,\, C_c)")
    st.latex(r"R = \frac{C_{\min}}{C_{\max}}, \qquad 0 < R \leq 1")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="step-card" style="border-left-color:#68d391; margin-top:0;">
            <div class="step-label" style="color:#68d391;">Cas R → 0</div>
            <div class="step-body">
                Un fluide a une capacité très grande (ex. condensation/évaporation).
                Sa température reste quasi constante.
                <b>ε → 1 − e<sup>−NTU</sup></b>, indépendant de la configuration.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="step-card" style="border-left-color:#fc8181; margin-top:0;">
            <div class="step-label" style="color:#fc8181;">Cas R = 1</div>
            <div class="step-body">
                Les deux fluides ont la même capacité. En contre-courant :
                <b>ε = NTU/(1+NTU)</b>.
                En co-courant, la limite thermique est ε = 0.5.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # STEP 3 — Flux maximal & efficacité
    # ══════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Étape 03</div>
        <div class="step-title">Flux maximal théorique & Efficacité ε</div>
        <div class="step-body">
            Le flux <b>maximum théorique Φ<sub>max</sub></b> serait échangé dans un
            échangeur de <b>longueur infinie</b> (NTU → ∞) : le fluide limitant
            atteint la température d'entrée de l'autre fluide.<br><br>
            L'<b>efficacité ε</b> mesure le rapport entre le flux réellement échangé
            et ce maximum thermodynamique.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\Phi_{\max} = C_{\min}\,\bigl(T_{h,e} - T_{c,e}\bigr)")
    st.latex(r"\varepsilon = \frac{\Phi}{\Phi_{\max}} = \frac{C_h(T_{h,e}-T_{h,s})}{C_{\min}(T_{h,e}-T_{c,e})} \in [0,\,1]")

    st.markdown("""
    <div class="step-body" style="font-family: IBM Plex Sans; font-size:13.5px; color:#718096; margin-top:8px;">
        → Une fois ε connu, les températures de sortie s'obtiennent directement :
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"T_{h,s} = T_{h,e} - \frac{\varepsilon\,\Phi_{\max}}{C_h}")
    st.latex(r"T_{c,s} = T_{c,e} + \frac{\varepsilon\,\Phi_{\max}}{C_c}")

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # STEP 4 — NTU
    # ══════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Étape 04</div>
        <div class="step-title">Nombre de Transfert d'Unités — NTU</div>
        <div class="step-body">
            Le NTU (ou NUT en français) est un nombre adimensionnel qui quantifie
            la <b>capacité d'échange thermique</b> de l'échangeur relativement
            au fluide limitant. Il dépend du coefficient global d'échange <b>U</b>
            (W/m²·K) et de la surface d'échange <b>A</b> (m²), regroupés dans le
            produit <b>UA</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\mathrm{NTU} = \frac{UA}{C_{\min}}")
    st.latex(r"UA = \frac{1}{R_{tot}} = \frac{1}{\dfrac{1}{h_h A} + R_{paroi} + \dfrac{1}{h_c A}}")

    col3, col4 = st.columns([1,1])
    with col3:
        st.markdown("""
        <div class="step-card" style="border-left-color:#b794f4; margin-top:0;">
            <div class="step-label" style="color:#b794f4;">NTU faible (< 0.5)</div>
            <div class="step-body">
                Échangeur sous-dimensionné ou fluides à grands débits.
                ε reste faible, peu d'échange thermique réalisé.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="step-card" style="border-left-color:#68d391; margin-top:0;">
            <div class="step-label" style="color:#68d391;">NTU élevé (> 3)</div>
            <div class="step-body">
                Échangeur bien dimensionné. ε tend vers 1 (limite).
                Au-delà de NTU ≈ 5, le gain marginal est faible.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # STEP 5 — Formules ε(NTU, R)
    # ══════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Étape 05</div>
        <div class="step-title">Formules analytiques ε = f(NTU, R) par configuration</div>
        <div class="step-body">
            Chaque configuration géométrique possède sa propre relation analytique
            entre ε, NTU et R. Ces formules sont dérivées de la résolution du système
            d'équations différentielles décrivant le transfert de chaleur le long
            de l'échangeur.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Contre-courant
    st.markdown("#### 🔁 Contre-courant *(configuration la plus efficace)*")
    st.latex(r"""
    \varepsilon = \begin{cases}
        \dfrac{1 - e^{-\mathrm{NTU}(1-R)}}{1 - R\,e^{-\mathrm{NTU}(1-R)}} & \text{si } R \neq 1 \\[12pt]
        \dfrac{\mathrm{NTU}}{1 + \mathrm{NTU}} & \text{si } R = 1
    \end{cases}
    """)

    # Co-courant
    st.markdown("#### ➡️ Co-courant *(parallèle)*")
    st.latex(r"""
    \varepsilon = \frac{1 - e^{-\mathrm{NTU}(1+R)}}{1 + R}
    """)
    st.markdown("""
    <div class="step-body" style="font-family:IBM Plex Sans; font-size:13px; color:#718096;">
    ⚠️ Limite thermique : ε<sub>max</sub> = 1/(1+R) atteint quand NTU → ∞.
    Pour R = 1, ε ≤ 0.5 — jamais possible de récupérer plus de 50% en co-courant.
    </div>
    """, unsafe_allow_html=True)

    # Croisé
    st.markdown("#### ✛ Croisé — fluides non mélangés *(corrélation approchée)*")
    st.latex(r"""
    \varepsilon \approx 1 - \exp\!\left(\frac{e^{-R\,\mathrm{NTU}^{0.22}}-1}{R\,\mathrm{NTU}^{-0.78}}\right)
    """)

    # Calandre
    st.markdown("#### 🔩 Calandre 1 passe / Tubes 2 passes")
    st.latex(r"""
    \varepsilon = \frac{2}{1 + R + \sqrt{1+R^2}\;\dfrac{1+e^{-\mathrm{NTU}\sqrt{1+R^2}}}{1-e^{-\mathrm{NTU}\sqrt{1+R^2}}}}
    """)

    # Config table
    st.markdown("""
    <table class="cfg-table">
        <tr>
            <th>Configuration</th>
            <th>ε max théorique (NTU → ∞)</th>
            <th>Avantage principal</th>
        </tr>
        <tr>
            <td>Contre-courant</td>
            <td>ε → 1 (R quelconque)</td>
            <td>Meilleur rendement thermique possible</td>
        </tr>
        <tr>
            <td>Co-courant</td>
            <td>ε → 1/(1+R)</td>
            <td>Simple à construire, contrôle T sortie</td>
        </tr>
        <tr>
            <td>Croisé non mélangé</td>
            <td>Entre co et contre-courant</td>
            <td>Compacité, aéro-thermique</td>
        </tr>
        <tr>
            <td>Calandre 1P/2P</td>
            <td>Proche contre-courant</td>
            <td>Standard industriel, robuste</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # STEP 6 — Dimensionnement inverse
    # ══════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Étape 06</div>
        <div class="step-title">Dimensionnement inverse — Trouver UA pour une cible T</div>
        <div class="step-body">
            En mode dimensionnement, on fixe une <b>température de sortie cible</b>
            (T<sub>cs</sub> ou T<sub>fe</sub>) et on cherche le <b>UA nécessaire</b>.
            Il n'existe pas de formule inverse analytique simple pour toutes les
            configurations — on résout numériquement par <b>dichotomie</b> (bisection).
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Algorithme de bisection :**")
    st.latex(r"""
    \text{Trouver } UA^* \text{ tel que : } f(UA^*) = T_{sortie}(UA^*) - T_{cible} = 0
    """)
    st.latex(r"""
    UA_{mid} = \frac{UA_{low} + UA_{high}}{2}, \quad
    \text{convergence : } |f(UA_{mid})| < 10^{-8}
    """)

    st.markdown("""
    <div class="step-body" style="font-family:IBM Plex Sans; font-size:13.5px; color:#718096; margin-top:8px;">
        L'algorithme converge en <b>≤ 100 itérations</b> pour UA ∈ [10⁻⁶, 10⁸] W/K.
        Il détecte automatiquement si la cible est physiquement <b>impossible</b>
        (croisement de températures, limites thermodynamiques).
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # STEP 7 — Synthèse algorithmique
    # ══════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Étape 07</div>
        <div class="step-title">Synthèse — Algorithme complet de calcul</div>
        <div class="step-body">
            Résumé de la séquence de calcul implémentée dans cette application,
            du bilan d'entrée jusqu'aux températures de sortie.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <table class="sum-table">
        <tr>
            <th>#</th><th>Calcul</th><th>Formule</th><th>Résultat</th>
        </tr>
        <tr><td>1</td><td>Capacités calorifiques</td>
            <td>C = ṁ · cp</td><td>Ch, Cc [W/K]</td></tr>
        <tr><td>2</td><td>Fluide limitant</td>
            <td>Cmin = min(Ch, Cc)</td><td>Cmin, Cmax [W/K]</td></tr>
        <tr><td>3</td><td>Rapport de capacités</td>
            <td>R = Cmin/Cmax</td><td>R ∈ [0, 1]</td></tr>
        <tr><td>4</td><td>Nombre de transfert</td>
            <td>NTU = UA/Cmin</td><td>NTU [−]</td></tr>
        <tr><td>5</td><td>Efficacité</td>
            <td>ε = f(NTU, R, config)</td><td>ε ∈ [0, 1]</td></tr>
        <tr><td>6</td><td>Flux max théorique</td>
            <td>Φmax = Cmin·(The − Tce)</td><td>Φmax [W]</td></tr>
        <tr><td>7</td><td>Flux échangé réel</td>
            <td>Φ = ε · Φmax</td><td>Φ [W]</td></tr>
        <tr><td>8</td><td>Températures de sortie</td>
            <td>Tfe = The − Φ/Ch ; Tcs = Tce + Φ/Cc</td><td>Tfe, Tcs [°C]</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # STEP 8 — Validité & Hypothèses
    # ══════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-label">Étape 08</div>
        <div class="step-title">Hypothèses & Domaine de validité</div>
        <div class="step-body">
            La méthode ε–NTU repose sur un ensemble d'hypothèses simplificatrices.
            Leur respect conditionne la validité du résultat.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="chip-row">
        <span class="chip chip-ok">✓ Régime permanent</span>
        <span class="chip chip-ok">✓ Propriétés constantes</span>
        <span class="chip chip-ok">✓ Pas de pertes vers l'extérieur</span>
        <span class="chip chip-ok">✓ Échange par convection seul</span>
        <span class="chip chip-warn">⚠ Pas de changement de phase</span>
        <span class="chip chip-warn">⚠ Pas de génération interne</span>
        <span class="chip chip-err">✗ Non valide si T croisent</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="step-body" style="font-family:IBM Plex Sans; font-size:13.5px; color:#718096; margin-top:14px;">
        <b>Condition de non-croisement :</b>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"T_{h,s} > T_{c,e} \qquad \text{et} \qquad T_{c,s} < T_{h,e}")

    st.markdown("""
    <div class="step-body" style="font-family:IBM Plex Sans; font-size:13.5px; color:#718096; margin-top:8px;">
        En contre-courant, ces deux conditions sont toujours satisfaites physiquement.
        En co-courant, on doit vérifier que l'efficacité ε ne dépasse pas 1/(1+R).
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # Références
    # ══════════════════════════════════════════
    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step-card" style="border-left-color:#4a5568;">
        <div class="step-label" style="color:#718096;">Références</div>
        <div class="step-body">
            · Incropera, F.P. et al. — <i>Fundamentals of Heat and Mass Transfer</i>, 7ᵉ éd., Wiley<br>
            · Çengel, Y.A. — <i>Transfert thermique</i>, De Boeck<br>
            · Kays, W.M. & London, A.L. — <i>Compact Heat Exchangers</i>, McGraw-Hill<br>
            · Rohsenow, W.M. et al. — <i>Handbook of Heat Transfer</i>, 3ᵉ éd., McGraw-Hill
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

st.title("🌡️ Échangeurs de Chaleur – Méthode ε–NTU")

# ── Sidebar (global – visible on both tabs) ──
with st.sidebar:
    st.header("⚙️ Configuration")

    flow_type = st.selectbox("Type d'échangeur", [
        "Contre-courant",
        "Co-courant",
        "Croisé – non mélangé",
        "Croisé – Cmin mélangé",
        "Calandre (1 passe) / Tubes (2 passes)",
    ])

    st.divider()
    st.subheader("🔴 Fluide chaud")
    Th_in = st.number_input("The – Température entrée (°C)", value=80.0, step=1.0)
    mh    = st.number_input("ṁh – Débit massique (kg/s)", value=1.0, min_value=0.001, step=0.1)
    cph   = st.number_input("cph – Chaleur spécifique (J/kg·K)", value=4180.0, min_value=1.0, step=10.0)

    st.divider()
    st.subheader("🔵 Fluide froid")
    Tc_in = st.number_input("Tce – Température entrée (°C)", value=20.0, step=1.0)
    mc    = st.number_input("ṁc – Débit massique (kg/s)", value=1.0, min_value=0.001, step=0.1)
    cpc   = st.number_input("cpc – Chaleur spécifique (J/kg·K)", value=4180.0, min_value=1.0, step=10.0)

    st.divider()
    st.subheader("🧊 Géométrie 3D")
    length_3d = st.slider("Longueur échangeur (m)", 1.0, 6.0, 3.0, 0.1)
    if flow_type in ("Contre-courant", "Co-courant", "Calandre (1 passe) / Tubes (2 passes)"):
        n_tubes = st.slider("Nombre de tubes internes", 2, 16, 6, 2)
        shell_r = st.slider("Rayon calandre (m)", 0.2, 1.0, 0.5, 0.05)
        params_3d = {"length": length_3d, "shell_radius": shell_r,
                     "tube_radius": 0.05, "n_tubes": n_tubes}
    else:
        n_ch = st.slider("Nombre de canaux", 3, 12, 6, 1)
        params_3d = {"length": length_3d, "width": length_3d,
                     "height": 1.2, "n_channels": n_ch}
    st.divider()

# ── Tabs navigation ──
tab_calc, tab_theory = st.tabs(["⚙️  Calculateur", "📐  Théorie & Méthodologie"])

with tab_theory:
    show_theory_page()

with tab_calc:
    st.markdown(
        "Calculez **Q (Φ)**, les **températures de sortie**, l'**efficacité** et le "
        "**NTU** pour différentes configurations d'échangeurs. "
        "Téléchargez un **rapport PDF complet** avec schémas."
    )

    # ── Mode principal ──
    col_mode, _ = st.columns([3, 1])
    with col_mode:
        mode = st.radio(
            "Mode de calcul",
            ["Calcul direct (UA connu)", "Dimensionnement (cible de température)"],
            horizontal=True
        )

    UA = None
    res = None

    if mode == "Calcul direct (UA connu)":
        UA = st.number_input("UA (W/K)", value=2000.0, min_value=0.0, step=100.0)
    else:
        c1, c2 = st.columns(2)
        with c1:
            target_choice = st.selectbox("Température cible", [
                "Tcs – Sortie froid", "Tfe – Sortie chaud"
            ])
        with c2:
            target_value = st.number_input("Valeur cible (°C)", value=50.0, step=1.0)

        target_key = "Tc_out" if target_choice.startswith("Tcs") else "Th_out"

        if st.button("🔍 Calculer UA requis", type="primary"):
            UA_sol = solve_UA_for_target(
                Th_in, Tc_in, mh, cph, mc, cpc, flow_type,
                target=target_key, target_value=target_value
            )
            if UA_sol is None:
                st.error(
                    "❌ Aucune solution trouvée. Vérifiez que la cible est physiquement "
                    "atteignable (non hors plage thermique)."
                )
            else:
                UA = float(UA_sol)
                st.success(f"✅ UA requis ≈ **{UA:,.2f} W/K**")

        # Garder UA entre re-runs via session state
        if UA is None and "last_UA" in st.session_state:
            UA = st.session_state["last_UA"]
        if UA is not None:
            st.session_state["last_UA"] = UA

    # ── Calcul et affichage ──
    if UA is not None and UA > 0:
        if Th_in <= Tc_in:
            st.error("❌ The doit être supérieure à Tce.")
        else:
            res = compute_outputs(Th_in, Tc_in, mh, cph, mc, cpc, UA, flow_type)
            if res is None:
                st.error("❌ Débits ou capacités thermiques invalides.")
            else:
                # ── Métriques ──
                st.markdown("---")
                st.subheader("📊 Résultats")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Φ = Q (W)",      f"{res['Q']:,.1f}")
                m2.metric("ε (efficacité)", f"{res['epsilon']:.4f}")
                m3.metric("NTU",            f"{res['NTU']:.4f}")
                m4.metric("Tfe (°C)",       f"{res['Th_out']:.2f}")
                m5.metric("Tcs (°C)",       f"{res['Tc_out']:.2f}")

                # Check physique
                if res["Th_out"] < Tc_in or res["Tc_out"] > Th_in:
                    st.warning("⚠️ Croisement de températures détecté – vérifiez les paramètres.")
                else:
                    st.success("✅ Configuration physiquement cohérente.")

                with st.expander("🔎 Détails complets (Cmin, Cmax, Cr, Qmax...)"):
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Ch (W/K)",   f"{res['Ch']:.2f}")
                    d1.metric("Cc (W/K)",   f"{res['Cc']:.2f}")
                    d2.metric("Cmin (W/K)", f"{res['Cmin']:.2f}")
                    d2.metric("Cmax (W/K)", f"{res['Cmax']:.2f}")
                    d3.metric("R = Cr",     f"{res['Cr']:.4f}")
                    d3.metric("UA (W/K)",   f"{UA:,.2f}")
                    d4.metric("Qmax (W)",   f"{res['Qmax']:,.1f}")

                # ── Schéma ──
                st.markdown("---")
                st.subheader("🖼️ Schéma de l'échangeur")
                fig_schema = draw_schema(res, UA, Th_in, Tc_in)
                st.pyplot(fig_schema)
                plt.close(fig_schema)

                # ── Profil de températures ──
                st.subheader("📈 Profil de températures")
                fig_profile = draw_temperature_profile(res, Th_in, Tc_in)
                st.pyplot(fig_profile)
                plt.close(fig_profile)

                # ── Courbe ε–NTU ──
                st.subheader("📉 Courbes ε–NTU (toutes configurations)")
                fig_eps = draw_eps_vs_NTU(res)
                st.pyplot(fig_eps)
                plt.close(fig_eps)

                # ── Visualisation 3D ──
                st.markdown("---")
                st.subheader("🧊 Visualisation 3D interactive")
                st.markdown(
                    "Modèle 3D de l'échangeur avec gradient thermique réel. "
                    "**Tournez**, **zoomez** et **explorez** le modèle librement. "
                    "Le dégradé de couleur bleu→rouge représente la montée en température."
                )

                with st.spinner("Génération du modèle 3D..."):
                    fig_3d = build_3d_visualization(res, Th_in, Tc_in, params_3d)
                st.plotly_chart(fig_3d, use_container_width=True)

                # Légende thermique colorbar
                col_leg1, col_leg2, col_leg3 = st.columns([1,2,1])
                with col_leg2:
                    cbar_fig = go.Figure(go.Heatmap(
                        z=[np.linspace(min(Tc_in, res["Tc_out"]),
                                       max(Th_in, res["Th_out"]), 100).reshape(1, -1)[0]],
                        colorscale=[
                            [0.0,  "rgb(0,0,255)"],
                            [0.25, "rgb(0,200,255)"],
                            [0.5,  "rgb(0,255,0)"],
                            [0.75, "rgb(255,200,0)"],
                            [1.0,  "rgb(255,0,0)"],
                        ],
                        showscale=True,
                        colorbar=dict(
                            title="T (°C)", titleside="right",
                            thickness=18, len=0.9,
                            tickvals=[
                                min(Tc_in, res["Tc_out"]),
                                (min(Tc_in, res["Tc_out"]) + max(Th_in, res["Th_out"])) / 2,
                                max(Th_in, res["Th_out"]),
                            ],
                            ticktext=[
                                f"{min(Tc_in, res['Tc_out']):.1f}°C (froid)",
                                f"{(min(Tc_in, res['Tc_out'])+max(Th_in, res['Th_out']))/2:.1f}°C",
                                f"{max(Th_in, res['Th_out']):.1f}°C (chaud)",
                            ],
                        )
                    ))
                    cbar_fig.update_layout(
                        height=120, margin=dict(l=0, r=100, t=10, b=10),
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(cbar_fig, use_container_width=True)

                # ── Génération PDF ──
                st.markdown("---")
                st.subheader("📄 Rapport PDF")
                if st.button("📥 Générer et télécharger le rapport PDF", type="primary"):
                    with st.spinner("Génération du rapport en cours..."):
                        # Re-créer les figures pour le PDF
                        f_schema  = draw_schema(res, UA, Th_in, Tc_in)
                        f_profile = draw_temperature_profile(res, Th_in, Tc_in)
                        f_eps     = draw_eps_vs_NTU(res)

                        pdf_buf = generate_pdf(
                            res, UA, Th_in, Tc_in, mc, mh, cpc, cph,
                            f_schema, f_profile, f_eps
                        )
                        plt.close("all")

                        fname = f"rapport_echangeur_{flow_type.replace(' ','_').replace('–','').replace('/','_')}.pdf"
                        st.download_button(
                            label="⬇️ Télécharger le rapport PDF",
                            data=pdf_buf,
                            file_name=fname,
                            mime="application/pdf"
                        )
                        st.success("✅ Rapport prêt !")

    elif mode == "Calcul direct (UA connu)" and UA == 0:
        st.info("ℹ️ Entrez un UA > 0 pour lancer le calcul.")