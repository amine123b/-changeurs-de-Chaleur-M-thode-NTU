"""
Echangeurs de Chaleur — Application Streamlit Avancee
Methodes : e-NTU, LMTD, Stanton
Surfaces d'echange, nombres adimensionnels, visualisation 3D avancee, rapport PDF.
"""

import io
import math
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
    Image as RLImage, HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

st.set_page_config(page_title="Echangeurs Avances", layout="wide")

# ═══════════════════════════════════════════════════════════════
# ── MODULE 1 : PHYSIQUE e-NTU ──────────────────────────────────
# ═══════════════════════════════════════════════════════════════

FLOW_TYPES = [
    "Contre-courant",
    "Co-courant",
    "Croise non melange",
    "Croise Cmin melange",
    "Calandre (1 passe) / Tubes (2 passes)",
]


def epsilon_parallel(NTU: float, Cr: float) -> float:
    """Co-current (parallel flow) effectiveness.  Ref: Incropera eq. 11.29"""
    return (1.0 - np.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)


def epsilon_counter(NTU: float, Cr: float) -> float:
    """Counter-flow effectiveness.  Ref: Incropera eq. 11.30"""
    if np.isclose(Cr, 1.0):
        return NTU / (1.0 + NTU)
    a = NTU * (1.0 - Cr)
    return (1.0 - np.exp(-a)) / (1.0 - Cr * np.exp(-a))


def epsilon_crossflow_unmixed(NTU: float, Cr: float) -> float:
    """Cross-flow, both fluids unmixed.  Ref: Incropera eq. 11.32"""
    return 1.0 - np.exp(
        (np.exp(-Cr * NTU ** 0.22) - 1.0) / (Cr * NTU ** (-0.78) + 1e-12)
    )


def epsilon_crossflow_mixed_Cmin(NTU: float, Cr: float) -> float:
    """Cross-flow, Cmin mixed.  Ref: Incropera eq. 11.33"""
    return (1.0 / Cr) * (1.0 - np.exp(-Cr * (1.0 - np.exp(-NTU))))


def epsilon_shell_tube(NTU: float, Cr: float) -> float:
    """Shell-and-tube 1 shell pass / 2 tube passes.  Ref: Incropera eq. 11.35"""
    sq = np.sqrt(1.0 + Cr ** 2)
    num = 1.0 + np.exp(-NTU * sq)
    den = 1.0 - np.exp(-NTU * sq)
    if abs(den) < 1e-12:
        return 1.0
    return 2.0 / (1.0 + Cr + sq * num / den)


def compute_entu(Th_in, Tc_in, mh, cph, mc, cpc, UA, flow_type):
    """Compute all e-NTU outputs.

    Returns dict with thermal results or None on invalid inputs.
    Units: temperatures [C], mass flows [kg/s], cp [J/kg/K], UA [W/K].
    """
    Ch = mh * cph   # [W/K]
    Cc = mc * cpc   # [W/K]
    if Ch <= 0 or Cc <= 0:
        return None
    Cmin = min(Ch, Cc)
    Cmax = max(Ch, Cc)
    Cr = Cmin / Cmax
    NTU = UA / Cmin

    eps_fn = {
        "Co-courant":                       epsilon_parallel,
        "Contre-courant":                   epsilon_counter,
        "Croise non melange":               epsilon_crossflow_unmixed,
        "Croise Cmin melange":              epsilon_crossflow_mixed_Cmin,
        "Calandre (1 passe) / Tubes (2 passes)": epsilon_shell_tube,
    }
    eps = eps_fn.get(flow_type, epsilon_counter)(NTU, Cr)
    eps = min(eps, 1.0)

    Qmax = Cmin * (Th_in - Tc_in)   # [W]
    Q    = eps * Qmax                 # [W]
    Th_out = Th_in - Q / Ch
    Tc_out = Tc_in + Q / Cc

    return dict(
        Ch=Ch, Cc=Cc, Cmin=Cmin, Cmax=Cmax,
        Cr=Cr, NTU=NTU, epsilon=eps,
        Qmax=Qmax, Q=Q,
        Th_out=Th_out, Tc_out=Tc_out,
        flow_type=flow_type,
        method="e-NTU",
    )


def solve_UA_bisection(Th_in, Tc_in, mh, cph, mc, cpc, flow_type,
                       target="Tc_out", target_value=40.0):
    """Bisection solver to find UA for a target outlet temperature. [W/K]"""
    UA_low, UA_high = 1e-6, 1e8

    def f(UA):
        r = compute_entu(Th_in, Tc_in, mh, cph, mc, cpc, UA, flow_type)
        return np.nan if r is None else r[target] - target_value

    fl, fh = f(UA_low), f(UA_high)
    if np.isnan(fl) or np.isnan(fh) or fl * fh > 0:
        return None
    for _ in range(120):
        UA_mid = 0.5 * (UA_low + UA_high)
        fm = f(UA_mid)
        if abs(fm) < 1e-9:
            break
        if fl * fm <= 0:
            UA_high, fh = UA_mid, fm
        else:
            UA_low, fl = UA_mid, fm
    return 0.5 * (UA_low + UA_high)


# ═══════════════════════════════════════════════════════════════
# ── MODULE 2 : METHODE LMTD ───────────────────────────────────
# ═══════════════════════════════════════════════════════════════

def compute_lmtd(Th_in, Th_out, Tc_in, Tc_out, flow="counter"):
    """Compute LMTD with numerically stable formula.

    Parameters
    ----------
    flow : 'counter' or 'parallel'

    Returns
    -------
    dict with dT1, dT2, LMTD [K or °C difference]
    """
    if flow == "counter":
        dT1 = Th_in  - Tc_out   # hot inlet vs cold outlet
        dT2 = Th_out - Tc_in    # hot outlet vs cold inlet
    else:  # parallel / co-current
        dT1 = Th_in  - Tc_in
        dT2 = Th_out - Tc_out

    if dT1 <= 0 or dT2 <= 0:
        return dict(dT1=dT1, dT2=dT2, LMTD=None, valid=False,
                    warning="Croisement de temperatures — LMTD non defini.")

    if abs(dT1 - dT2) < 1e-6:
        lmtd = dT1  # numerically stable limit when dT1 ≈ dT2
    else:
        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

    return dict(dT1=dT1, dT2=dT2, LMTD=lmtd, valid=True, warning=None)


def compute_lmtd_full(Th_in, Tc_in, mh, cph, mc, cpc, UA, lmtd_flow, flow_type):
    """Full LMTD calculation path.

    Assumes: Q = UA * LMTD (correction factor F = 1 — extendable).
    Computes outlet temperatures iteratively.
    """
    Ch = mh * cph
    Cc = mc * cpc
    if Ch <= 0 or Cc <= 0:
        return None

    # First pass: estimate with e-NTU to get outlet T, then compute LMTD
    res_entu = compute_entu(Th_in, Tc_in, mh, cph, mc, cpc, UA, flow_type)
    if res_entu is None:
        return None

    Th_out = res_entu["Th_out"]
    Tc_out = res_entu["Tc_out"]
    Q      = res_entu["Q"]

    lmtd_res = compute_lmtd(Th_in, Th_out, Tc_in, Tc_out, lmtd_flow)
    lmtd_val = lmtd_res["LMTD"]

    # Effective UA from LMTD: UA_eff = Q / LMTD
    UA_eff = (Q / lmtd_val) if (lmtd_val and lmtd_val > 0) else None

    return dict(
        Ch=Ch, Cc=Cc, Q=Q,
        Th_out=Th_out, Tc_out=Tc_out,
        dT1=lmtd_res["dT1"], dT2=lmtd_res["dT2"],
        LMTD=lmtd_val, UA_eff=UA_eff,
        flow_type=flow_type, lmtd_flow=lmtd_flow,
        valid=lmtd_res["valid"],
        warning=lmtd_res["warning"],
        method="LMTD",
        # carry e-NTU fields for downstream use
        Cmin=res_entu["Cmin"], Cmax=res_entu["Cmax"],
        Cr=res_entu["Cr"], NTU=res_entu["NTU"],
        epsilon=res_entu["epsilon"], Qmax=res_entu["Qmax"],
    )


# ═══════════════════════════════════════════════════════════════
# ── MODULE 3 : METHODE STANTON ────────────────────────────────
# ═══════════════════════════════════════════════════════════════

def compute_stanton_direct(St, rho, u, cp):
    """Compute h from Stanton number directly.

    h = St * rho * u * cp   [W/m²/K]
    """
    h = St * rho * u * cp
    return dict(St=St, h=h, rho=rho, u=u, cp=cp, mode="direct")


def compute_stanton_from_dimensionless(Re, Pr, Nu, L_char):
    """Compute Stanton number from Re, Pr, Nu.

    St = Nu / (Re * Pr)
    h  = Nu * k / L  — requires thermal conductivity k.
    """
    if Re > 0 and Pr > 0:
        St = Nu / (Re * Pr)
    else:
        St = None
    return dict(Re=Re, Pr=Pr, Nu=Nu, St=St, L_char=L_char)


def stanton_full_result(stanton_data, A_exchange, Th_in, Tc_in, mh, cph, mc, cpc):
    """Use h from Stanton to compute UA, then run e-NTU.

    Assumption: UA ≈ h * A  (one-side dominant resistance).
    """
    h   = stanton_data.get("h")
    A   = A_exchange
    if h is None or A is None or A <= 0:
        return None, None
    UA = h * A   # [W/K]
    res = compute_entu(Th_in, Tc_in, mh, cph, mc, cpc, UA, "Contre-courant")
    if res:
        res["method"] = "Stanton"
        res["h_stanton"] = h
        res["UA_stanton"] = UA
    return UA, res


# ═══════════════════════════════════════════════════════════════
# ── MODULE 4 : NOMBRES ADIMENSIONNELS ─────────────────────────
# ═══════════════════════════════════════════════════════════════

def compute_dimensionless(rho=None, u=None, L=None, mu=None,
                          cp=None, k=None, h=None, ks=None, Lc=None,
                          alpha=None, t=None, Fo_L=None):
    """Compute dimensionless numbers with engineering interpretations.

    All SI units:
      rho [kg/m³], u [m/s], L [m], mu [Pa·s],
      cp [J/kg/K], k [W/m/K], h [W/m²/K], ks [W/m/K],
      Lc [m] (Biot char length), alpha [m²/s] (thermal diffusivity),
      t [s] (time for Fourier), Fo_L [m] (Fourier char length)
    """
    result = {}

    # Reynolds  Re = rho * u * L / mu  [-]
    if all(v is not None and v > 0 for v in [rho, u, L, mu]):
        Re = rho * u * L / mu
        result["Re"] = Re
        if Re < 2300:
            result["regime"] = "Laminaire (Re < 2300)"
        elif Re < 4000:
            result["regime"] = "Transitoire (2300 < Re < 4000)"
        else:
            result["regime"] = "Turbulent (Re > 4000)"

    # Prandtl  Pr = mu * cp / k  [-]
    if all(v is not None and v > 0 for v in [mu, cp, k]):
        result["Pr"] = mu * cp / k

    # Nusselt  Nu = h * L / k  [-]
    if all(v is not None and v > 0 for v in [h, L, k]):
        result["Nu"] = h * L / k

    # Stanton  St = h / (rho * u * cp)  [-]
    if all(v is not None and v > 0 for v in [h, rho, u, cp]):
        result["St"] = h / (rho * u * cp)

    # Peclet  Pe = Re * Pr  [-]
    if "Re" in result and "Pr" in result:
        result["Pe"] = result["Re"] * result["Pr"]

    # Biot  Bi = h * Lc / ks  [-]
    if all(v is not None and v > 0 for v in [h, Lc, ks]):
        result["Bi"] = h * Lc / ks

    # Fourier  Fo = alpha * t / L²  [-]
    if all(v is not None and v > 0 for v in [alpha, t, Fo_L]):
        result["Fo"] = alpha * t / (Fo_L ** 2)

    # Graetz  Gz = (D/L) * Re * Pr  [-]  (needs D = hydraulic diameter = L here)
    if "Re" in result and "Pr" in result and all(v is not None and v > 0 for v in [L]):
        result["Gz"] = (L / max(L, 1e-9)) * result["Re"] * result["Pr"]

    # Colburn j-factor  j = St * Pr^(2/3)  [-]
    if "St" in result and "Pr" in result:
        result["j"] = result["St"] * result["Pr"] ** (2.0 / 3.0)

    return result


# ═══════════════════════════════════════════════════════════════
# ── MODULE 5 : SURFACES D'ECHANGE ─────────────────────────────
# ═══════════════════════════════════════════════════════════════

SURFACE_TYPES = [
    "Tube lisse",
    "Tube ailette",
    "Faisceau tubulaire",
    "Plaques planes",
    "Plaques ondulees",
    "Serpentin",
    "Surface annulaire",
    "Surface personnalisee",
]

SURFACE_FORMULAS = {
    "Tube lisse":         "A = π · D · L",
    "Tube ailette":       "A = π·D·L + 2·N_f·(π/4)·(D_f²-D²)",
    "Faisceau tubulaire": "A = N_t · π · D · L",
    "Plaques planes":     "A = N_p · W · H · 2  (2 faces actives par plaque)",
    "Plaques ondulees":   "A = N_p · W · H · 2 · η_corr  (η_corr ≈ 1.15–1.25)",
    "Serpentin":          "A = N_spires · π · D · (π · D_s)",
    "Surface annulaire":  "A = 2π · (R₂² - R₁²)  (anneau)",
    "Surface personnalisee": "A = A_user  (entree manuelle)",
}


def compute_exchange_area(surface_type: str, geo: dict) -> dict:
    """Compute heat exchange area A [m²] for the selected surface type.

    Parameters in geo dict (all in SI):
      Tube lisse       : D [m], L [m]
      Tube ailette     : D [m], L [m], D_f [m], N_f [-] (fins/m), n_fins (total fins)
      Faisceau tub.    : D [m], L [m], N_t [-]
      Plaques planes   : N_p [-], W [m], H [m]
      Plaques ondulees : N_p [-], W [m], H [m], eta_corr [-]
      Serpentin        : D [m], D_s [m], N_turns [-]
      Surface annulaire: R1 [m], R2 [m]
      Personnalisee    : A_user [m²]

    Assumption notes are returned in the 'note' field.
    """
    s = surface_type
    try:
        if s == "Tube lisse":
            D, L = geo["D"], geo["L"]
            A = math.pi * D * L
            note = "Aire laterale cylindrique. Paroi d'epaisseur negligee."

        elif s == "Tube ailette":
            D, L = geo["D"], geo["L"]
            D_f  = geo.get("D_f", D * 3)
            n_fins = int(geo.get("n_fins", 10))
            # base tube area (between fins)
            A_base = math.pi * D * L
            # fin area: two flat annular discs per fin (approximation)
            A_fin_one = 2 * math.pi / 4 * (D_f ** 2 - D ** 2)
            A = A_base + n_fins * A_fin_one
            note = (
                "Approximation : ailettes planes annulaires. "
                "Efficacite ailette η_f non appliquee (A brute)."
            )

        elif s == "Faisceau tubulaire":
            D, L, N_t = geo["D"], geo["L"], int(geo["N_t"])
            A = N_t * math.pi * D * L
            note = f"Somme des aires laterales de {N_t} tubes. Pas de correction pour les chicanes."

        elif s == "Plaques planes":
            N_p, W, H = int(geo["N_p"]), geo["W"], geo["H"]
            A = N_p * W * H * 2
            note = "2 faces actives par plaque. Bordures negligees."

        elif s == "Plaques ondulees":
            N_p  = int(geo["N_p"])
            W, H = geo["W"], geo["H"]
            eta  = geo.get("eta_corr", 1.20)
            A = N_p * W * H * 2 * eta
            note = f"Facteur de correction ondulation η = {eta:.2f} (typique 1.15–1.25)."

        elif s == "Serpentin":
            D      = geo["D"]
            D_s    = geo["D_s"]
            N_t    = int(geo.get("N_turns", 10))
            # developed length per turn ≈ π * D_s (coil mean diameter)
            A = N_t * math.pi * D * (math.pi * D_s)
            note = "Developpement tubulaire helicoidale. Pas = D (fil touche fil)."

        elif s == "Surface annulaire":
            R1, R2 = geo["R1"], geo["R2"]
            A = 2 * math.pi * (R2 ** 2 - R1 ** 2)
            note = "Aire des deux faces d'un anneau plat."

        elif s == "Surface personnalisee":
            A = geo.get("A_user", 1.0)
            note = "Valeur saisie manuellement par l'utilisateur."

        else:
            A = 1.0
            note = "Surface inconnue — valeur par defaut."

    except (KeyError, ZeroDivisionError, ValueError) as err:
        return dict(A=None, formula=SURFACE_FORMULAS.get(s, ""), note=str(err), valid=False)

    return dict(A=A, formula=SURFACE_FORMULAS.get(s, ""), note=note, valid=True)


# ═══════════════════════════════════════════════════════════════
# ── MODULE 6 : VALIDATION INGENIEUR ───────────────────────────
# ═══════════════════════════════════════════════════════════════

def validate_inputs(Th_in, Tc_in, mh, mc, cph, cpc, UA, A_exchange):
    """Return list of (level, message) tuples — 'error', 'warning', 'ok'."""
    checks = []

    if Th_in <= Tc_in:
        checks.append(("error", f"The ({Th_in}°C) doit etre > Tce ({Tc_in}°C)."))
    else:
        checks.append(("ok", f"The > Tce : {Th_in}°C > {Tc_in}°C ✓"))

    for name, val in [("mh", mh), ("mc", mc), ("cph", cph), ("cpc", cpc)]:
        if val <= 0:
            checks.append(("error", f"{name} doit etre > 0 (valeur = {val})."))
        else:
            checks.append(("ok", f"{name} = {val:.4g} > 0 ✓"))

    if UA is not None:
        if UA <= 0:
            checks.append(("error", f"UA = {UA:.2f} W/K doit etre > 0."))
        else:
            checks.append(("ok", f"UA = {UA:.2f} W/K > 0 ✓"))

    if A_exchange is not None:
        if A_exchange <= 0:
            checks.append(("error", f"Surface A = {A_exchange:.4g} m² non physique."))
        else:
            checks.append(("ok", f"Surface A = {A_exchange:.4g} m² ✓"))

    return checks


def validate_thermal_outputs(res):
    """Validate computed outlet temperatures for physical consistency."""
    checks = []
    if res is None:
        return [("error", "Calcul impossible — donnees invalides.")]

    Th_in, Tc_in = res.get("Th_in_ref"), res.get("Tc_in_ref")
    Th_out, Tc_out = res["Th_out"], res["Tc_out"]

    if Th_out < Tc_in if Tc_in else False:
        checks.append(("error", f"Croisement : Tfe = {Th_out:.2f}°C < Tce = {Tc_in}°C"))
    if Tc_out > Th_in if Th_in else False:
        checks.append(("error", f"Croisement : Tcs = {Tc_out:.2f}°C > The = {Th_in}°C"))
    if not checks:
        checks.append(("ok", "Sorties thermiques coherentes."))

    eps = res.get("epsilon", 0)
    if eps > 1.001:
        checks.append(("warning", f"Efficacite e = {eps:.4f} > 1 (non physique)."))
    elif eps > 0.95:
        checks.append(("warning", f"Efficacite tres elevee : e = {eps:.4f}. Verifier le dimensionnement."))

    return checks


# ═══════════════════════════════════════════════════════════════
# ── MODULE 7 : TRACÉS 2D ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

def draw_schema(res, UA, Th_in, Tc_in):
    flow_type = res["flow_type"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")
    HOT, COLD = "#e74c3c", "#2980b9"
    BODY, BORDER, GREEN = "#ecf0f1", "#2c3e50", "#27ae60"

    ax.add_patch(FancyBboxPatch((2.5, 1.5), 5, 2, boxstyle="round,pad=0.15",
                                linewidth=2, edgecolor=BORDER, facecolor=BODY, zorder=2))
    for y in [2.0, 2.5, 3.0, 3.5]:
        ax.plot([2.65, 7.35], [y, y], color="#bdc3c7", lw=0.8, ls="--", zorder=3)

    kw_h = dict(arrowstyle="-|>", color=HOT,  lw=2.5)
    kw_c = dict(arrowstyle="-|>", color=COLD, lw=2.5)

    if flow_type in ("Contre-courant", "Calandre (1 passe) / Tubes (2 passes)"):
        ax.annotate("", xy=(2.5, 3.2), xytext=(0.3, 3.2), arrowprops=kw_h, zorder=5)
        ax.plot([0.3, 7.7], [3.2, 3.2], color=HOT,  lw=2.5, zorder=4)
        ax.annotate("", xy=(9.7, 3.2), xytext=(7.5, 3.2), arrowprops=kw_h, zorder=5)
        ax.annotate("", xy=(7.5, 2.0), xytext=(9.7, 2.0), arrowprops=kw_c, zorder=5)
        ax.plot([0.3, 9.7], [2.0, 2.0], color=COLD, lw=2.5, zorder=4)
        ax.annotate("", xy=(0.3, 2.0), xytext=(2.5, 2.0), arrowprops=kw_c, zorder=5)
        ax.text(0.1, 3.5, f"The\n{Th_in:.1f}°C",    ha="center", va="center", color=HOT,  fontsize=10, fontweight="bold")
        ax.text(9.9, 3.5, f"Tfe\n{res['Th_out']:.2f}°C", ha="center", va="center", color=HOT,  fontsize=10, fontweight="bold")
        ax.text(9.9, 1.6, f"Tce\n{Tc_in:.1f}°C",    ha="center", va="center", color=COLD, fontsize=10, fontweight="bold")
        ax.text(0.1, 1.6, f"Tcs\n{res['Tc_out']:.2f}°C", ha="center", va="center", color=COLD, fontsize=10, fontweight="bold")
    elif flow_type == "Co-courant":
        for y, col, in_T, out_T in [
            (3.2, HOT,  Th_in,  res["Th_out"]),
            (2.0, COLD, Tc_in,  res["Tc_out"]),
        ]:
            kw = dict(arrowstyle="-|>", color=col, lw=2.5)
            ax.annotate("", xy=(2.5, y), xytext=(0.3, y), arrowprops=kw, zorder=5)
            ax.plot([0.3, 7.7], [y, y], color=col, lw=2.5, zorder=4)
            ax.annotate("", xy=(9.7, y), xytext=(7.5, y), arrowprops=kw, zorder=5)
        ax.text(0.1, 3.55, f"The\n{Th_in:.1f}°C", ha="center", va="center", color=HOT,  fontsize=10, fontweight="bold")
        ax.text(9.9, 3.55, f"Tfe\n{res['Th_out']:.2f}°C", ha="center", va="center", color=HOT,  fontsize=10, fontweight="bold")
        ax.text(0.1, 1.55, f"Tce\n{Tc_in:.1f}°C", ha="center", va="center", color=COLD, fontsize=10, fontweight="bold")
        ax.text(9.9, 1.55, f"Tcs\n{res['Tc_out']:.2f}°C", ha="center", va="center", color=COLD, fontsize=10, fontweight="bold")
    else:  # croise
        ax.annotate("", xy=(2.5, 2.6), xytext=(0.3, 2.6), arrowprops=kw_h, zorder=5)
        ax.plot([0.3, 7.7], [2.6, 2.6], color=HOT, lw=2.5, zorder=4)
        ax.annotate("", xy=(9.7, 2.6), xytext=(7.5, 2.6), arrowprops=kw_h, zorder=5)
        ax.annotate("", xy=(5.0, 1.5), xytext=(5.0, 0.3), arrowprops=kw_c, zorder=5)
        ax.plot([5.0, 5.0], [0.3, 3.7], color=COLD, lw=2.5, zorder=4)
        ax.annotate("", xy=(5.0, 4.7), xytext=(5.0, 3.5), arrowprops=kw_c, zorder=5)
        ax.text(0.1, 3.0, f"The\n{Th_in:.1f}°C",    ha="center", va="center", color=HOT,  fontsize=10, fontweight="bold")
        ax.text(9.9, 3.0, f"Tfe\n{res['Th_out']:.2f}°C", ha="center", va="center", color=HOT,  fontsize=10, fontweight="bold")
        ax.text(5.0, 0.1, f"Tce  {Tc_in:.1f}°C",  ha="center", va="center", color=COLD, fontsize=10, fontweight="bold")
        ax.text(5.0, 4.9, f"Tcs  {res['Tc_out']:.2f}°C", ha="center", va="center", color=COLD, fontsize=10, fontweight="bold")

    ax.text(5.0, 4.6, flow_type, ha="center", va="center",
            fontsize=13, fontweight="bold", color=BORDER)
    method_tag = res.get("method", "e-NTU")
    ax.text(5.0, 0.3,
            f"Q = {res['Q']:,.1f} W  |  {method_tag}  |  UA = {UA:,.1f} W/K",
            ha="center", va="center", fontsize=9, color=GREEN, fontweight="bold")
    plt.tight_layout()
    return fig


def draw_temperature_profile(res, Th_in, Tc_in):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(0, 1, 200)
    Th_x = res["Th_out"] + (Th_in - res["Th_out"]) * (1 - x)
    if res["flow_type"] == "Co-courant":
        Tc_x  = Tc_in + (res["Tc_out"] - Tc_in) * x
        clabel = "Fluide froid (->)"
    else:
        Tc_x  = Tc_in + (res["Tc_out"] - Tc_in) * (1 - x)
        clabel = "Fluide froid (<-)"
    ax.plot(x, Th_x, color="#e74c3c", lw=2.5, label="Fluide chaud")
    ax.plot(x, Tc_x, color="#2980b9", lw=2.5, label=clabel,
            linestyle="--" if res["flow_type"] != "Co-courant" else "-")
    ax.fill_between(x, Tc_x, Th_x, alpha=0.08, color="#e74c3c")

    # LMTD delta annotations
    method = res.get("method", "e-NTU")
    if method == "LMTD" and res.get("LMTD") is not None:
        ax.axhline(y=Th_in - res["dT1"],  color="grey", lw=0.8, ls=":")
        ax.text(0.02, (Th_x[0] + Tc_x[-1]) / 2,
                f"ΔT₁={res['dT1']:.1f}°C", fontsize=9, color="grey")
        ax.text(0.85, (Th_x[-1] + Tc_x[0]) / 2,
                f"ΔT₂={res['dT2']:.1f}°C  LMTD={res['LMTD']:.2f}°C",
                fontsize=9, color="#8e44ad")

    ax.set_xlabel("Position relative (0→1)", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_title(f"Profil de temperatures — {res['flow_type']}  [{method}]",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, ls=":", alpha=0.6)
    ax.set_xlim(-0.02, 1.02)
    plt.tight_layout()
    return fig


def draw_eps_vs_NTU(res):
    Cr = res["Cr"]
    NTU_range = np.linspace(0, 8, 400)
    fig, ax = plt.subplots(figsize=(8, 4))
    styles = {
        "Contre-courant":     (epsilon_counter,            "#8e44ad", "-"),
        "Co-courant":         (epsilon_parallel,           "#e67e22", "--"),
        "Croise non melange": (epsilon_crossflow_unmixed,  "#27ae60", "-."),
        "Calandre (1P/2P)":   (epsilon_shell_tube,         "#2980b9", ":"),
    }
    for lbl, (fn, col, ls) in styles.items():
        ax.plot(NTU_range, [fn(n, Cr) for n in NTU_range], color=col, lw=2, ls=ls, label=lbl)
    ax.plot(res["NTU"], res["epsilon"], "ko", ms=9, zorder=5,
            label=f"Point de fonctionnement (ε={res['epsilon']:.3f})")
    ax.set_xlabel("NTU"); ax.set_ylabel("ε (efficacite)")
    ax.set_title(f"ε–NTU  pour  R = {Cr:.3f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, ls=":", alpha=0.6)
    ax.set_xlim(0, 8); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig


def draw_lmtd_diagram(res, Th_in, Tc_in):
    """Bar chart showing ΔT1, ΔT2, LMTD."""
    if not res.get("LMTD"):
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["ΔT₁", "ΔT₂", "LMTD"]
    vals   = [res["dT1"], res["dT2"], res["LMTD"]]
    cols   = ["#e74c3c", "#2980b9", "#8e44ad"]
    bars = ax.bar(labels, vals, color=cols, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                f"{v:.2f} °C", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Difference de temperature (°C)")
    ax.set_title(f"Diagramme LMTD — {res.get('lmtd_flow','counter')} flow", fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.5)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# ── MODULE 8 : VISUALISATION 3D ÉCHANGEUR ─────────────────────
# ═══════════════════════════════════════════════════════════════

def _line3(x, y, z, col, w=6, name=""):
    return go.Scatter3d(x=x, y=y, z=z, mode="lines",
                        line=dict(color=col, width=w), name=name, hoverinfo="name")


def _cone(x, y, z, u, v, w, col, name=""):
    return go.Cone(x=[x], y=[y], z=[z], u=[u], v=[v], w=[w],
                   colorscale=[[0, col], [1, col]], showscale=False,
                   sizemode="absolute", sizeref=0.18, anchor="tail",
                   name=name, showlegend=False)


def _label3(x, y, z, txt, col):
    return go.Scatter3d(x=[x], y=[y], z=[z], mode="text", text=[txt],
                        textfont=dict(color=col, size=13, family="Arial Black"),
                        hoverinfo="skip", showlegend=False)


def _grad_segs(xs, ys, zs, Tv, Tlo, Thi):
    out = []
    for i in range(len(xs) - 1):
        n = np.clip((Tv[i] - Tlo) / max(Thi - Tlo, 1), 0, 1)
        r = int(min(255, n * 2 * 255))
        b = int(min(255, (1 - n) * 2 * 255))
        g = int(255 - abs(n - 0.5) * 2 * 255)
        out.append(go.Scatter3d(
            x=[xs[i], xs[i+1]], y=[ys[i], ys[i+1]], z=[zs[i], zs[i+1]],
            mode="lines", line=dict(color=f"rgb({r},{g},{b})", width=5),
            showlegend=False, hoverinfo="skip",
        ))
    return out


def build_3d_counterflow(res, Th_in, Tc_in, params):
    L  = params.get("length", 3.0)
    Ro = params.get("shell_radius", 0.5)
    nt = params.get("n_tubes", 6)
    Tlo = min(Tc_in, res["Tc_out"])
    Thi = max(Th_in, res["Th_out"])
    ft  = res["flow_type"]
    tr  = []

    theta = np.linspace(0, 2*np.pi, 60)
    xring = np.linspace(0, L, 2)
    TG, XG = np.meshgrid(theta, xring)
    tr.append(go.Surface(x=XG, y=Ro*np.cos(TG), z=Ro*np.sin(TG),
                         colorscale=[[0,"rgba(100,160,220,0.12)"],[1,"rgba(100,160,220,0.12)"]],
                         showscale=False, opacity=0.18, name="Shell", hoverinfo="name", showlegend=True))
    for xp in [0, L]:
        th2 = np.linspace(0, 2*np.pi, 60); r2 = np.linspace(0, Ro, 5)
        TH, RR = np.meshgrid(th2, r2)
        tr.append(go.Surface(x=np.full_like(RR, xp), y=RR*np.cos(TH), z=RR*np.sin(TH),
                             colorscale=[[0,"rgba(80,80,120,0.35)"],[1,"rgba(80,80,120,0.35)"]],
                             showscale=False, opacity=0.35, showlegend=False, hoverinfo="skip"))

    cols = max(1, nt // 2)
    yp = np.linspace(-0.25, 0.25, cols)
    zp = [-0.12, 0.12] if nt > 1 else [0.0]
    xs = np.linspace(0, L, 30); tf = xs / L
    for z0 in zp:
        for y0 in yp:
            Th_x = Th_in + (res["Th_out"] - Th_in) * tf
            Tc_x = (res["Tc_out"] + (Tc_in - res["Tc_out"]) * tf) if ft == "Contre-courant" else (Tc_in + (res["Tc_out"] - Tc_in) * tf)
            tr += _grad_segs(xs, np.full(30, y0), np.full(30, z0), (Th_x+Tc_x)/2, Tlo, Thi)

    tr += [_line3([-0.6,0],[0,0],[0,0],"#e74c3c",8,"Chaud entree"),
           _cone(0,0,0,0.5,0,0,"#e74c3c"),
           _line3([L,L+0.6],[0,0],[0,0],"#c0392b",8,"Chaud sortie"),
           _cone(L,0,0,0.5,0,0,"#c0392b")]
    if ft == "Contre-courant":
        tr += [_line3([L+0.6,L],[0,0],[0.55,0.55],"#2980b9",8,"Froid entree"),
               _cone(L,0,0.55,-0.5,0,0,"#2980b9"),
               _line3([0,-0.6],[0,0],[0.55,0.55],"#1a6fa0",8,"Froid sortie"),
               _label3(L+0.1,0,0.65,f"Tce={Tc_in:.1f}°C","#2980b9"),
               _label3(-0.7,0,0.65,f"Tcs={res['Tc_out']:.2f}°C","#1a6fa0")]
    else:
        tr += [_line3([-0.6,0],[0,0],[0.55,0.55],"#2980b9",8,"Froid entree"),
               _cone(0,0,0.55,0.5,0,0,"#2980b9"),
               _line3([L,L+0.6],[0,0],[0.55,0.55],"#1a6fa0",8,"Froid sortie"),
               _label3(-0.7,0,0.65,f"Tce={Tc_in:.1f}°C","#2980b9"),
               _label3(L+0.1,0,0.65,f"Tcs={res['Tc_out']:.2f}°C","#1a6fa0")]
    tr += [_label3(-0.7,0,0,f"The={Th_in:.1f}°C","#e74c3c"),
           _label3(L+0.1,0,0,f"Tfe={res['Th_out']:.2f}°C","#c0392b")]

    fig = go.Figure(data=tr)
    fig.update_layout(
        title=dict(text=f"Visualisation 3D — {ft}<br><sup>Q={res['Q']:,.0f} W | ε={res['epsilon']:.4f} | NTU={res['NTU']:.4f}</sup>", x=0.5, font=dict(size=15)),
        scene=dict(xaxis_title="Longueur (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                   bgcolor="rgba(240,244,250,1)", camera=dict(eye=dict(x=1.6,y=1.1,z=0.9)), aspectmode="data"),
        legend=dict(x=0.01,y=0.99,bgcolor="rgba(255,255,255,0.8)",bordercolor="#aaa",borderwidth=1),
        margin=dict(l=0,r=0,t=80,b=0), height=600, paper_bgcolor="rgba(245,248,252,1)")
    return fig


def build_3d_crossflow(res, Th_in, Tc_in, params):
    L  = params.get("length", 2.5)
    W  = params.get("width",  2.5)
    H  = params.get("height", 1.2)
    nc = params.get("n_channels", 7)
    Tlo = min(Tc_in, res["Tc_out"])
    Thi = max(Th_in, res["Th_out"])
    tr  = []
    ch  = H / (2*nc+1)
    for i in range(nc):
        zh = (2*i+1)*ch; zc = (2*i+2)*ch; zp = zc - ch/2
        tr.append(go.Surface(x=[[0,L],[0,L]], y=[[0,0],[W,W]], z=[[zp,zp],[zp,zp]],
                             colorscale=[[0,"rgba(80,90,110,0.55)"],[1,"rgba(80,90,110,0.55)"]],
                             showscale=False, opacity=0.55, showlegend=False, hoverinfo="skip"))
        tf = np.linspace(0,1,20)
        tr += _grad_segs(tf*L, np.full(20,W/2), np.full(20,zh),
                         Th_in+(res["Th_out"]-Th_in)*tf, Tlo, Thi)
        tr += _grad_segs(np.full(20,L/2), tf*W, np.full(20,zc),
                         Tc_in+(res["Tc_out"]-Tc_in)*tf, Tlo, Thi)
    bx=[0,L,L,0,0,0,L,L,0,0,L,L,0,0,L,L]
    by=[0,0,W,W,0,0,0,W,W,0,0,0,0,0,W,W]
    bz=[0,0,0,0,0,H,H,H,H,H,0,H,0,H,0,H]
    tr.append(go.Scatter3d(x=bx,y=by,z=bz,mode="lines",line=dict(color="#2c3e50",width=3),name="Boitier"))
    tr += [_cone(-0.3,W/2,H*0.25,0.5,0,0,"#e74c3c","Chaud →"),
           _cone(L/2,-0.3,H*0.75,0,0.5,0,"#2980b9","Froid →"),
           _label3(-0.5,W/2,H*0.25,f"The={Th_in:.1f}°C","#e74c3c"),
           _label3(L+0.1,W/2,H*0.25,f"Tfe={res['Th_out']:.2f}°C","#c0392b"),
           _label3(L/2,-0.5,H*0.75,f"Tce={Tc_in:.1f}°C","#2980b9"),
           _label3(L/2,W+0.1,H*0.75,f"Tcs={res['Tc_out']:.2f}°C","#1a6fa0")]
    fig = go.Figure(data=tr)
    fig.update_layout(
        title=dict(text=f"Visualisation 3D — {res['flow_type']}<br><sup>Q={res['Q']:,.0f} W | ε={res['epsilon']:.4f}</sup>",x=0.5,font=dict(size=15)),
        scene=dict(xaxis_title="X-Chaud (m)",yaxis_title="Y-Froid (m)",zaxis_title="Z (m)",
                   bgcolor="rgba(240,244,250,1)",camera=dict(eye=dict(x=1.8,y=-1.5,z=1.2)),aspectmode="cube"),
        margin=dict(l=0,r=0,t=80,b=0),height=600,paper_bgcolor="rgba(245,248,252,1)")
    return fig


def build_3d_shell_tube(res, Th_in, Tc_in, params):
    L  = params.get("length", 3.0)
    Ro = params.get("shell_radius", 0.55)
    nt = params.get("n_tubes", 8)
    Tlo = min(Tc_in, res["Tc_out"])
    Thi = max(Th_in, res["Th_out"])
    tr  = []
    theta = np.linspace(0,2*np.pi,60); xring = np.linspace(0,L,2)
    TG,XG = np.meshgrid(theta,xring)
    tr.append(go.Surface(x=XG,y=Ro*np.cos(TG),z=Ro*np.sin(TG),
                         colorscale=[[0,"rgba(180,200,230,0.13)"],[1,"rgba(180,200,230,0.13)"]],
                         showscale=False,opacity=0.18,name="Calandre",showlegend=True))
    tb = np.linspace(0,np.pi,40)
    tr.append(go.Scatter3d(x=np.full(40,L/2),y=Ro*np.cos(tb),z=Ro*np.sin(tb),
                           mode="lines",line=dict(color="rgba(80,80,80,0.6)",width=4),name="Chicane"))
    cols2 = max(1,nt//2)
    yp = np.linspace(-Ro*0.55,Ro*0.55,cols2)
    xs1 = np.linspace(0,L,30); xs2 = np.linspace(L,0,30)
    t1  = np.linspace(0,1,30); t2  = np.linspace(1,0,30)
    Thm = (Th_in+res["Th_out"])/2
    for y0 in yp:
        tr += _grad_segs(xs1,np.full(30,y0),np.full(30,-0.12), Th_in+(Thm-Th_in)*t1, Tlo,Thi)
        tr += _grad_segs(xs2,np.full(30,y0),np.full(30, 0.12), Thm+(res["Th_out"]-Thm)*t2, Tlo,Thi)
    tr.append(_line3([0,L/2,L/2,L,L],[0,0,0,0,0],[Ro*0.65,Ro*0.65,-Ro*0.65,-Ro*0.65,-Ro*0.65],"#2980b9",7,"Froid calandre"))
    tr += [_label3(-0.5,0,0,f"The={Th_in:.1f}°C","#e74c3c"),
           _label3(L+0.2,0,0,f"Tfe={res['Th_out']:.2f}°C","#c0392b"),
           _label3(0,0,Ro*0.85,f"Tce={Tc_in:.1f}°C","#2980b9"),
           _label3(L,0,-Ro*0.85,f"Tcs={res['Tc_out']:.2f}°C","#1a6fa0")]
    fig = go.Figure(data=tr)
    fig.update_layout(
        title=dict(text=f"Visualisation 3D — Calandre 1P/Tubes 2P<br><sup>Q={res['Q']:,.0f} W | ε={res['epsilon']:.4f}</sup>",x=0.5,font=dict(size=15)),
        scene=dict(xaxis_title="Longueur (m)",yaxis_title="Y (m)",zaxis_title="Z (m)",
                   bgcolor="rgba(240,244,250,1)",camera=dict(eye=dict(x=1.8,y=1.2,z=1.0)),aspectmode="data"),
        margin=dict(l=0,r=0,t=80,b=0),height=620,paper_bgcolor="rgba(245,248,252,1)")
    return fig


def build_3d_visualization(res, Th_in, Tc_in, params):
    ft = res["flow_type"]
    if ft in ("Contre-courant","Co-courant"):
        return build_3d_counterflow(res, Th_in, Tc_in, params)
    elif ft in ("Croise non melange","Croise Cmin melange"):
        return build_3d_crossflow(res, Th_in, Tc_in, params)
    elif ft == "Calandre (1 passe) / Tubes (2 passes)":
        return build_3d_shell_tube(res, Th_in, Tc_in, params)
    return build_3d_counterflow(res, Th_in, Tc_in, params)


# ═══════════════════════════════════════════════════════════════
# ── MODULE 9 : VISUALISATION 3D SURFACE D'ÉCHANGE ─────────────
# ═══════════════════════════════════════════════════════════════

def build_surface_3d(surface_type: str, geo: dict, res=None) -> go.Figure:
    """Build a dedicated 3D Plotly figure for the selected exchange surface.

    Color: highlighted exchange surface in orange/gold, structure in grey.
    """
    SURF_COL  = "#f39c12"
    STRUCT_COL = "#7f8c8d"

    def _layout(title_txt, fig):
        fig.update_layout(
            title=dict(text=f"Visualisation 3D — Surface d'échange<br><sup>{title_txt}</sup>",
                       x=0.5, font=dict(size=14)),
            scene=dict(bgcolor="rgba(240,244,250,1)",
                       xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                       camera=dict(eye=dict(x=1.6, y=1.2, z=1.0)),
                       aspectmode="data"),
            margin=dict(l=0, r=0, t=80, b=0), height=500,
            paper_bgcolor="rgba(245,248,252,1)",
        )
        return fig

    tr = []

    # ── Tube lisse ──────────────────────────────────────────────
    if surface_type == "Tube lisse":
        D = geo.get("D", 0.05); L = geo.get("L", 1.0)
        R = D / 2
        theta = np.linspace(0, 2*np.pi, 60)
        xv    = np.linspace(0, L, 40)
        TH, XV = np.meshgrid(theta, xv)
        tr.append(go.Surface(x=XV, y=R*np.cos(TH), z=R*np.sin(TH),
                             colorscale=[[0,SURF_COL],[1,"#e67e22"]],
                             showscale=False, opacity=0.9,
                             name="Surface tube lisse", showlegend=True))
        return _layout(f"Tube lisse : D={D*100:.1f} cm, L={L:.2f} m", go.Figure(data=tr))

    # ── Tube ailetté ────────────────────────────────────────────
    elif surface_type == "Tube ailette":
        D    = geo.get("D", 0.03); L = geo.get("L", 0.5)
        D_f  = geo.get("D_f", D*3)
        n_fins = int(geo.get("n_fins", 8))
        R, Rf = D/2, D_f/2
        # base tube
        theta = np.linspace(0, 2*np.pi, 40)
        xv    = np.linspace(0, L, 40)
        TH, XV = np.meshgrid(theta, xv)
        tr.append(go.Surface(x=XV, y=R*np.cos(TH), z=R*np.sin(TH),
                             colorscale=[[0,STRUCT_COL],[1,STRUCT_COL]],
                             showscale=False, opacity=0.6, name="Tube base", showlegend=True))
        # fins: flat annular discs
        r_fin = np.linspace(R, Rf, 20)
        for i in range(n_fins):
            x_fin = (i + 0.5) * L / n_fins
            TH2, RR2 = np.meshgrid(theta, r_fin)
            X_fin = np.full_like(RR2, x_fin)
            tr.append(go.Surface(x=X_fin, y=RR2*np.cos(TH2), z=RR2*np.sin(TH2),
                                 colorscale=[[0,SURF_COL],[1,"#e67e22"]],
                                 showscale=False, opacity=0.85,
                                 showlegend=(i == 0), name="Ailette"))
        return _layout(f"Tube ailette : D={D*100:.1f} cm, Df={D_f*100:.1f} cm, {n_fins} ailettes",
                       go.Figure(data=tr))

    # ── Faisceau tubulaire ──────────────────────────────────────
    elif surface_type == "Faisceau tubulaire":
        D   = geo.get("D", 0.025); L = geo.get("L", 1.0); N_t = int(geo.get("N_t", 9))
        R   = D / 2
        theta = np.linspace(0, 2*np.pi, 40)
        xv    = np.linspace(0, L, 30)
        TH, XV = np.meshgrid(theta, xv)
        cols_t = math.ceil(math.sqrt(N_t))
        rows_t = math.ceil(N_t / cols_t)
        pitch  = D * 2.5
        k = 0
        for row in range(rows_t):
            for col in range(cols_t):
                if k >= N_t:
                    break
                y0 = (col - cols_t/2 + 0.5) * pitch
                z0 = (row - rows_t/2 + 0.5) * pitch
                tr.append(go.Surface(x=XV, y=y0+R*np.cos(TH), z=z0+R*np.sin(TH),
                                     colorscale=[[0,SURF_COL],[1,"#e67e22"]],
                                     showscale=False, opacity=0.85,
                                     showlegend=(k==0), name=f"Tube {k+1}"))
                k += 1
        return _layout(f"Faisceau : {N_t} tubes, D={D*100:.1f} cm, L={L:.2f} m",
                       go.Figure(data=tr))

    # ── Plaques planes / ondulées ───────────────────────────────
    elif surface_type in ("Plaques planes", "Plaques ondulees"):
        N_p = int(geo.get("N_p", 5)); W = geo.get("W", 0.4); H = geo.get("H", 0.3)
        gap = H * 0.5
        for i in range(N_p):
            z0 = i * (H + gap)
            # plate surface
            xp = [[0,W],[0,W]]; yp = [[0,0],[H,H]]; zp = [[z0,z0],[z0,z0]]
            tr.append(go.Surface(x=xp, y=yp, z=zp,
                                 colorscale=[[0,SURF_COL],[1,"#e67e22"]],
                                 showscale=False, opacity=0.85,
                                 showlegend=(i==0), name="Plaque active"))
            if surface_type == "Plaques ondulees":
                # overlay a wavy pattern
                xw = np.linspace(0, W, 30)
                yw = np.linspace(0, H, 20)
                XW, YW = np.meshgrid(xw, yw)
                amp = H * 0.04
                ZW  = z0 + amp * np.sin(XW / W * 4 * np.pi)
                tr.append(go.Surface(x=XW, y=YW, z=ZW,
                                     colorscale=[[0,"#f0b429"],[1,"#d35400"]],
                                     showscale=False, opacity=0.5, showlegend=False))
        return _layout(f"{'Plaques ondulees' if 'ondulees' in surface_type else 'Plaques planes'} : {N_p} plaques {W*100:.0f}×{H*100:.0f} cm",
                       go.Figure(data=tr))

    # ── Serpentin ───────────────────────────────────────────────
    elif surface_type == "Serpentin":
        D      = geo.get("D", 0.025)
        D_s    = geo.get("D_s", 0.3)
        N_t    = int(geo.get("N_turns", 8))
        R_tube = D / 2
        R_coil = D_s / 2
        t_coil = np.linspace(0, N_t * 2 * np.pi, 400)
        pitch  = D * 2.5
        cx = R_coil * np.cos(t_coil)
        cy = R_coil * np.sin(t_coil)
        cz = pitch * t_coil / (2 * np.pi)
        # tube centerline as thick line
        tr.append(go.Scatter3d(x=cx, y=cy, z=cz, mode="lines",
                               line=dict(color=SURF_COL, width=8),
                               name="Centerline serpentin", showlegend=True))
        # tube surface (cylinder swept along helix — approx with ribbon)
        theta_t = np.linspace(0, 2*np.pi, 20)
        for j in range(0, len(t_coil)-1, 5):
            tx = np.array([-np.sin(t_coil[j]),  np.cos(t_coil[j]), 0])
            tz = np.array([0, 0, 1])
            c_pt = np.array([cx[j], cy[j], cz[j]])
            ring_x = c_pt[0] + R_tube * (np.cos(theta_t)*tx[0])
            ring_y = c_pt[1] + R_tube * (np.cos(theta_t)*tx[1])
            ring_z = c_pt[2] + R_tube * np.sin(theta_t)
            tr.append(go.Scatter3d(x=ring_x, y=ring_y, z=ring_z,
                                   mode="lines", line=dict(color="#e67e22", width=2),
                                   showlegend=False, hoverinfo="skip"))
        return _layout(f"Serpentin : {N_t} spires, D={D*100:.1f} cm, Ds={D_s*100:.0f} cm",
                       go.Figure(data=tr))

    # ── Surface annulaire ───────────────────────────────────────
    elif surface_type == "Surface annulaire":
        R1 = geo.get("R1", 0.05); R2 = geo.get("R2", 0.15)
        theta = np.linspace(0, 2*np.pi, 60)
        r_ann = np.linspace(R1, R2, 30)
        TH, RR = np.meshgrid(theta, r_ann)
        # front face
        tr.append(go.Surface(x=RR*np.cos(TH), y=RR*np.sin(TH), z=np.zeros_like(RR),
                             colorscale=[[0,SURF_COL],[1,"#e67e22"]],
                             showscale=False, opacity=0.9, name="Face avant", showlegend=True))
        # back face
        tr.append(go.Surface(x=RR*np.cos(TH), y=RR*np.sin(TH), z=np.full_like(RR,-0.01),
                             colorscale=[[0,"#e67e22"],[1,"#d35400"]],
                             showscale=False, opacity=0.7, name="Face arriere", showlegend=True))
        return _layout(f"Surface annulaire : R1={R1*100:.1f} cm, R2={R2*100:.1f} cm",
                       go.Figure(data=tr))

    # ── Personnalisée ───────────────────────────────────────────
    else:
        A = geo.get("A_user", 1.0)
        side = math.sqrt(A / 2)
        tr.append(go.Surface(x=[[0,side],[0,side]], y=[[0,0],[side,side]], z=[[0,0],[0,0]],
                             colorscale=[[0,SURF_COL],[1,"#e67e22"]],
                             showscale=False, opacity=0.9,
                             name=f"Surface A={A:.2f} m²", showlegend=True))
        return _layout(f"Surface personnalisee : A={A:.2f} m²  (représentation symbolique)",
                       go.Figure(data=tr))


# ═══════════════════════════════════════════════════════════════
# ── MODULE 10 : RAPPORT PDF ───────────────────────────────────
# ═══════════════════════════════════════════════════════════════

def _rl_table(data, col_widths, header_color):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), header_color),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("ALIGN",         (1,0),(-1,-1), "CENTER"),
    ]))
    return t


def fig_to_buf(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


def generate_pdf(
    res, UA, Th_in, Tc_in, mh, mc, cph, cpc,
    fig_schema, fig_profile, fig_method,
    surface_type, area_result, dim_numbers,
    validation_checks, method_name, lmtd_res=None,
    stanton_data=None,
    fig_lmtd=None,
):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    s_title   = ParagraphStyle("T",  parent=styles["Title"],   fontSize=18, spaceAfter=4, textColor=colors.HexColor("#2c3e50"))
    s_h1      = ParagraphStyle("H1", parent=styles["Heading1"],fontSize=12, spaceBefore=12, spaceAfter=3, textColor=colors.HexColor("#2980b9"))
    s_body    = ParagraphStyle("B",  parent=styles["Normal"],  fontSize=9,  leading=13)
    s_caption = ParagraphStyle("C",  parent=styles["Normal"],  fontSize=8,  leading=10, textColor=colors.grey, alignment=TA_CENTER)

    now   = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    story = []

    # ── Cover ──
    story += [
        Paragraph("Rapport — Echangeur de Chaleur", s_title),
        Paragraph(f"<font color='grey'>Méthode : {method_name} | {now}</font>", s_body),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2980b9"), spaceAfter=8),
        Spacer(1, 0.2*cm),
    ]

    # ── 1. Configuration ──
    story.append(Paragraph("1. Configuration", s_h1))
    cfg = [["Parametre","Valeur"],
           ["Type d'echangeur", res["flow_type"]],
           ["Methode de calcul", method_name],
           ["Surface d'echange", surface_type]]
    story.append(_rl_table(cfg, [9*cm, 8*cm], colors.HexColor("#2c3e50")))
    story.append(Spacer(1, 0.3*cm))

    # ── 2. Données entrée ──
    story.append(Paragraph("2. Données d'entrée", s_h1))
    inp = [["Parametre","Fluide chaud","Fluide froid"],
           ["Temperature entree (°C)",f"{Th_in:.2f}",f"{Tc_in:.2f}"],
           ["Debit massique (kg/s)",f"{mh:.4f}",f"{mc:.4f}"],
           ["Chaleur specifique (J/kg·K)",f"{cph:.2f}",f"{cpc:.2f}"],
           ["Capacite C = m·cp (W/K)",f"{res['Ch']:.2f}",f"{res['Cc']:.2f}"]]
    story.append(_rl_table(inp, [7*cm,5*cm,5*cm], colors.HexColor("#2c3e50")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(f"<b>UA :</b> {UA:,.2f} W/K", s_body))
    story.append(Spacer(1, 0.3*cm))

    # ── 3. Surface et géométrie ──
    story.append(Paragraph("3. Surface d'échange et géométrie", s_h1))
    A_val = area_result.get("A")
    A_str = f"{A_val:.4f} m²" if A_val else "N/A"
    surf_data = [["Grandeur","Valeur"],
                 ["Type de surface", surface_type],
                 ["Formule", area_result.get("formula","")],
                 ["Surface calculée A", A_str],
                 ["Note / hypothese", area_result.get("note","")]]
    story.append(_rl_table(surf_data, [6*cm, 11*cm], colors.HexColor("#16a085")))
    story.append(Spacer(1, 0.3*cm))

    # ── 4. Résultats thermiques ──
    story.append(Paragraph("4. Résultats thermiques", s_h1))
    th = [["Grandeur","Valeur","Unité"],
          ["Flux Q (Phi)",f"{res['Q']:,.2f}","W"],
          ["Temperature sortie chaud Tfe",f"{res['Th_out']:.2f}","°C"],
          ["Temperature sortie froid Tcs",f"{res['Tc_out']:.2f}","°C"],
          ["Efficacite epsilon",f"{res['epsilon']:.4f}","-"],
          ["NTU",f"{res['NTU']:.4f}","-"],
          ["UA",f"{UA:,.2f}","W/K"]]
    story.append(_rl_table(th, [8*cm,5*cm,4*cm], colors.HexColor("#27ae60")))
    story.append(Spacer(1, 0.2*cm))

    if method_name == "LMTD" and lmtd_res and lmtd_res.get("LMTD"):
        lm = [["LMTD","ΔT₁","ΔT₂"],
              [f"{lmtd_res['LMTD']:.3f} °C",f"{lmtd_res['dT1']:.3f} °C",f"{lmtd_res['dT2']:.3f} °C"]]
        story.append(_rl_table(lm, [6*cm,5.5*cm,5.5*cm], colors.HexColor("#8e44ad")))
    elif method_name == "Stanton" and stanton_data:
        sd = [["Parametre Stanton","Valeur"],
              ["Nombre de Stanton St", f"{stanton_data.get('St','N/A'):.6g}"],
              ["Coefficient h", f"{stanton_data.get('h','N/A'):.4g} W/m²/K"],
              ["UA = h·A", f"{UA:.2f} W/K"]]
        story.append(_rl_table(sd, [8*cm,9*cm], colors.HexColor("#8e44ad")))
    story.append(Spacer(1, 0.3*cm))

    # ── 5. Nombres adimensionnels ──
    if dim_numbers:
        story.append(Paragraph("5. Nombres adimensionnels", s_h1))
        dim_rows = [["Nombre","Valeur","Formule"]]
        labels = {
            "Re": ("Reynolds",  "ρ·u·L/μ"),
            "Pr": ("Prandtl",   "μ·cp/k"),
            "Nu": ("Nusselt",   "h·L/k"),
            "St": ("Stanton",   "h/(ρ·u·cp)"),
            "Pe": ("Peclet",    "Re·Pr"),
            "Bi": ("Biot",      "h·Lc/ks"),
            "Fo": ("Fourier",   "α·t/L²"),
            "Gz": ("Graetz",    "(D/L)·Re·Pr"),
            "j":  ("Colburn j", "St·Pr^(2/3)"),
        }
        for key, (name, form) in labels.items():
            if key in dim_numbers:
                dim_rows.append([name, f"{dim_numbers[key]:.5g}", form])
        if "regime" in dim_numbers:
            dim_rows.append(["Regime d'ecoulement", dim_numbers["regime"], "f(Re)"])
        story.append(_rl_table(dim_rows, [5*cm,5*cm,7*cm], colors.HexColor("#2c3e50")))
        story.append(Spacer(1, 0.3*cm))

    # ── 6. Validation ──
    story.append(Paragraph("6. Vérifications ingénieur", s_h1))
    for level, msg in validation_checks:
        col = "green" if level == "ok" else ("red" if level == "error" else "orange")
        story.append(Paragraph(f"<font color='{col}'>• {msg}</font>", s_body))
    story.append(Spacer(1, 0.2*cm))

    # ── Figures ──
    story.append(PageBreak())
    story.append(Paragraph("7. Schéma et profil de températures", s_h1))
    story.append(RLImage(fig_to_buf(fig_schema), width=16*cm, height=7.5*cm))
    story.append(Paragraph(f"Figure 1 — Schéma {res['flow_type']}.", s_caption))
    story.append(Spacer(1, 0.4*cm))
    story.append(RLImage(fig_to_buf(fig_profile), width=15*cm, height=7*cm))
    story.append(Paragraph("Figure 2 — Profil de températures.", s_caption))

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("8. Courbes méthode", s_h1))
    story.append(RLImage(fig_to_buf(fig_method), width=15*cm, height=7*cm))
    story.append(Paragraph("Figure 3 — Courbes ε-NTU ou LMTD.", s_caption))

    if fig_lmtd is not None:
        story.append(Spacer(1, 0.3*cm))
        story.append(RLImage(fig_to_buf(fig_lmtd), width=10*cm, height=7*cm))
        story.append(Paragraph("Figure 4 — Diagramme LMTD.", s_caption))

    # ── Conclusion ──
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("9. Interprétation ingénieur", s_h1))
    eps = res["epsilon"]
    NTU = res["NTU"]
    if eps > 0.85:
        interpret = f"L'echangeur est tres efficace (ε = {eps:.3f}). Le NTU = {NTU:.2f} confirme un bon dimensionnement."
    elif eps > 0.5:
        interpret = f"L'echangeur est correctement dimensionne (ε = {eps:.3f}, NTU = {NTU:.2f})."
    else:
        interpret = f"L'echangeur est sous-dimensionne (ε = {eps:.3f}, NTU = {NTU:.2f}). Augmenter UA ou la surface."
    story.append(Paragraph(interpret, s_body))

    story += [
        Spacer(1, 0.8*cm),
        HRFlowable(width="100%", thickness=1, color=colors.grey),
        Paragraph(f"Rapport généré par l'application Echangeurs Avancés — {now}", s_caption),
    ]

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════
# ── MODULE 11 : PAGE THEORIE ──────────────────────────────────
# ═══════════════════════════════════════════════════════════════

def show_theory_page():
    st.markdown("""
    <style>
    .step-card{background:#0f1623;border:1px solid #1e2d45;border-left:3px solid #f6ad55;
               border-radius:6px;padding:20px 24px;margin-bottom:22px;font-family:sans-serif;}
    .step-label{font-size:9px;font-weight:700;letter-spacing:3px;color:#f6ad55;text-transform:uppercase;margin-bottom:4px;}
    .step-title{font-size:17px;font-weight:500;color:#e2e8f0;margin-bottom:8px;}
    .step-body{font-size:13px;color:#8899aa;line-height:1.8;}
    .step-body b{color:#cbd5e0;}
    .sec-div{height:1px;background:linear-gradient(to right,#1e2d45,transparent);margin:28px 0;}
    </style>""", unsafe_allow_html=True)

    st.markdown("## Théorie — Méthodes de calcul des échangeurs")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="step-card" style="border-left-color:#f6ad55;">
        <div class="step-label">Méthode 1</div>
        <div class="step-title">ε–NTU</div>
        <div class="step-body">Basée sur l'efficacité et le nombre de transfert d'unités.
        Ne nécessite pas les températures de sortie. Idéale pour la vérification.</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="step-card" style="border-left-color:#2980b9;">
        <div class="step-label">Méthode 2</div>
        <div class="step-title">LMTD</div>
        <div class="step-body">Différence logarithmique moyenne de température.
        Utilisée quand les températures d'entrée ET de sortie sont connues. Q = UA·LMTD.</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="step-card" style="border-left-color:#27ae60;">
        <div class="step-label">Méthode 3</div>
        <div class="step-title">Stanton</div>
        <div class="step-body">Caractérise le transfert convectif via St = h/(ρ·u·cp).
        Permet d'obtenir h puis UA = h·A pour alimenter le calcul thermique.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

    st.subheader("Méthode ε–NTU")
    st.latex(r"C_h = \dot{m}_h c_{p,h},\quad C_c = \dot{m}_c c_{p,c},\quad C_{min}=\min(C_h,C_c)")
    st.latex(r"NTU = \frac{UA}{C_{min}},\quad R = \frac{C_{min}}{C_{max}}")
    st.latex(r"\varepsilon = f(NTU, R, \text{configuration}),\quad Q = \varepsilon\,C_{min}(T_{h,e}-T_{c,e})")

    st.subheader("Méthode LMTD")
    st.latex(r"\Delta T_1 = T_{h,e}-T_{c,s},\quad \Delta T_2 = T_{h,s}-T_{c,e}\quad \text{(contre-courant)}")
    st.latex(r"LMTD = \frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1 / \Delta T_2)}")
    st.latex(r"Q = UA \cdot LMTD \cdot F \quad (F=1 \text{ ici, extensible})")

    st.subheader("Méthode Stanton")
    st.latex(r"St = \frac{h}{\rho\,u\,c_p}")
    st.latex(r"h = St\cdot\rho\cdot u\cdot c_p \quad [W/m^2/K]")
    st.latex(r"UA = h \cdot A_{\text{echange}} \quad [W/K]")

    st.subheader("Nombres adimensionnels")
    st.latex(r"Re=\frac{\rho u L}{\mu},\quad Pr=\frac{\mu c_p}{k},\quad Nu=\frac{hL}{k},\quad Pe=Re\cdot Pr")
    st.latex(r"Bi=\frac{h L_c}{k_s},\quad Fo=\frac{\alpha t}{L^2},\quad j=St\cdot Pr^{2/3}")


# ═══════════════════════════════════════════════════════════════
# ══════════════ INTERFACE STREAMLIT ═══════════════════════════
# ═══════════════════════════════════════════════════════════════

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Calculation method
    st.subheader("Méthode de calcul")
    calc_method = st.selectbox("Méthode", ["ε-NTU", "LMTD", "Stanton"])

    st.divider()
    # Exchanger type
    flow_type = st.selectbox("Type d'échangeur", FLOW_TYPES)

    st.divider()
    # Exchange surface
    st.subheader("Surface d'échange")
    surface_type = st.selectbox("Type de surface", SURFACE_TYPES)

    # Dynamic geometry inputs per surface
    geo_params = {}
    st.markdown("**Géométrie**")
    if surface_type == "Tube lisse":
        geo_params["D"] = st.number_input("Diamètre D (m)", value=0.05, min_value=0.001, step=0.005, format="%.4f")
        geo_params["L"] = st.number_input("Longueur L (m)", value=2.0,  min_value=0.01,  step=0.1)
    elif surface_type == "Tube ailette":
        geo_params["D"]      = st.number_input("Diamètre tube D (m)", value=0.03, min_value=0.001, step=0.002, format="%.4f")
        geo_params["L"]      = st.number_input("Longueur L (m)", value=0.5, min_value=0.01, step=0.05)
        geo_params["D_f"]    = st.number_input("Diamètre ailette Df (m)", value=0.09, min_value=0.01, step=0.005, format="%.4f")
        geo_params["n_fins"] = st.number_input("Nombre d'ailettes", value=20, min_value=1, step=1)
    elif surface_type == "Faisceau tubulaire":
        geo_params["D"]   = st.number_input("Diamètre D (m)", value=0.025, min_value=0.001, step=0.002, format="%.4f")
        geo_params["L"]   = st.number_input("Longueur L (m)", value=1.5, min_value=0.1, step=0.1)
        geo_params["N_t"] = st.number_input("Nombre de tubes N_t", value=16, min_value=1, step=1)
    elif surface_type in ("Plaques planes", "Plaques ondulees"):
        geo_params["N_p"] = st.number_input("Nombre de plaques N_p", value=10, min_value=2, step=1)
        geo_params["W"]   = st.number_input("Largeur W (m)", value=0.4, min_value=0.01, step=0.05)
        geo_params["H"]   = st.number_input("Hauteur H (m)", value=0.3, min_value=0.01, step=0.05)
        if surface_type == "Plaques ondulees":
            geo_params["eta_corr"] = st.slider("Facteur η ondulation", 1.0, 1.5, 1.20, 0.01)
    elif surface_type == "Serpentin":
        geo_params["D"]       = st.number_input("Diamètre tube D (m)", value=0.025, min_value=0.001, step=0.002, format="%.4f")
        geo_params["D_s"]     = st.number_input("Diamètre spire Ds (m)", value=0.3, min_value=0.05, step=0.05)
        geo_params["N_turns"] = st.number_input("Nombre de spires", value=8, min_value=1, step=1)
    elif surface_type == "Surface annulaire":
        geo_params["R1"] = st.number_input("Rayon interne R1 (m)", value=0.05, min_value=0.001, step=0.005, format="%.4f")
        geo_params["R2"] = st.number_input("Rayon externe R2 (m)", value=0.15, min_value=0.01, step=0.01)
    else:  # personnalisée
        geo_params["A_user"] = st.number_input("Surface A (m²)", value=2.0, min_value=0.001, step=0.1)

    st.divider()
    st.subheader("Fluide chaud")
    Th_in = st.number_input("The — Température entrée (°C)", value=80.0, step=1.0)
    mh    = st.number_input("ṁh — Débit massique (kg/s)",    value=1.0, min_value=0.001, step=0.1)
    cph   = st.number_input("cph — Chaleur spécifique (J/kg·K)", value=4180.0, min_value=1.0, step=10.0)

    st.divider()
    st.subheader("Fluide froid")
    Tc_in = st.number_input("Tce — Température entrée (°C)", value=20.0, step=1.0)
    mc    = st.number_input("ṁc — Débit massique (kg/s)",    value=1.0, min_value=0.001, step=0.1)
    cpc   = st.number_input("cpc — Chaleur spécifique (J/kg·K)", value=4180.0, min_value=1.0, step=10.0)

    st.divider()
    # Fluid properties for dimensionless numbers
    st.subheader("Propriétés fluide (nombres adim.)")
    with st.expander("Propriétés (optionnel)"):
        rho_f  = st.number_input("ρ — Densité (kg/m³)",         value=1000.0, min_value=0.001, step=10.0)
        mu_f   = st.number_input("μ — Viscosité dyn. (Pa·s)",   value=0.001,  min_value=1e-7, step=0.0001, format="%.5f")
        k_f    = st.number_input("k — Cond. thermique (W/m·K)", value=0.6,    min_value=1e-4, step=0.01)
        u_f    = st.number_input("u — Vitesse (m/s)",           value=1.0,    min_value=0.001, step=0.1)
        L_char = st.number_input("L — Long. caract. (m)",       value=geo_params.get("D", 0.05), min_value=0.001, step=0.01)
        ks_f   = st.number_input("ks — Cond. solide (W/m·K)",   value=50.0,   min_value=0.001, step=1.0)
        Lc_f   = st.number_input("Lc — Epaisseur Biot (m)",     value=0.005,  min_value=1e-4, step=0.001, format="%.4f")

    st.divider()
    st.subheader("Géométrie 3D échangeur")
    length_3d = st.slider("Longueur échangeur (m)", 1.0, 6.0, 3.0, 0.1)
    if flow_type in ("Contre-courant","Co-courant","Calandre (1 passe) / Tubes (2 passes)"):
        n_tubes_3d = st.slider("Tubes internes", 2, 16, 6, 2)
        shell_r_3d = st.slider("Rayon calandre (m)", 0.2, 1.0, 0.5, 0.05)
        params_3d = dict(length=length_3d, shell_radius=shell_r_3d, tube_radius=0.05, n_tubes=n_tubes_3d)
    else:
        n_ch_3d = st.slider("Nombre de canaux", 3, 12, 6, 1)
        params_3d = dict(length=length_3d, width=length_3d, height=1.2, n_channels=n_ch_3d)


# ═══════════════════════════════════════════════════════════════
# ── MAIN TABS ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

tab_calc, tab_theory = st.tabs(["🔢 Calculateur", "📖 Théorie & Méthodologie"])

with tab_theory:
    show_theory_page()

with tab_calc:

    # ────────────────────────────────────────────────────────────
    #  SECTION A — CALCUL DE LA SURFACE
    # ────────────────────────────────────────────────────────────
    st.markdown("### 📐 Surface d'échange")
    area_result = compute_exchange_area(surface_type, geo_params)
    if area_result["valid"]:
        ca1, ca2, ca3 = st.columns(3)
        ca1.metric("Surface A (m²)", f"{area_result['A']:.4f}")
        ca2.metric("Type", surface_type)
        ca3.markdown(f"**Formule :** `{area_result['formula']}`")
        st.caption(f"ℹ️ {area_result['note']}")
    else:
        st.error(f"Calcul de surface invalide : {area_result['note']}")

    A_exchange = area_result.get("A")

    st.markdown("---")

    # ────────────────────────────────────────────────────────────
    #  SECTION B — MODE DE CALCUL PAR MÉTHODE
    # ────────────────────────────────────────────────────────────
    UA   = None
    res  = None
    lmtd_res    = None
    stanton_data = None

    # ── ε-NTU ──────────────────────────────────────────────────
    if calc_method == "ε-NTU":
        st.markdown("### 🔵 Méthode ε–NTU")
        col_mode, _ = st.columns([3,1])
        with col_mode:
            mode = st.radio("Mode", ["Calcul direct (UA connu)", "Dimensionnement (cible T)"], horizontal=True)

        if mode == "Calcul direct (UA connu)":
            UA = st.number_input("UA (W/K)", value=2000.0, min_value=0.0, step=100.0)
        else:
            c1, c2 = st.columns(2)
            with c1:
                tgt_choice = st.selectbox("T cible", ["Tcs — Sortie froid","Tfe — Sortie chaud"])
            with c2:
                tgt_val = st.number_input("Valeur (°C)", value=50.0, step=1.0)
            tgt_key = "Tc_out" if tgt_choice.startswith("Tcs") else "Th_out"
            if st.button("Calculer UA requis", type="primary"):
                UA_sol = solve_UA_bisection(Th_in, Tc_in, mh, cph, mc, cpc, flow_type,
                                           target=tgt_key, target_value=tgt_val)
                if UA_sol is None:
                    st.error("Pas de solution — vérifiez que la cible est physiquement atteignable.")
                else:
                    UA = float(UA_sol)
                    st.session_state["last_UA"] = UA
                    st.success(f"UA requis = {UA:,.2f} W/K")
            if UA is None:
                UA = st.session_state.get("last_UA")

        if UA and UA > 0 and Th_in > Tc_in:
            res = compute_entu(Th_in, Tc_in, mh, cph, mc, cpc, UA, flow_type)
            if res:
                res["Th_in_ref"] = Th_in; res["Tc_in_ref"] = Tc_in

    # ── LMTD ───────────────────────────────────────────────────
    elif calc_method == "LMTD":
        st.markdown("### 🟣 Méthode LMTD")
        l1, l2 = st.columns(2)
        with l1:
            lmtd_flow = st.selectbox("Configuration écoulement", ["counter","parallel"],
                                     format_func=lambda x: "Contre-courant" if x=="counter" else "Co-courant")
        with l2:
            UA = st.number_input("UA (W/K)", value=2000.0, min_value=0.0, step=100.0)

        if UA and UA > 0 and Th_in > Tc_in:
            lmtd_res = compute_lmtd_full(Th_in, Tc_in, mh, cph, mc, cpc, UA, lmtd_flow, flow_type)
            if lmtd_res:
                res = lmtd_res
                res["Th_in_ref"] = Th_in; res["Tc_in_ref"] = Tc_in

    # ── Stanton ─────────────────────────────────────────────────
    elif calc_method == "Stanton":
        st.markdown("### 🟢 Méthode Stanton")
        st_mode = st.radio("Mode Stanton", ["Option A — Stanton direct", "Option B — Depuis Re, Pr, Nu"], horizontal=True)

        if st_mode == "Option A — Stanton direct":
            s1, s2, s3, s4 = st.columns(4)
            St_val = s1.number_input("Nombre de Stanton St", value=0.003, min_value=1e-7, step=0.0001, format="%.6f")
            rho_st = s2.number_input("ρ (kg/m³)", value=1000.0, min_value=0.01, step=10.0)
            u_st   = s3.number_input("u (m/s)",   value=1.5,    min_value=0.001, step=0.1)
            cp_st  = s4.number_input("cp (J/kg·K)",value=4180.0,min_value=1.0,  step=10.0)
            stanton_data = compute_stanton_direct(St_val, rho_st, u_st, cp_st)
            st.info(f"**h calculé = {stanton_data['h']:.4f} W/m²/K**  (h = St·ρ·u·cp)")
        else:
            s1, s2, s3, s4 = st.columns(4)
            Re_st  = s1.number_input("Reynolds Re", value=10000.0, min_value=1.0, step=100.0)
            Pr_st  = s2.number_input("Prandtl Pr",  value=7.0,     min_value=0.01,step=0.1)
            Nu_st  = s3.number_input("Nusselt Nu",  value=100.0,   min_value=0.1, step=1.0)
            L_st   = s4.number_input("L caract. (m)",value=0.05,   min_value=0.001,step=0.005,format="%.4f")
            dim_st = compute_stanton_from_dimensionless(Re_st, Pr_st, Nu_st, L_st)
            St_val = dim_st["St"]
            k_st   = st.number_input("k fluide (W/m·K) pour h = Nu·k/L", value=0.6, min_value=0.001, step=0.01)
            h_val  = Nu_st * k_st / L_st
            stanton_data = {"St": St_val, "h": h_val, "mode": "from_dimless",
                            "Re": Re_st, "Pr": Pr_st, "Nu": Nu_st}
            st.info(f"**St = {St_val:.6g}** | **h = {h_val:.4f} W/m²/K** (h = Nu·k/L)")

        if stanton_data and A_exchange and A_exchange > 0:
            UA, res = stanton_full_result(stanton_data, A_exchange, Th_in, Tc_in, mh, cph, mc, cpc)
            if res:
                res["Th_in_ref"] = Th_in; res["Tc_in_ref"] = Tc_in
                st.success(f"UA = h·A = {stanton_data['h']:.4f} × {A_exchange:.4f} = **{UA:.2f} W/K**")
            else:
                st.error("Calcul thermique impossible avec ces données Stanton.")

    # ────────────────────────────────────────────────────────────
    #  SECTION C — RÉSULTATS
    # ────────────────────────────────────────────────────────────
    if res is not None and UA is not None and UA > 0:

        if Th_in <= Tc_in:
            st.error("⛔ The doit être > Tce.")
        else:
            st.markdown("---")
            st.subheader("📊 Résultats thermiques")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Q (W)",         f"{res['Q']:,.1f}")
            m2.metric("ε (efficacité)",f"{res['epsilon']:.4f}")
            m3.metric("NTU",           f"{res['NTU']:.4f}")
            m4.metric("Tfe (°C)",      f"{res['Th_out']:.2f}")
            m5.metric("Tcs (°C)",      f"{res['Tc_out']:.2f}")

            # LMTD extra
            if calc_method == "LMTD" and lmtd_res and lmtd_res.get("LMTD"):
                lc1, lc2, lc3 = st.columns(3)
                lc1.metric("ΔT₁ (°C)",  f"{lmtd_res['dT1']:.3f}")
                lc2.metric("ΔT₂ (°C)",  f"{lmtd_res['dT2']:.3f}")
                lc3.metric("LMTD (°C)", f"{lmtd_res['LMTD']:.3f}")
            elif calc_method == "LMTD" and lmtd_res and not lmtd_res.get("valid"):
                st.error(lmtd_res.get("warning","LMTD invalide"))

            # Stanton extra
            if calc_method == "Stanton" and stanton_data:
                sc1, sc2 = st.columns(2)
                sc1.metric("St", f"{stanton_data.get('St', 0):.6g}")
                sc2.metric("h (W/m²/K)", f"{stanton_data.get('h', 0):.4f}")

            # Validation
            val_checks = validate_inputs(Th_in, Tc_in, mh, mc, cph, cpc, UA, A_exchange)
            val_checks += validate_thermal_outputs(res)
            has_err = any(v[0] == "error" for v in val_checks)
            if has_err:
                st.warning("⚠️ Des vérifications ont échoué — voir détails ci-dessous.")
            else:
                st.success("✅ Configuration physiquement cohérente.")

            with st.expander("📋 Détails validation"):
                for level, msg in val_checks:
                    icon = "✅" if level=="ok" else ("❌" if level=="error" else "⚠️")
                    st.markdown(f"{icon} {msg}")

            with st.expander("📐 Détails C, NTU, Cr"):
                d1,d2,d3,d4 = st.columns(4)
                d1.metric("Ch (W/K)",  f"{res['Ch']:.2f}")
                d1.metric("Cc (W/K)",  f"{res['Cc']:.2f}")
                d2.metric("Cmin (W/K)",f"{res['Cmin']:.2f}")
                d2.metric("Cmax (W/K)",f"{res['Cmax']:.2f}")
                d3.metric("R = Cr",    f"{res['Cr']:.4f}")
                d3.metric("UA (W/K)",  f"{UA:,.2f}")
                d4.metric("Qmax (W)",  f"{res['Qmax']:,.1f}")

            # ────────────────────────────────────────────────────
            #  SECTION D — NOMBRES ADIMENSIONNELS
            # ────────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("🔢 Nombres adimensionnels")

            h_for_dim = stanton_data.get("h") if stanton_data else None
            dim_numbers = compute_dimensionless(
                rho=rho_f, u=u_f, L=L_char, mu=mu_f,
                cp=cph, k=k_f, h=h_for_dim, ks=ks_f, Lc=Lc_f,
            )

            if dim_numbers:
                dim_labels = {
                    "Re": "Reynolds",  "Pr": "Prandtl",   "Nu": "Nusselt",
                    "St": "Stanton",   "Pe": "Peclet",     "Bi": "Biot",
                    "Fo": "Fourier",   "Gz": "Graetz",     "j":  "Colburn j",
                }
                cols_dim = st.columns(4)
                k_idx = 0
                for key, name in dim_labels.items():
                    if key in dim_numbers:
                        cols_dim[k_idx % 4].metric(name, f"{dim_numbers[key]:.5g}")
                        k_idx += 1
                if "regime" in dim_numbers:
                    st.info(f"**Régime :** {dim_numbers['regime']}")
                if h_for_dim is None:
                    st.caption("Nu, St, Bi, Colburn j nécessitent h. Entrez h via la méthode Stanton ou activez les propriétés fluide.")
            else:
                st.info("Entrez les propriétés fluide (sidebar → Propriétés fluide) pour calculer les nombres adimensionnels.")

            # ────────────────────────────────────────────────────
            #  SECTION E — VISUALISATIONS 2D
            # ────────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📐 Schéma 2D")
            fig_schema = draw_schema(res, UA, Th_in, Tc_in)
            st.pyplot(fig_schema); plt.close(fig_schema)

            st.subheader("🌡️ Profil de températures")
            fig_profile = draw_temperature_profile(res, Th_in, Tc_in)
            st.pyplot(fig_profile); plt.close(fig_profile)

            if calc_method == "LMTD" and lmtd_res and lmtd_res.get("LMTD"):
                st.subheader("📊 Diagramme LMTD")
                fig_lmtd_plot = draw_lmtd_diagram(lmtd_res, Th_in, Tc_in)
                if fig_lmtd_plot:
                    col_l, col_r = st.columns([2, 3])
                    with col_l:
                        st.pyplot(fig_lmtd_plot)
                    plt.close(fig_lmtd_plot)

            st.subheader("📈 Courbes ε–NTU")
            fig_eps = draw_eps_vs_NTU(res)
            st.pyplot(fig_eps); plt.close(fig_eps)

            # ────────────────────────────────────────────────────
            #  SECTION F — VISUALISATION 3D ÉCHANGEUR
            # ────────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("🧊 Visualisation 3D — Échangeur")
            with st.spinner("Génération du modèle 3D…"):
                fig_3d = build_3d_visualization(res, Th_in, Tc_in, params_3d)
            st.plotly_chart(fig_3d, use_container_width=True)

            # Color legend
            T_lo_leg = min(Tc_in, res["Tc_out"])
            T_hi_leg = max(Th_in, res["Th_out"])
            Tv = np.linspace(T_lo_leg, T_hi_leg, 50)
            nv = (Tv - T_lo_leg) / max(T_hi_leg - T_lo_leg, 1)
            col_l2, col_c2, col_r2 = st.columns([1, 3, 1])
            with col_c2:
                leg = go.Figure()
                for i in range(len(Tv)-1):
                    r = int(min(255, nv[i]*2*255))
                    g = int(255 - abs(nv[i]-0.5)*2*255)
                    b = int(min(255, (1-nv[i])*2*255))
                    leg.add_trace(go.Scatter(x=[Tv[i],Tv[i+1]], y=[0,0], mode="lines",
                                             line=dict(color=f"rgb({r},{g},{b})", width=20),
                                             showlegend=False, hoverinfo="skip"))
                leg.update_layout(height=70, margin=dict(l=30,r=30,t=5,b=25),
                                  xaxis=dict(title="Température (°C)", showgrid=False,
                                             tickvals=[T_lo_leg,(T_lo_leg+T_hi_leg)/2,T_hi_leg],
                                             ticktext=[f"{T_lo_leg:.0f}°C",f"{(T_lo_leg+T_hi_leg)/2:.0f}°C",f"{T_hi_leg:.0f}°C"]),
                                  yaxis=dict(visible=False),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(leg, use_container_width=True)

            # ────────────────────────────────────────────────────
            #  SECTION G — VISUALISATION 3D SURFACE D'ÉCHANGE
            # ────────────────────────────────────────────────────
            st.markdown("---")
            st.subheader(f"🔶 Visualisation 3D — Surface d'échange ({surface_type})")
            st.caption("Surface active mise en évidence en orange.")
            with st.spinner("Génération de la surface 3D…"):
                fig_surf_3d = build_surface_3d(surface_type, geo_params, res)
            st.plotly_chart(fig_surf_3d, use_container_width=True)

            # ────────────────────────────────────────────────────
            #  SECTION H — RAPPORT PDF
            # ────────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📄 Rapport PDF")
            if st.button("Générer le rapport PDF", type="primary"):
                with st.spinner("Génération du rapport…"):
                    f_sch  = draw_schema(res, UA, Th_in, Tc_in)
                    f_pro  = draw_temperature_profile(res, Th_in, Tc_in)
                    f_eps2 = draw_eps_vs_NTU(res)
                    f_lmtd_pdf = draw_lmtd_diagram(lmtd_res, Th_in, Tc_in) if (lmtd_res and lmtd_res.get("LMTD")) else None

                    pdf_bytes = generate_pdf(
                        res, UA, Th_in, Tc_in, mh, mc, cph, cpc,
                        f_sch, f_pro, f_eps2,
                        surface_type, area_result, dim_numbers,
                        val_checks, calc_method,
                        lmtd_res=lmtd_res,
                        stanton_data=stanton_data,
                        fig_lmtd=f_lmtd_pdf,
                    )
                    plt.close("all")

                fname = ("rapport_echangeur_"
                         + flow_type.replace(" ","_").replace("/","_")
                         + f"_{calc_method.replace(' ','_')}.pdf")
                st.download_button("⬇️ Télécharger le rapport PDF",
                                   data=pdf_bytes, file_name=fname, mime="application/pdf")

    else:
        if calc_method == "Stanton" and (not A_exchange or A_exchange <= 0):
            st.info("La méthode Stanton nécessite une surface A valide — vérifiez la géométrie.")
        elif UA is None or UA == 0:
            st.info("Configurez un UA > 0 ou lancez le dimensionnement pour afficher les résultats.")
