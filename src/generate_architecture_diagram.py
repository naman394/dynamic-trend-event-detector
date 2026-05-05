"""
Publication-Ready Architecture Diagram
=======================================
Generates a conference-paper-quality architecture diagram for the
Dynamic Trend & Event Detector hybrid system.

Features:
  • Two-path layout: ML (blue) and DL (orange) with explicit tensor shapes
  • Fusion mechanism shown with green box and formula
  • Dashed feedback arrow (SBERT confidence → LDA weight)
  • Standard ML diagram notation (hexagons for models, cylinders for data)
  • Color legend, 300 DPI, saved as both PNG and PDF

Run: python src/generate_architecture_diagram.py
Output: reports/architecture_diagram.png
        reports/architecture_diagram.pdf
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

ROOT     = Path(__file__).parent.parent
OUT_DIR  = ROOT / "reports"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────────────────
C_DATA    = "#1e293b"   # dark slate — data stores
C_ML      = "#1d4ed8"   # blue — ML path
C_DL      = "#c2410c"   # orange — DL path
C_FUSION  = "#15803d"   # green — fusion / output
C_GDELT   = "#7e22ce"   # purple — live feed
C_ARROW   = "#475569"   # arrow colour
C_FB      = "#dc2626"   # feedback arrow — red dashed
BG        = "#f8fafc"   # figure background

FONT = "DejaVu Sans"

# ── Layout constants (data coords in an 18 × 13.5 axes) ────────────────────────
W_BOX  = 3.4    # standard box width
H_BOX  = 0.72   # standard box height
W_WIDE = 4.0    # wider boxes (SBERT, K-Means)
W_FUSE = 9.2    # fusion box width

X_ML    =  2.5   # ML path x-centre
X_DL    = 14.5   # DL path x-centre
X_MID   =  8.5   # centre lane
X_SEED  =  6.7   # seed arrow/box x-centre

Y_TITLE = 13.0
Y_DATA  = 11.8
Y_PREP  = 10.2
Y_MODEL =  8.4
Y_REPR  =  6.6   # seeds / UMAP
Y_CLUST =  4.8   # K-Means (DL side), topic rarity (ML side)
Y_VEL   =  3.2   # semantic velocity + GDELT
Y_FUSE  =  1.4   # fusion box
Y_OUT   =  0.2   # output


def ax_box(ax, xc, yc, w, h, label, sublabel, color, text_color="white",
           style="round,pad=0.05", zorder=4, fontsize=9.5):
    """Draw a labelled rounded rectangle."""
    x0 = xc - w / 2
    y0 = yc - h / 2
    patch = FancyBboxPatch((x0, y0), w, h,
                           boxstyle=style,
                           facecolor=color, edgecolor="white",
                           linewidth=1.6, zorder=zorder)
    ax.add_patch(patch)
    ax.text(xc, yc + 0.08, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=zorder + 1,
            fontfamily=FONT)
    if sublabel:
        ax.text(xc, yc - 0.22, sublabel,
                ha="center", va="center", fontsize=7.2,
                color=text_color, alpha=0.88, zorder=zorder + 1,
                fontfamily=FONT, style="italic")


def ax_arrow(ax, x0, y0, x1, y1, label="", color=C_ARROW,
             ls="-", lw=1.6, arrowstyle="->", zorder=3):
    """Draw an annotated arrow."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=arrowstyle, color=color,
                        lw=lw, linestyle=ls,
                        connectionstyle="arc3,rad=0.0"),
        zorder=zorder,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.12, my, label,
                ha="left", va="center", fontsize=7.0,
                color=color, fontfamily=FONT, zorder=zorder + 1,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=color, alpha=0.85, linewidth=0.8))


def make_diagram():
    fig, ax = plt.subplots(figsize=(18, 13.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 13.8)
    ax.axis("off")

    # ── Title ──────────────────────────────────────────────────────────────────
    ax.text(9.0, 13.25,
            "Dynamic Trend & Event Detector — Hybrid Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=C_DATA, fontfamily=FONT)
    ax.text(9.0, 12.75,
            "LDA-Seeded SBERT K-Means  ·  Learned Fusion Weight α*  ·  "
            "Live GDELT Integration",
            ha="center", va="center", fontsize=9.5,
            color="#64748b", fontfamily=FONT)

    # ── Path headers ───────────────────────────────────────────────────────────
    for xc, txt, col in [(X_ML, "ML PATH — Statistical", C_ML),
                          (X_DL, "DL PATH — Neural", C_DL)]:
        ax.text(xc, 12.35, txt, ha="center", va="center",
                fontsize=10, fontweight="bold", color=col,
                fontfamily=FONT,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=col,
                          alpha=0.12, edgecolor=col, linewidth=1.2))

    # ──────────────────────────────────────────────────────────────────────────
    # ROW 1 — Data sources
    # ──────────────────────────────────────────────────────────────────────────
    ax_box(ax, X_ML, Y_DATA, W_BOX, H_BOX,
           "ABC News Corpus", "1,244,184 × 2  (date, text)", C_DATA)
    ax_box(ax, X_DL, Y_DATA, W_WIDE, H_BOX,
           "Stratified Sample", "49,989 × seq_len  (2,631/year)", C_DATA)

    # ──────────────────────────────────────────────────────────────────────────
    # ROW 2 — Preprocessing
    # ──────────────────────────────────────────────────────────────────────────
    ax_box(ax, X_ML, Y_PREP, W_BOX, H_BOX,
           "CountVectorizer", "1,244,184 × 15,000  (raw counts)", C_ML, fontsize=9)
    ax_box(ax, X_DL, Y_PREP, W_WIDE, H_BOX,
           "SBERT  all-MiniLM-L6-v2", "6 transformer layers · mean pool", C_DL)

    ax_arrow(ax, X_ML, Y_DATA - H_BOX/2, X_ML, Y_PREP + H_BOX/2,
             "(1,244,184 × raw text)")
    ax_arrow(ax, X_DL, Y_DATA - H_BOX/2, X_DL, Y_PREP + H_BOX/2,
             "(49,989 × raw text)")

    # ──────────────────────────────────────────────────────────────────────────
    # ROW 3 — Core models
    # ──────────────────────────────────────────────────────────────────────────
    ax_box(ax, X_ML, Y_MODEL, W_BOX, H_BOX,
           "LDA  [K=10 topics]",
           "C_V=0.3575 · perplexity=5,869", C_ML)
    ax_box(ax, X_DL, Y_MODEL, W_WIDE, H_BOX,
           "UMAP  (384→3D + 384→2D)",
           "n_neighbors=30 · cosine metric", C_DL)

    ax_arrow(ax, X_ML, Y_PREP - H_BOX/2, X_ML, Y_MODEL + H_BOX/2,
             "(1,244,184 × 15,000)")
    ax_arrow(ax, X_DL, Y_PREP - H_BOX/2, X_DL, Y_MODEL + H_BOX/2,
             "(49,989 × 384)")

    # ──────────────────────────────────────────────────────────────────────────
    # ROW 4 — Seeds + K-Means (key hybrid interaction)
    # ──────────────────────────────────────────────────────────────────────────
    ax_box(ax, X_ML, Y_REPR, W_BOX, H_BOX,
           "LDA Topic Seeds",
           "top-10 words/topic → SBERT → (10 × 384)", C_ML)
    ax_box(ax, X_DL, Y_REPR, W_WIDE, H_BOX,
           "K-Means  [K=10, init=LDA seeds]",
           "n_init=1 · deterministic centroids", C_DL)

    ax_arrow(ax, X_ML, Y_MODEL - H_BOX/2, X_ML, Y_REPR + H_BOX/2,
             "(10 × 15,000 word dist.)")
    ax_arrow(ax, X_DL, Y_MODEL - H_BOX/2, X_DL, Y_REPR + H_BOX/2,
             "(49,989 × 3)  UMAP")

    # ── FORWARD arrow: LDA seeds → K-Means ────────────────────────────────────
    ax.annotate(
        "", xy=(X_DL - W_WIDE/2, Y_REPR),
        xytext=(X_ML + W_BOX/2, Y_REPR),
        arrowprops=dict(
            arrowstyle="-|>", color=C_ML, lw=2.2,
            connectionstyle="arc3,rad=-0.25",
        ),
        zorder=5,
    )
    ax.text((X_ML + X_DL) / 2, Y_REPR + 0.52,
            "① FORWARD: LDA seeds\nK-Means init  (10 × 384)",
            ha="center", va="center", fontsize=8.2, fontweight="bold",
            color=C_ML, fontfamily=FONT,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#eff6ff",
                      edgecolor=C_ML, linewidth=1.2))

    # ──────────────────────────────────────────────────────────────────────────
    # ROW 5 — Topic rarity + Confidence / Semantic Velocity
    # ──────────────────────────────────────────────────────────────────────────
    ax_box(ax, X_ML, Y_CLUST, W_BOX, H_BOX,
           "LDA Topic Rarity",
           "1 − max cosine_sim(event, LDA seeds)", C_ML)
    ax_box(ax, X_DL, Y_CLUST, W_WIDE, H_BOX,
           "SBERT Confidence",
           "cos_sim(doc, centroid) per doc  (49,989,)", C_DL)

    ax_arrow(ax, X_ML, Y_REPR - H_BOX/2, X_ML, Y_CLUST + H_BOX/2,
             "(10 topic seeds)")
    ax_arrow(ax, X_DL, Y_REPR - H_BOX/2, X_DL, Y_CLUST + H_BOX/2,
             "(49,989,) cluster labels")

    # ── FEEDBACK arrow: SBERT confidence → LDA weight ─────────────────────────
    ax.annotate(
        "", xy=(X_ML + W_BOX/2, Y_CLUST),
        xytext=(X_DL - W_WIDE/2, Y_CLUST),
        arrowprops=dict(
            arrowstyle="-|>", color=C_FB, lw=2.0,
            linestyle="dashed",
            connectionstyle="arc3,rad=0.3",
        ),
        zorder=5,
    )
    ax.text((X_ML + X_DL) / 2, Y_CLUST - 0.62,
            "② FEEDBACK: SBERT confidence\n→ LDA rarity weight  (doc-level α)",
            ha="center", va="center", fontsize=8.2, fontweight="bold",
            color=C_FB, fontfamily=FONT,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff1f2",
                      edgecolor=C_FB, linewidth=1.2))

    # ──────────────────────────────────────────────────────────────────────────
    # ROW 6 — Semantic Velocity + GDELT
    # ──────────────────────────────────────────────────────────────────────────
    ax_box(ax, X_DL, Y_VEL, W_WIDE, H_BOX,
           "Semantic Velocity  V_s(t)",
           "V_s = 1 − cos(centroid_t,  centroid_t−1)", C_DL)
    ax_box(ax, X_MID - 1.0, Y_VEL, 3.8, H_BOX,
           "GDELT Live  (every 15 min)",
           "100+ languages · themes + tone · V2THEMES", C_GDELT)

    ax_arrow(ax, X_DL, Y_CLUST - H_BOX/2, X_DL, Y_VEL + H_BOX/2,
             "(52 weeks/year × 384)")
    ax_arrow(ax, X_ML, Y_CLUST - H_BOX/2,
             X_MID - 1.0, Y_VEL + H_BOX/2,
             "(N × 384  GDELT themes)")

    # ──────────────────────────────────────────────────────────────────────────
    # FUSION BOX
    # ──────────────────────────────────────────────────────────────────────────
    fx  = 9.0
    fy  = Y_FUSE
    fw  = W_FUSE
    fh  = 1.05

    fusion_patch = FancyBboxPatch(
        (fx - fw/2, fy - fh/2), fw, fh,
        boxstyle="round,pad=0.08",
        facecolor="#dcfce7", edgecolor=C_FUSION, linewidth=2.2, zorder=4
    )
    ax.add_patch(fusion_patch)
    ax.text(fx, fy + 0.25,
            "③ HYBRID FUSION",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_FUSION, fontfamily=FONT, zorder=5)
    ax.text(fx, fy - 0.08,
            r"$S_I^{hybrid} = (\alpha^* \cdot SBERT\_uniq + (1-\alpha^*) \cdot LDA\_rarity) \times |\tau|$",
            ha="center", va="center", fontsize=9.5,
            color="#166534", fontfamily=FONT, zorder=5)
    ax.text(fx, fy - 0.38,
            r"$\alpha^*$ = argmin(−MRR)  on anchor events  ·  "
            r"$\alpha \to 0$: LDA-trust  ·  $\alpha \to 1$: SBERT-trust",
            ha="center", va="center", fontsize=8.0,
            color="#166534", fontfamily=FONT, zorder=5, style="italic")

    # Arrows into fusion
    ax_arrow(ax, X_DL,           Y_VEL - H_BOX/2, fx + 1.8, fy + fh/2,
             "V_s rupture weeks")
    ax_arrow(ax, X_MID - 1.0,   Y_VEL - H_BOX/2, fx,       fy + fh/2,
             "GDELT events + tone")
    ax_arrow(ax, X_ML,           Y_CLUST - H_BOX/2, fx - 2.0, fy + fh/2,
             "LDA rarity score")

    # ──────────────────────────────────────────────────────────────────────────
    # OUTPUT
    # ──────────────────────────────────────────────────────────────────────────
    ax_box(ax, fx, Y_OUT, W_FUSE - 1.0, 0.65,
           "Ranked Alert Feed  ·  Rupture Detection  ·  Real-time Dashboard",
           "", C_FUSION, fontsize=9)
    ax_arrow(ax, fx, fy - fh/2, fx, Y_OUT + 0.32, "scored + ranked events")

    # ──────────────────────────────────────────────────────────────────────────
    # Legend
    # ──────────────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=C_DATA,   label="Data Store"),
        mpatches.Patch(facecolor=C_ML,     label="ML Component (LDA / TF-IDF)"),
        mpatches.Patch(facecolor=C_DL,     label="DL Component (SBERT / UMAP / K-Means)"),
        mpatches.Patch(facecolor=C_FUSION, label="Hybrid Fusion"),
        mpatches.Patch(facecolor=C_GDELT,  label="Live Data Feed (GDELT)"),
        mpatches.Patch(facecolor=C_FB,     label="Feedback loop (SBERT→LDA)"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=8.2,
              framealpha=0.92, edgecolor="#cbd5e1",
              bbox_to_anchor=(0.01, 0.01))

    # ── Tensor shape annotation strip (right margin) ───────────────────────────
    tensor_notes = [
        (Y_DATA,  "Input: 1,244,184 × 2  /  49,989 × seq"),
        (Y_PREP,  "Vocab: 15,000 terms  /  Embed: 384-dim"),
        (Y_MODEL, "LDA: 10 × 15,000  /  UMAP: N × 3"),
        (Y_REPR,  "Seeds: 10 × 384  /  Labels: 49,989"),
        (Y_CLUST, "Rarity: N,  /  Confidence: 49,989"),
        (Y_VEL,   "Velocity: T × 1  /  GDELT: M × 384"),
        (Y_FUSE,  "S_I: M × 1  (hybrid scored)"),
    ]
    for y, note in tensor_notes:
        ax.text(17.9, y, note,
                ha="right", va="center", fontsize=6.8,
                color="#64748b", fontfamily=FONT,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="#f1f5f9",
                          edgecolor="#cbd5e1", linewidth=0.7))

    plt.tight_layout(pad=0.3)
    return fig


def run():
    print("Generating publication-quality architecture diagram …")
    fig = make_diagram()

    png_path = OUT_DIR / "architecture_diagram.png"
    pdf_path = OUT_DIR / "architecture_diagram.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    fig.savefig(pdf_path, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)

    print(f"  Saved (300 DPI PNG) → {png_path}")
    print(f"  Saved (vector PDF)  → {pdf_path}")
    print("✅ Architecture diagram complete.")


if __name__ == "__main__":
    run()
