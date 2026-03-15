"""
gen_completion_pdf.py — Generate publication-quality OSF completion report PDF
with embedded figures, summary tables, and GitHub link.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

REPO_URL = "https://github.com/0ryant/cma-m2-eval"
OSF_URL  = "https://osf.io/gesyh"
OSF_DOI  = "10.17605/OSF.IO/GESYH"

RESULTS = Path(__file__).parent / "results"
ABLATION = RESULTS / "v5_ablation" / "ablation_results.json"
SCALE    = RESULTS / "v5_scale" / "scale_results.json"

# Colour palette
C_PASS   = "#2ecc71"
C_FAIL   = "#e74c3c"
C_WEAK   = "#f39c12"
C_BG     = "#fafafa"
C_M2     = "#3498db"
C_ALT    = "#95a5a6"
C_BIO    = "#2ecc71"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.facecolor": C_BG,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def title_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.78, "CMA / M2 Pre-Registration", fontsize=24, ha="center",
            fontweight="bold", color="#2c3e50")
    ax.text(0.5, 0.72, "Completion Report", fontsize=20, ha="center",
            color="#2c3e50")

    ax.text(0.5, 0.62, "Typed Strategic Families Produce Distinct Failure Morphologies:",
            fontsize=11, ha="center", color="#555", style="italic")
    ax.text(0.5, 0.585, "A Preregistered Evaluation of the M2 Policy Layer",
            fontsize=11, ha="center", color="#555", style="italic")
    ax.text(0.5, 0.55, "Against a Structurally Matched Latent-State Baseline",
            fontsize=11, ha="center", color="#555", style="italic")

    ax.text(0.5, 0.46, "Ryan Tilcock", fontsize=14, ha="center", color="#2c3e50")
    ax.text(0.5, 0.42, "March 2026", fontsize=12, ha="center", color="#7f8c8d")

    ax.text(0.5, 0.34, f"OSF Registration: {OSF_URL}", fontsize=10, ha="center",
            color="#3498db")
    ax.text(0.5, 0.31, f"DOI: {OSF_DOI}", fontsize=10, ha="center", color="#3498db")
    ax.text(0.5, 0.28, f"Source Code: {REPO_URL}", fontsize=10, ha="center",
            color="#3498db")

    ax.axhline(y=0.22, xmin=0.15, xmax=0.85, color="#bdc3c7", linewidth=1)

    ax.text(0.5, 0.16, "Registered before publication run  •  Verdict contract frozen 2026-03-11",
            fontsize=9, ha="center", color="#95a5a6")
    ax.text(0.5, 0.13, "Paper submitted to WOA 2026 (27th Workshop \"From Objects to Agents\")",
            fontsize=9, ha="center", color="#95a5a6")

    pdf.savefig(fig)
    plt.close(fig)


def observable_summary_page(pdf):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    obs = [
        ("Obs 1\nTactic CV", 2.118, "> 0.25", "PASS"),
        ("Obs 2\nSwitch Cost", None, "≤ 0.90×", "UNDERPOWERED"),
        ("Obs 3\nCollapse ρ", 0.878, "≥ 0.85", "PASS"),
        ("Obs 4\nSocial Signal", -0.363, "≥ +0.05", "FAIL"),
        ("Obs 5\nCharacter r", 0.897, "≥ 0.70", "FRAGILE PASS"),
        ("Obs 6\nCF Coherence", 0.011, "M2 > LSM", "WEAK PASS"),
    ]

    colors = {
        "PASS": C_PASS, "FAIL": C_FAIL, "FRAGILE PASS": C_WEAK,
        "WEAK PASS": C_WEAK, "UNDERPOWERED": "#95a5a6",
    }

    labels = [o[0] for o in obs]
    verdicts = [o[3] for o in obs]
    bars = [colors.get(v, C_ALT) for v in verdicts]

    # Bar height = 1 for pass-family, 0.5 for fail-family
    heights = [1.0 if v in ("PASS",) else 0.7 if "PASS" in v else 0.3 for v in verdicts]

    x = range(len(obs))
    rects = ax.bar(x, heights, color=bars, edgecolor="white", linewidth=2, width=0.7)

    for i, (label, val, target, verdict) in enumerate(obs):
        y_pos = heights[i] + 0.05
        ax.text(i, y_pos, verdict, ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=colors.get(verdict, "#333"))
        if val is not None:
            ax.text(i, heights[i] / 2, f"{val:+.3f}" if val < 0 else f"{val:.3f}",
                    ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        else:
            ax.text(i, heights[i] / 2, "n/a", ha="center", va="center",
                    fontsize=11, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.set_title("Pre-Registered Observable Outcomes", fontsize=14, fontweight="bold",
                 pad=15, color="#2c3e50")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_PASS, label="PASS"),
        Patch(facecolor=C_WEAK, label="FRAGILE / WEAK"),
        Patch(facecolor=C_FAIL, label="FAIL"),
        Patch(facecolor="#95a5a6", label="UNDERPOWERED"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def ablation_page(pdf):
    with open(ABLATION) as f:
        data = json.load(f)

    abl = data["ablation"]
    labels = [r["label"] for r in abl]
    rhos   = [r["rho_mean"] for r in abl]

    fig, ax = plt.subplots(figsize=(8.5, 5))

    colors = [C_BIO if l == "biological" else C_ALT for l in labels]
    bars = ax.barh(range(len(labels)), rhos, color=colors, edgecolor="white", linewidth=1.5)

    ax.axvline(x=0.85, color=C_FAIL, linestyle="--", linewidth=1.5, label="Target ρ ≥ 0.85")
    ax.axvline(x=0, color="#bdc3c7", linewidth=0.5)

    for i, (l, r) in enumerate(zip(labels, rhos)):
        ax.text(max(r + 0.02, 0.02), i, f"ρ = {r:.3f}", va="center", fontsize=9,
                fontweight="bold" if l == "biological" else "normal",
                color="#2c3e50" if r >= 0.85 else "#e74c3c")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l.replace("_", " ").title() for l in labels], fontsize=10)
    ax.set_xlim(-0.4, 1.15)
    ax.set_xlabel("Spearman ρ vs Biological Prior", fontsize=11)
    ax.set_title("Collapse-Ordering Ablation (8 conditions × 10 seeds)",
                 fontsize=13, fontweight="bold", color="#2c3e50", pad=15)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def scale_page(pdf):
    with open(SCALE) as f:
        data = json.load(f)

    ns     = [d["n_agents"] for d in data]
    obs3s  = [d["obs3"] for d in data]
    obs5s  = [d["obs5"] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.5))

    # Obs 3
    ax1.plot(ns, obs3s, "o-", color=C_M2, linewidth=2, markersize=8, label="M2 Spearman ρ")
    ax1.axhline(y=0.85, color=C_FAIL, linestyle="--", linewidth=1.5, label="Target ≥ 0.85")
    ax1.set_xlabel("Number of Agents", fontsize=10)
    ax1.set_ylabel("Spearman ρ", fontsize=10)
    ax1.set_title("Obs 3 — Collapse Order", fontsize=12, fontweight="bold", color="#2c3e50")
    ax1.set_xticks(ns)
    ax1.set_ylim(0.80, 0.92)
    ax1.legend(fontsize=8)
    for n, v in zip(ns, obs3s):
        ax1.annotate(f"{v:.3f}", (n, v), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    # Obs 5
    ax2.plot(ns, obs5s, "s-", color="#9b59b6", linewidth=2, markersize=8, label="M2 Pearson r")
    ax2.axhline(y=0.70, color=C_FAIL, linestyle="--", linewidth=1.5, label="Target ≥ 0.70")
    ax2.set_xlabel("Number of Agents", fontsize=10)
    ax2.set_ylabel("Pearson r", fontsize=10)
    ax2.set_title("Obs 5 — Character Stability", fontsize=12, fontweight="bold", color="#2c3e50")
    ax2.set_xticks(ns)
    ax2.set_ylim(0.60, 1.0)
    ax2.legend(fontsize=8)
    for n, v in zip(ns, obs5s):
        ax2.annotate(f"{v:.3f}", (n, v), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Scale Sensitivity (n = 32, 64, 128)", fontsize=13,
                 fontweight="bold", color="#2c3e50")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def downgrade_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.95
    ax.text(0.5, y, "Downgrade Verdict", fontsize=16, ha="center",
            fontweight="bold", color="#2c3e50")

    y -= 0.05
    ax.text(0.5, y, "Per registered downgrade tree (verdict_contract.md Part 3)",
            fontsize=10, ha="center", color="#7f8c8d")

    # Conditions table
    conditions = [
        ("battery_win (topology + counterfactual)", "YES", True),
        ("tier_a_gate (precedence > 0.70)", "YES (0.989)", True),
        ("obs1_pass (CV > 0.25)", "YES (2.118)", True),
        ("obs3_pass (Spearman ρ ≥ 0.85)", "YES (0.878)", True),
        ("social_signal_win", "NO (−0.363)", False),
    ]

    y -= 0.06
    for cond, val, met in conditions:
        color = C_PASS if met else C_FAIL
        marker = "●" if met else "○"
        ax.text(0.12, y, marker, fontsize=14, color=color, fontweight="bold")
        ax.text(0.17, y, cond, fontsize=10, color="#2c3e50")
        ax.text(0.82, y, val, fontsize=10, ha="right", color=color, fontweight="bold")
        y -= 0.04

    # Verdict box
    y -= 0.03
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.1, y - 0.025), 0.8, 0.05,
                         boxstyle="round,pad=0.01", facecolor="#eaf7ea",
                         edgecolor=C_PASS, linewidth=2)
    ax.add_patch(box)
    ax.text(0.5, y, "FULL_TIER_A at n=32 registered scale",
            fontsize=13, ha="center", fontweight="bold", color="#27ae60")

    # Narrowed claim
    y -= 0.07
    ax.text(0.5, y, "Publicly Supported Claim", fontsize=11, ha="center",
            fontweight="bold", color="#2c3e50")
    y -= 0.025
    ax.text(0.5, y, "(narrower than internal label)", fontsize=9, ha="center",
            color="#7f8c8d")
    y -= 0.04
    lines = [
        "A threshold-gated controller with the registered authored collapse",
        "ordering produces seed-stable ordered accessibility collapse under",
        "stress at the registered 32-agent scale.",
        "This is an architectural-specificity result.",
    ]
    for line in lines:
        ax.text(0.5, y, line, fontsize=10, ha="center", color="#555", style="italic")
        y -= 0.03

    # Not supported
    y -= 0.04
    ax.text(0.5, y, "Not Supported:", fontsize=11, ha="center",
            fontweight="bold", color=C_FAIL)
    y -= 0.035
    not_lines = [
        "Typed-family superiority over LSM on social legibility",
        "Semantic ontology claims beyond the threshold scaffold",
        "LM-deployment transfer claims",
    ]
    for line in not_lines:
        ax.text(0.5, y, f"•  {line}", fontsize=10, ha="center", color="#555")
        y -= 0.03

    # Three findings
    y -= 0.04
    ax.axhline(y=y, xmin=0.1, xmax=0.9, color="#bdc3c7", linewidth=0.5)
    y -= 0.04
    ax.text(0.5, y, "Three Findings", fontsize=12, ha="center",
            fontweight="bold", color="#2c3e50")
    y -= 0.04
    findings = [
        "1.  Ordered collapse topology — seed-stable, ablation-confirmed",
        "2.  Internal structure / external legibility trade-off — discovered, not resolved",
        "3.  Evaluation methodology — pre-registration + hostile baseline as template",
    ]
    for line in findings:
        ax.text(0.5, y, line, fontsize=10, ha="center", color="#555")
        y -= 0.035

    pdf.savefig(fig)
    plt.close(fig)


def links_page(pdf):
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.92, "Resources & Links", fontsize=16, ha="center",
            fontweight="bold", color="#2c3e50")

    links = [
        ("OSF Registration", OSF_URL),
        ("DOI", OSF_DOI),
        ("Source Code", REPO_URL),
        ("Raw Data", f"{REPO_URL}/tree/main/eval/results"),
        ("Ablation Results", f"{REPO_URL}/tree/main/eval/results/v5_ablation"),
        ("Scale Results", f"{REPO_URL}/tree/main/eval/results/v5_scale"),
        ("Verdict Contract", f"{REPO_URL}/blob/main/docs/verdict_contract.md"),
        ("Pre-Registration", f"{REPO_URL}/blob/main/docs/preregistration.md"),
    ]

    y = 0.82
    for label, url in links:
        ax.text(0.15, y, label, fontsize=11, color="#2c3e50", fontweight="bold")
        ax.text(0.15, y - 0.035, url, fontsize=8, color="#3498db")
        y -= 0.09

    pdf.savefig(fig)
    plt.close(fig)


def main():
    out = Path(__file__).parent.parent / "docs" / "OSF_Completion_Report.pdf"
    out.parent.mkdir(exist_ok=True)

    with PdfPages(str(out)) as pdf:
        title_page(pdf)
        observable_summary_page(pdf)
        ablation_page(pdf)
        scale_page(pdf)
        downgrade_page(pdf)
        links_page(pdf)

    print(f"PDF generated: {out}")
    print(f"  Pages: 6")
    print(f"  Size: {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
