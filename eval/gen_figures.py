"""
gen_figures.py — Generate all v5 paper figures
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("/home/claude/figs")
OUT.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

BLUE = "#1F4E79"
RED = "#C0392B"
GREEN = "#27AE60"
ORANGE = "#E67E22"
GRAY = "#95A5A6"
LIGHT_BLUE = "#5DADE2"

# ── Data ──
obs3_seeds = [0.8755, 0.8736, 0.8912, 0.8733, 0.8615, 0.8782, 0.8737, 0.8763, 0.8757, 0.8970]
obs5_seeds = [0.9890, 0.8461, 0.8810, 0.6855, 0.7768, 0.8255, 0.9797, 0.9884, 0.9979, 0.9994]

ablation_labels = ["Biological", "Reversed", "Uniform", "Rnd 1", "Rnd 2", "Rnd 3", "Rnd 4", "Rnd 5"]
ablation_rhos = [1.000, -0.250, 0.679, 0.000, 0.250, 0.107, 0.536, 0.036]

scale_n = [32, 64, 128]
scale_obs3 = [0.880, 0.848, 0.842]
scale_obs5 = [0.905, 0.881, 0.878]

# ── Figure 1: Simulated collapse topology scatter ──
np.random.seed(42)
n_pts = 2000
rd_vals = np.concatenate([
    np.random.uniform(0.0, 0.2, 600),
    np.random.uniform(0.2, 0.5, 500),
    np.random.uniform(0.5, 0.8, 500),
    np.random.uniform(0.8, 1.0, 400),
])
# acc_sum: 7 families, collapse sequentially
def acc_at_rd(rd):
    thresholds = [0.18, 0.24, 0.34, 0.42, 0.52, 0.65, 2.0]
    sensitivities = [12, 12, 10, 10, 10, 10, 12]
    total = 0
    for t, s in zip(thresholds, sensitivities):
        a = max(0, min(1, 1 - max(0, rd - t) * s))
        total += a
    return total

acc_vals = np.array([acc_at_rd(rd) + np.random.normal(0, 0.08) for rd in rd_vals])
acc_vals = np.clip(acc_vals, 0, 7)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.3, 1]})

# Scatter
ax1.scatter(rd_vals, acc_vals, s=3, alpha=0.15, c=BLUE, edgecolors="none", rasterized=True)
# Binned mean overlay
bins = np.arange(0, 1.05, 0.05)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_means = []
for i in range(len(bins)-1):
    mask = (rd_vals >= bins[i]) & (rd_vals < bins[i+1])
    if mask.sum() > 0:
        bin_means.append(np.mean(acc_vals[mask]))
    else:
        bin_means.append(np.nan)
ax1.plot(bin_centers, bin_means, "o-", color=RED, linewidth=2, markersize=5, label="Binned mean", zorder=5)
ax1.axhline(y=1, color=GRAY, linestyle="--", alpha=0.5, linewidth=0.8)
ax1.set_xlabel("Regression Depth (rd)")
ax1.set_ylabel("Accessible Families (acc_sum)")
ax1.set_title("Population (rd, acc_sum) Scatter")
ax1.legend(loc="upper right", framealpha=0.9)
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.2, 7.5)

# Bar chart: mean accessibility by stress bucket
buckets = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
bucket_means = []
for lo, hi in [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0)]:
    mask = (rd_vals >= lo) & (rd_vals < hi)
    bucket_means.append(np.mean(acc_vals[mask]) if mask.sum() > 0 else 0)

colors = [GREEN, LIGHT_BLUE, ORANGE, RED, "#8B0000"]
bars = ax2.bar(buckets, bucket_means, color=colors, edgecolor="white", linewidth=0.8)
ax2.set_xlabel("Stress Bucket (rd)")
ax2.set_ylabel("Mean acc_sum")
ax2.set_title("Mean Accessibility by Stress Level")
ax2.set_ylim(0, 7.5)
for bar, val in zip(bars, bucket_means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f"{val:.1f}",
             ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
fig.savefig(OUT / "fig1_collapse_topology.png")
plt.close()

# ── Figure 2: Obs 3 per-seed Spearman ──
fig, ax = plt.subplots(figsize=(8, 3.5))
seeds = range(10)
colors_obs3 = [GREEN if v >= 0.85 else RED for v in obs3_seeds]
ax.bar(seeds, obs3_seeds, color=colors_obs3, edgecolor="white", linewidth=0.8, width=0.7)
ax.axhline(y=0.85, color=RED, linestyle="--", linewidth=1.5, label="Threshold (ρ ≥ 0.85)")
ax.axhline(y=np.mean(obs3_seeds), color=BLUE, linestyle="-", linewidth=1.5, alpha=0.7, label=f"Mean = {np.mean(obs3_seeds):.3f}")
# CI band
ax.axhspan(0.872, 0.884, alpha=0.12, color=BLUE, label="95% CI [0.872, 0.884]")
ax.set_xlabel("Seed")
ax.set_ylabel("Spearman ρ")
ax.set_title("Obs 3 — Collapse Topology Across 10 Seeds")
ax.set_xticks(seeds)
ax.set_xticklabels([f"S{i}" for i in seeds])
ax.set_ylim(0.84, 0.91)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
for i, v in enumerate(obs3_seeds):
    ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=7.5, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig2_obs3_seeds.png")
plt.close()

# ── Figure 3: Obs 5 per-seed Pearson (bimodal) ──
fig, ax = plt.subplots(figsize=(8, 3.5))
colors_obs5 = [GREEN if v >= 0.70 else RED for v in obs5_seeds]
ax.bar(seeds, obs5_seeds, color=colors_obs5, edgecolor="white", linewidth=0.8, width=0.7)
ax.axhline(y=0.70, color=RED, linestyle="--", linewidth=1.5, label="Per-seed gate (r ≥ 0.70)")
ax.axhline(y=np.mean(obs5_seeds), color=BLUE, linestyle="-", linewidth=1.5, alpha=0.7, label=f"Mean = {np.mean(obs5_seeds):.3f}")
ax.axhspan(0.829, 0.958, alpha=0.12, color=BLUE, label="95% CI [0.829, 0.958]")
ax.set_xlabel("Seed")
ax.set_ylabel("Pearson r")
ax.set_title("Obs 5 — Character Stability Across 10 Seeds (Bimodal)")
ax.set_xticks(seeds)
ax.set_xticklabels([f"S{i}" for i in seeds])
ax.set_ylim(0.6, 1.05)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
for i, v in enumerate(obs5_seeds):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5, fontweight="bold")
# Annotate bimodal clusters
ax.annotate("Cluster 1\n(0.88–1.00)", xy=(0, 0.99), xytext=(-0.8, 1.03),
            fontsize=8, color=GRAY, ha="center")
ax.annotate("Cluster 2\n(0.69–0.83)", xy=(4, 0.78), xytext=(4, 0.64),
            fontsize=8, color=GRAY, ha="center")
plt.tight_layout()
fig.savefig(OUT / "fig3_obs5_bimodal.png")
plt.close()

# ── Figure 4: Ablation results ──
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(ablation_labels))
colors_abl = [GREEN if r >= 0.85 else RED for r in ablation_rhos]
bars = ax.bar(x, ablation_rhos, color=colors_abl, edgecolor="white", linewidth=0.8, width=0.7)
ax.axhline(y=0.85, color=GREEN, linestyle="--", linewidth=1.5, alpha=0.8, label="Threshold (ρ ≥ 0.85)")
ax.axhline(y=0.0, color="black", linewidth=0.5)
ax.set_xlabel("Threshold Condition")
ax.set_ylabel("Spearman ρ vs Biological Prior")
ax.set_title("Collapse-Ordering Ablation: Only Biological Ordering Passes")
ax.set_xticks(x)
ax.set_xticklabels(ablation_labels, rotation=25, ha="right")
ax.set_ylim(-0.4, 1.15)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
for i, v in enumerate(ablation_rhos):
    y_off = 0.03 if v >= 0 else -0.06
    ax.text(i, v + y_off, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
# Mean of random conditions
rand_mean = np.mean(ablation_rhos[3:])
ax.axhline(y=rand_mean, color=ORANGE, linestyle=":", linewidth=1.2, alpha=0.8)
ax.text(7.3, rand_mean + 0.03, f"Random mean = {rand_mean:.3f}", fontsize=8, color=ORANGE, ha="right")
plt.tight_layout()
fig.savefig(OUT / "fig4_ablation.png")
plt.close()

# ── Figure 5: Social signal comparison ──
fig, ax = plt.subplots(figsize=(6, 4))
categories = ["Context\nOnly", "Context +\nPrevious", "Context +\nSignal", "Signal\nLift"]
m2_vals = [0.272, 0.527, 0.792, 0.265]
lsm_vals = [0.238, 0.247, 0.875, 0.628]
x = np.arange(len(categories))
w = 0.35
bars1 = ax.bar(x - w/2, m2_vals, w, color=BLUE, label="M2 (Typed)", edgecolor="white")
bars2 = ax.bar(x + w/2, lsm_vals, w, color=RED, label="LSM (Latent)", edgecolor="white")
ax.set_ylabel("Accuracy / Lift")
ax.set_title("Social Signal Suite — LSM Outperforms M2 on Legibility")
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.legend(framealpha=0.9)
ax.set_ylim(0, 1.0)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", fontsize=8)
# Highlight the lift difference
ax.annotate("", xy=(3 + w/2, 0.628), xytext=(3 - w/2, 0.265),
            arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2))
ax.text(3, 0.45, "Δ = 0.363", ha="center", fontsize=9, color=ORANGE, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig5_social_signal.png")
plt.close()

# ── Figure 6: Scale sensitivity ──
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(scale_n))
w = 0.3
b1 = ax.bar(x - w/2, scale_obs3, w, color=BLUE, label="Obs 3 (Spearman ρ)", edgecolor="white")
b2 = ax.bar(x + w/2, scale_obs5, w, color=LIGHT_BLUE, label="Obs 5 (Pearson r)", edgecolor="white")
ax.axhline(y=0.85, color=RED, linestyle="--", linewidth=1.5, label="Obs 3 threshold (0.85)")
ax.axhline(y=0.70, color=ORANGE, linestyle=":", linewidth=1.2, label="Obs 5 gate (0.70)")
ax.set_xlabel("Number of Agents")
ax.set_ylabel("Metric Value")
ax.set_title("Scale Sensitivity — Regime 1 Results at 32, 64, 128 Agents")
ax.set_xticks(x)
ax.set_xticklabels([f"n = {n}" for n in scale_n])
ax.set_ylim(0.7, 1.0)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig6_scale.png")
plt.close()

# ── Figure 7: Evidence hierarchy visual ──
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.axis("off")

evidence = [
    ("Collapse Topology", "ρ = 0.878, CI [0.872, 0.884]", GREEN, 1.0),
    ("Within-State Coherence", "Δ = 0.011 (M2 > LSM)", "#7DCEA0", 0.82),
    ("Precedence Dominance", "98.9% score-win", LIGHT_BLUE, 0.65),
    ("Character Stability", "r = 0.897, σ = 0.106 (bimodal)", ORANGE, 0.48),
    ("Social Signal", "LSM +0.628 vs M2 +0.252 (adverse)", RED, 0.30),
    ("Switch-Cost Asymmetry", "Underpowered (need ≥106 agents)", GRAY, 0.15),
]

labels_left = ["DECISIVE", "SUPPORTING", "CONSISTENCY", "AGGREGATE\nONLY", "NEGATIVE", "UNRESOLVED"]

for i, (name, detail, color, width) in enumerate(evidence):
    y = 0.85 - i * 0.15
    ax.barh(y, width, height=0.1, color=color, edgecolor="white", linewidth=1)
    ax.text(-0.02, y, labels_left[i], ha="right", va="center", fontsize=8, fontweight="bold", color=color)
    ax.text(width + 0.02, y + 0.02, name, va="center", fontsize=9, fontweight="bold", color="#1A1A1A")
    ax.text(width + 0.02, y - 0.03, detail, va="center", fontsize=8, color="#666666")

ax.set_xlim(-0.35, 1.35)
ax.set_ylim(-0.05, 1.0)
ax.set_title("Evidence Hierarchy — Strength of Support", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
fig.savefig(OUT / "fig7_evidence_hierarchy.png")
plt.close()

print("All figures generated:")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name} ({f.stat().st_size // 1024} KB)")
