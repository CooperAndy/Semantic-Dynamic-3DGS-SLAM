from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "confidence_threshold_ablation.csv"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS)
expected = [0.5, 0.6, 0.7, 0.8, 0.9]
if df["threshold"].tolist() != expected:
    raise ValueError(f"expected sorted five-threshold S1 table: {expected}")
x = df["threshold"]

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)

panels = [
    ("full_rgb_psnr", "Full-image RGB PSNR (dB)", True),
    ("full_rgb_ms_ssim", "Full-image MS-SSIM", True),
    ("full_rgb_lpips", "Full-image LPIPS", False),
    ("common_depth_l1_all_valid_m", "Common-support depth L1 (m)", False),
]
for ax, (column, title, higher) in zip(axes.flat, panels):
    ax.plot(x, df[column], marker="o", linewidth=2.2, color="#2563eb")
    ax.set_title(title + (" higher is better" if higher else " lower is better"))
    ax.set_xlabel("Confidence threshold T")
    ax.set_xticks(x)
    best = df[column].idxmax() if higher else df[column].idxmin()
    ax.scatter([df.loc[best, "threshold"]], [df.loc[best, column]], s=75,
               color="#16a34a", zorder=3, label="best tested")
    ax.scatter([0.5], [df.loc[df["threshold"] == 0.5, column].iloc[0]], s=95,
               color="#ea580c", marker="*", zorder=4, label="T=0.50 later extension")
    ax.legend(frameon=False, loc="best")

fig.suptitle("S1 quality on identical evaluation support", fontsize=15, fontweight="bold")
for ext in ("png", "pdf"):
    fig.savefig(FIGURES / f"S1_common_support_quality.{ext}", dpi=220)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
tradeoffs = [
    ("eligible_ratio_total", "Eligible seed pixels / all pixels", 100.0, "%"),
    ("silhouette_coverage_common_depth", "Coverage on common depth support", 100.0, "%"),
    ("final_gaussians", "Final Gaussian count", 1e-6, "million"),
    ("mapping_time_s_per_mapped_frame", "Mapping time / mapped frame", 1.0, "seconds"),
]
colors = ["#7c3aed", "#0891b2", "#ea580c", "#475569"]
for ax, (column, title, scale, unit), color in zip(axes.flat, tradeoffs, colors):
    y = df[column] * scale
    ax.plot(x, y, marker="o", linewidth=2.2, color=color)
    ax.scatter([0.5], [y.iloc[0]], s=95, color="#ea580c", marker="*", zorder=4)
    ax.set_title(title)
    ax.set_xlabel("Confidence threshold T")
    ax.set_ylabel(unit)
    ax.set_xticks(x)

fig.suptitle("S1 coverage and mapping cost (orange star: later T=0.50)", fontsize=15, fontweight="bold")
for ext in ("png", "pdf"):
    fig.savefig(FIGURES / f"S1_cost_coverage_tradeoff.{ext}", dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
ax.plot(x, df["legacy_threshold_specific_psnr"], marker="o", linewidth=2.2,
        label="Legacy threshold-specific mask (not comparable)", color="#dc2626")
ax.plot(x, df["full_rgb_psnr"], marker="o", linewidth=2.2,
        label="Corrected full-image common support", color="#2563eb")
ax.set_title("Why the legacy protocol reverses the threshold ranking")
ax.set_xlabel("Confidence threshold T")
ax.set_ylabel("PSNR (dB)")
ax.set_xticks(x)
ax.axvline(0.5, color="#ea580c", linewidth=1.2, linestyle=":", label="T=0.50 later extension")
ax.legend(frameon=False)
for ext in ("png", "pdf"):
    fig.savefig(FIGURES / f"S1_legacy_vs_common_support.{ext}", dpi=220)
plt.close(fig)

print("wrote S1 final figures to", FIGURES)
