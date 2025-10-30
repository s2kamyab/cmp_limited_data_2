# boxplot_and_radar.py
# Matplotlib-only visualizations comparing scratch vs pretrained models.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 0) Load your data
# -----------------------------
# Option A: load from CSV
# df = pd.read_csv("results.csv")

# Option B: construct a small example (replace with your real data)
df = pd.DataFrame({
    "Dataset": ["AAL","AAL","AAL","KO","KO","KO","BRKB","BRKB","BRKB"],
    "Model":   ["TimesNet","GPT2","LSTM","TimesNet","GPT2","LSTM","TimesNet","GPT2","LSTM"],
    "Scheme":  ["scratch","scratch","scratch","pretrained","pretrained","pretrained","pretrained","scratch","scratch"],
    "MAE":     [0.736,0.739,0.761,0.809,0.787,0.904,3.974,3.919,4.455],
    "R2":      [0.938,0.938,0.936,0.952,0.955,0.940,0.989,0.989,0.986],
    "SMAPE":   [4.57,4.65,4.77,1.48,1.44,1.65,1.49,1.48,1.67],
})

# Ensure Scheme is standardized
df["Scheme"] = df["Scheme"].str.lower().map({"from scratch":"scratch","scratch":"scratch",
                                             "pretrained":"pretrained"})

# Sanity: only keep rows that have both schemes per (Dataset,Model) if you want paired analysis
# (For plotting, unpaired is fine; for stats you may prefer paired.)
# paired = df.pivot_table(index=["Dataset","Model"], columns="Scheme", values=["MAE","R2","SMAPE"])
# df_paired = paired.dropna().stack("Scheme").reset_index()
# We'll just use df directly for plots.


# -----------------------------
# 1) Boxplots: scratch vs pretrained per metric
# -----------------------------
def boxplot_by_scheme(df, metrics=("MAE","R2","SMAPE"), figpath="boxplots.png"):
    schemes = ["scratch", "pretrained"]
    data_per_metric = []
    for m in metrics:
        data_per_metric.append([df.loc[df["Scheme"]==s, m].dropna().values for s in schemes])

    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4), constrained_layout=True)
    if len(metrics)==1:
        axes = [axes]

    for ax, m, boxes in zip(axes, metrics, data_per_metric):
        # boxes is [values_scratch, values_pretrained]
        bp = ax.boxplot(boxes, labels=["Scratch","Pretrained"], showmeans=True)
        ax.set_title(f"{m} by Scheme")
        ax.set_ylabel(m)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Performance Distribution: Scratch vs Pretrained (All datasets/models)")
    fig.savefig(figpath, dpi=200)
    print(f"Saved boxplots to {figpath}")


# -----------------------------
# 2) Radar (spider) plot: normalized medians per scheme
# -----------------------------
def minmax01(x):
    x = np.asarray(x, dtype=float)
    if np.all(~np.isfinite(x)) or np.nanmin(x)==np.nanmax(x):
        # avoid 0/0; return 0.5 defaults
        return (x*0)+0.5
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def normalized_scores(df):
    """
    Returns a dict: scheme -> {metric -> 0..1 score}, where higher = better.
    MAE/SMAPE are inverted (lower is better).
    R2 is used directly (higher is better), but minmax scaled to [0,1].
    We use median within each scheme after normalization per metric across all rows.
    """
    out = {}
    # Compute global min-max per metric (across both schemes)
    metrics = ["MAE","R2","SMAPE"]

    # Build normalized columns
    tmp = df.copy()
    # MAE, SMAPE: lower is better => score = 1 - minmax01(value)
    tmp["_MAE_score"]   = 1 - minmax01(tmp["MAE"])
    tmp["_SMAPE_score"] = 1 - minmax01(tmp["SMAPE"])
    # R2: higher is better => score = minmax01(value)
    tmp["_R2_score"]    = minmax01(tmp["R2"])

    for scheme, g in tmp.groupby("Scheme"):
        out[scheme] = {
            "MAE":   np.nanmedian(g["_MAE_score"].values),
            "R2":    np.nanmedian(g["_R2_score"].values),
            "SMAPE": np.nanmedian(g["_SMAPE_score"].values),
        }
    return out

def radar_plot(df, figpath="radar.png"):
    # Prepare scores
    scores = normalized_scores(df)  # scheme -> metric->score
    metrics = ["MAE","R2","SMAPE"]
    labels  = ["MAE (lower→better)", "R² (higher→better)", "SMAPE (lower→better)"]

    # Angles for each axis
    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])  # close the loop

    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)

    for scheme, color in zip(["scratch","pretrained"], [None, None]):  # no explicit colors to keep it simple
        if scheme not in scores: 
            continue
        vals = [scores[scheme][m] for m in metrics]
        vals = np.concatenate([vals, vals[:1]])  # close
        ax.plot(angles, vals, label=scheme.capitalize())
        ax.fill(angles, vals, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2,0.4,0.6,0.8])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8"])
    ax.set_ylim(0,1.0)
    ax.set_title("Normalized Median Performance (0–1, higher better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
    fig.tight_layout()
    fig.savefig(figpath, dpi=200)
    print(f"Saved radar plot to {figpath}")


# -----------------------------
# 3) (Optional) Quick paired statistics
# -----------------------------
def paired_summary(df):
    """
    Prints paired deltas (pretrained - scratch) per Dataset/Model for each metric.
    Useful for sanity-checking direction of change before plotting.
    """
    piv = df.pivot_table(index=["Dataset","Model"], columns="Scheme", values=["MAE","R2","SMAPE"])
    piv = piv.dropna()  # keep pairs
    for m in ["MAE","R2","SMAPE"]:
        if (m, "scratch") in piv.columns and (m, "pretrained") in piv.columns:
            delta = piv[(m,"pretrained")] - piv[(m,"scratch")]
            print(f"\nPaired delta (pretrained - scratch) for {m}:")
            print(f"  n={delta.shape[0]}, mean={delta.mean():.4g}, median={delta.median():.4g}")
            print(f"  25/75%: {np.nanpercentile(delta,25):.4g} / {np.nanpercentile(delta,75):.4g}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    paired_summary(df)
    boxplot_by_scheme(df, metrics=("MAE","R2","SMAPE"), figpath="boxplots.png")
    radar_plot(df, figpath="radar.png")
