#!/usr/bin/env python3
"""
acf_subset_plots.py

Plot ACF (target vs. its lags) for 100%, 75%, 50%, and 25% of a series.

Usage:
  python acf_subset_plots.py \
      --csv my_series.csv \
      --target-col target \
      --time-col timestamp \
      --nlags 40 \
      --diff 1 \
      --interpolate linear \
      --dropna after \
      --standardize \
      --outdir acf_subset_output

Notes:
- If --time-col is omitted, rows are treated as equally spaced.
- If --target-col is omitted, the first numeric column is used.
- ACF is computed with statsmodels (fft=True, alpha=0.05) like your reference.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# ---------------------
# Args
# ---------------------
def parse_args():
    p = argparse.ArgumentParser(description="ACF plots for progressive subsets of a target series.")
    p.add_argument("--csv", default = "data/KO.csv", help="Path to input CSV.")
    p.add_argument("--target-col", default="Close", help="Target column name. If None, first numeric column is used.")
    p.add_argument("--time-col", default=None, help="Optional time column to parse as datetime index.")
    p.add_argument("--nlags", type=int, default=40, help="Max number of lags for ACF.")
    p.add_argument("--diff", type=int, default=1, help="Order of differencing to apply (0 = none).")
    p.add_argument("--interpolate", default="none",
                   choices=["none", "linear", "time", "nearest", "spline3"],
                   help="Interpolate missing values before analysis.")
    p.add_argument("--dropna", default="after", choices=["before", "after", "none"],
                   help="When to drop NA rows relative to interpolation.")
    p.add_argument("--standardize", action="store_true",
                   help="Z-score the target after differencing.")
    p.add_argument("--outdir", default="acf_subset_output", help="Directory to save outputs.")
    return p.parse_args()

# ---------------------
# I/O and preprocessing
# ---------------------
def load_csv(path, time_col=None):
    df = pd.read_csv(path)
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.set_index(time_col).sort_index()
    return df

def pick_target(df, target_col):
    if target_col and target_col in df.columns:
        return target_col
    # choose first numeric column
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("No numeric columns found; please specify --target-col.")
    return num_cols[0]

def apply_missing_handling(s: pd.Series, interpolate="none", dropna="after"):
    x = s.copy()
    if dropna == "before":
        x = x.dropna()

    if interpolate != "none":
        if interpolate == "spline3":
            if isinstance(x.index, pd.DatetimeIndex):
                x = x.interpolate(method="time", limit_direction="both")
            else:
                x = x.interpolate(method="spline", order=3, limit_direction="both")
        else:
            x = x.interpolate(method=interpolate, limit_direction="both")

    if dropna == "after":
        x = x.dropna()
    return x

def difference_series(s: pd.Series, d: int):
    x = s.copy()
    for _ in range(d):
        x = x.diff()
    if d > 0:
        x = x.dropna()
    return x

def standardize(s: pd.Series):
    x = (s - s.mean()) / s.std(ddof=0)
    return x.replace([np.inf, -np.inf], np.nan).dropna()

# ---------------------
# Plotting
# ---------------------
def plot_acf_subsets(s: pd.Series, nlags: int, outdir: str, name: str):
    """
    Make a 2x2 grid of ACF plots for 100%, 75%, 50%, 25% of the series (head of series).
    Saves PNG and returns path.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Build subsets (use head proportion to simulate "less data available")
    fracs = [1.00, 0.75, 0.50, 0.25]
    subsets = []
    n = len(s)
    for f in fracs:
        k = max(5, int(np.floor(n * f)))  # keep at least a few points
        subsets.append((f, s.iloc[:k]))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    for ax, (f, x) in zip(axes, subsets):
        # Use the same acf computation path as statsmodels.plot_acf
        plot_acf(x.values, lags=nlags, fft=True, alpha=0.05, ax=ax)
        ax.set_title(f"ACF: {name} ({int(f*100)}% of data, n={len(x)})", fontsize=11)
        ax.grid(True, which="both", alpha=0.35)
        ax.set_axisbelow(True)

    fig.suptitle(f"Autocorrelation across data availability — {name}", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = os.path.join(outdir, f"{name}__ACF_subsets.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path

# ---------------------
# Main
# ---------------------
def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df = load_csv(args.csv, time_col=args.time_col)
    target = pick_target(df, args.target_col)

    # keep target as a Series
    s = df[target].copy()

    # missing handling
    s = apply_missing_handling(s, interpolate=args.interpolate, dropna=args.dropna)

    # differencing
    if args.diff > 0:
        s = difference_series(s, d=args.diff)

    # standardize
    if args.standardize:
        s = standardize(s)

    # guard: need enough length for nlags
    if len(s) < args.nlags + 5:
        raise ValueError(f"Series too short after preprocessing (len={len(s)}) for nlags={args.nlags}.")

    # Save processed target for reproducibility
    s.to_frame(name=target).to_csv(os.path.join(args.outdir, f"{target}__processed.csv"))

    # Plot ACF for 100/75/50/25%
    out_path = plot_acf_subsets(s, nlags=args.nlags, outdir=args.outdir, name=target)
    print(f"Saved ACF subset grid to: {os.path.abspath(out_path)}")
    print(f"Processed series saved to: {os.path.abspath(os.path.join(args.outdir, f'{target}__processed.csv'))}")

if __name__ == "__main__":
    main()
