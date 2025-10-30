#!/usr/bin/env python3
"""
ccf_to_target_max_only.py

Compute the maximum cross-correlation (by absolute value) between each
non-target variable and the target over lags [-L..+L], with optional
differencing and z-score standardization. No plots produced.

Usage:
  python ccf_to_target_max_only.py \
      --csv my_data.csv \
      --target-col y \
      --time-col timestamp \
      --max-lag 40 \
      --diff 1 \
      --interpolate linear \
      --dropna after \
      --standardize \
      --outdir ccf_output
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Max CCF (abs) of all variables vs. target.")
    p.add_argument("--csv", default="data\\BABA.csv",
                   help="Path to input CSV.")
    p.add_argument("--target-col", default="Close",
                   help="Target column name. If None, first numeric column is used.")
    p.add_argument("--time-col", default=None,
                   help="Optional time column to parse as datetime index.")
    p.add_argument("--max-lag", type=int, default=40,
                   help="Compute lags from -max_lag..+max_lag.")
    p.add_argument("--diff", type=int, default=1,
                   help="Order of differencing (0 = none).")
    p.add_argument("--interpolate", default="none",
                   choices=["none", "linear", "time", "nearest", "spline3"],
                   help="Interpolate missing values before analysis.")
    p.add_argument("--dropna", default="after",
                   choices=["before", "after", "none"],
                   help="When to drop NA rows relative to interpolation.")
    # Keeping your original flag behavior:
    p.add_argument("--standardize", default=True, action="store_true",
                   help="Z-score each series after differencing.")
    p.add_argument("--outdir", default="ccf_output",
                   help="Directory to save outputs (summary CSV, processed data).")
    return p.parse_args()


# ----------------------------
# I/O and preprocessing
# ----------------------------
def load_csv(path, time_col=None):
    df = pd.read_csv(path)
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.set_index(time_col).sort_index()
    return df

def pick_target(df, target_col):
    if target_col and target_col in df.columns:
        return target_col
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("No numeric columns found; please specify --target-col.")
    return num_cols[0]

def apply_missing_handling(df: pd.DataFrame, interpolate="none", dropna="after"):
    x = df.copy()
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

def difference_df(df: pd.DataFrame, d: int):
    x = df.copy()
    for _ in range(d):
        x = x.diff()
    if d > 0:
        x = x.dropna()
    return x

def zscore_df(df: pd.DataFrame):
    x = (df - df.mean()) / df.std(ddof=0)
    return x.replace([np.inf, -np.inf], np.nan).dropna()


# ----------------------------
# CCF computation
# ----------------------------
def ccf_series(a: pd.Series, b: pd.Series, max_lag: int):
    """
    Pearson correlation of a(t) with b(t - lag) for lag in [-max_lag..+max_lag].
    Positive lag => target (a) leads; negative lag => other series leads.
    Returns (lags, ccf_values).
    """
    lags = np.arange(0, max_lag + 1)
    vals = []
    for L in lags:
        if L < 0:
            a_shifted = a
            b_shifted = b.shift(-L)  # -L > 0
        else:
            a_shifted = a
            b_shifted = b.shift(L)
        aligned = pd.concat([a_shifted, b_shifted], axis=1, join="inner").dropna()
        if len(aligned) < 5:
            vals.append(np.nan)
            continue
        r = np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
        vals.append(r)
    return lags, np.asarray(vals, dtype=float)


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df0 = load_csv(args.csv, time_col=args.time_col)
    # df0 = df0.as_ty

    # Keep numeric columns + target if it's numeric
    # num_df = df0.select_dtypes(include=[np.number]).copy() ### uncomment for soshianest
    num_df = df0[['Open', 'High', 'Low', 'Close', 'Adj close', 'Volume']].copy() # uncomment for fnspid
    target = pick_target(df0, args.target_col)
    if target not in num_df.columns:
        raise ValueError(f"Target '{target}' is not numeric. Please choose a numeric target.")

    # Preprocess
    num_df = apply_missing_handling(num_df, interpolate=args.interpolate, dropna=args.dropna)
    if args.diff > 0:
        num_df = difference_df(num_df, d=args.diff)
    if args.standardize:
        num_df = zscore_df(num_df)

    # Guards
    if target not in num_df.columns:
        raise ValueError(f"After preprocessing, target '{target}' is missing.")
    if len(num_df) < max(10, args.max_lag * 2 + 5):
        raise ValueError(f"Not enough data after preprocessing (n={len(num_df)}).")

    # Save processed data snapshot (optional but useful for reproducibility)
    num_df.to_csv(outdir / "processed_numeric.csv")

    tgt = num_df[target].dropna()
    others = [c for c in num_df.columns if c != target]

    summary_rows = []
    for col in others:
        s = num_df[col].dropna()
        aligned = pd.concat([tgt, s], axis=1, join="inner").dropna()
        if len(aligned) < max(10, args.max_lag + 5):
            continue

        lags, vals = ccf_series(aligned.iloc[:, 0], aligned.iloc[:, 1], args.max_lag)

        if np.all(np.isnan(vals)):
            best_lag, best_val = np.nan, np.nan
            n_overlap = 0
        else:
            k = int(np.nanargmax(np.abs(vals)))
            best_lag, best_val = int(lags[k]), float(vals[k])
            n_overlap = int(aligned.shape[0])

        summary_rows.append({
            "series": col,
            "target": target,
            "best_lag": best_lag,
            "max_abs_ccf": best_val,
            "n_overlap_for_zero_lag": n_overlap
        })

    
    summary = pd.DataFrame(summary_rows,
                           columns=["series", "target", "best_lag", "max_abs_ccf", "n_overlap_for_zero_lag"])
    
   

    # Compute the average of 'max_abs_ccf'
    mean_value = summary["max_abs_ccf"].mean()

    # Create the new boolean column (1 if >= mean, else 0)
    summary["above_average"] = (summary["max_abs_ccf"] >= mean_value).astype(int)

    # Save to a new CSV (optional)
    summary.to_csv("your_file_with_flag.csv", index=False)

    print(f"Average max_abs_ccf = {mean_value:.4f}")
    print(summary.head())
    summary_path = outdir / "ccf_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Done. Max-CCF summary saved to: {summary_path.resolve()}")
    if not summary.empty:
        # Show top few strongest relationships
        show = summary.copy()
        show["abs_val"] = show["max_abs_ccf"].abs()
        show = show.sort_values("abs_val", ascending=False).drop(columns=["abs_val"])
        print(show.head().to_string(index=False))
    else:
        print("No comparable series found (check data length and preprocessing).")
    # --- Plot: ccf values vs feature index, with mean line and ±1 STD band ---
    # Use original order; drop NaNs only for plotting to keep indices aligned
    ccf = summary["max_abs_ccf"]
    mask = ~ccf.isna()
    idx = np.arange(len(ccf))                 # feature index
    std_val = ccf[mask].std(ddof=0)           # population std (ddof=0); use ddof=1 if you prefer sample std
    mean_val = ccf[mask].mean()

    plt.figure(figsize=(5,3))
    plt.plot(idx[mask], ccf[mask].values, marker="o", linestyle="-", label="max_ccf", linewidth=2)

    # Mean line
    plt.axhline(mean_val, linestyle="--", linewidth=2, color='orange', label=f"mean = {mean_val:.3f}")

    # Shaded band: mean ± 1 std
    upper = mean_val + std_val
    lower = mean_val - std_val
    # plt.fill_between(idx, lower, upper, alpha=0.2, label=f"±1 std (σ={std_val:.3f})")

    plt.xlabel("Feature index", fontsize=14)
    plt.ylabel("Max CCF (over 40 lags)", fontsize=14)
    # plt.title("max_abs_ccf across features with mean ± 1σ band", fontsize=16)
    plt.grid(True, which="both", alpha=0.35)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("ccf_values_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
