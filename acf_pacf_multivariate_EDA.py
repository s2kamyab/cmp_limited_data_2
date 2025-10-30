#!/usr/bin/env python3
"""
acf_pacf_multivariate.py

Usage:
  python acf_pacf_multivariate.py \
      --csv my_series.csv \
      --time-col timestamp \
      --nlags 60 \
      --diff 0 \
      --interpolate linear \
      --dropna after \
      --ccf-scan \
      --max-ccf-lag 40 \
      --outdir acf_pacf_output

If your CSV has no explicit time column, omit --time-col and rows will be treated as equally spaced.
"""

import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf, ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def parse_args():
    p = argparse.ArgumentParser(description="ACF/PACF analysis for multivariate time series (CSV).")
    p.add_argument("--csv", default ="data\\aal.csv", help="Path to input CSV.")
    p.add_argument("--time-col", default=None, help="Optional time column name to parse as DateTime index.")
    p.add_argument("--nlags", type=int, default=40, help="Number of lags for ACF/PACF.")
    p.add_argument("--diff", type=int, default=1, help="Order of differencing to apply to each series (e.g., 1).")
    p.add_argument("--interpolate", default="linear",
                   choices=["none", "linear", "time", "nearest", "spline3"],
                   help="Interpolate missing values before analysis.")
    p.add_argument("--dropna", default="after", choices=["before", "after", "none"],
                   help="When to drop NA rows relative to interpolation.")
    p.add_argument("--standardize", default=True, action="store_true",
                   help="Z-score each column before analysis (applied after differencing).")
    p.add_argument("--ccf-scan", action="store_true",
                   help="Also compute pairwise cross-correlation (CCF) summary.")
    p.add_argument("--max-ccf-lag", type=int, default=40, help="Lag window for CCF scan (both +/-).")
    p.add_argument("--outdir", default="acf_pacf_output", help="Output directory.")
    return p.parse_args()

def load_csv(path, time_col=None):
    df = pd.read_csv(path)
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.set_index(time_col).sort_index()
    # Keep only numeric columns
    num_df = df.select_dtypes(include=[np.number]).copy()
    return num_df

def apply_missing_handling(df, interpolate="none", dropna="after"):
    x = df.copy()
    if dropna == "before":
        x = x.dropna()

    if interpolate != "none":
        if interpolate == "spline3":
            # Spline requires numeric index spacing; if DateTimeIndex, convert to ordinal
            if isinstance(x.index, pd.DatetimeIndex):
                # Use time-based interpolation
                x = x.interpolate(method="time", limit_direction="both")
            else:
                x = x.interpolate(method="spline", order=3, limit_direction="both")
        else:
            x = x.interpolate(method=interpolate, limit_direction="both")

    if dropna == "after":
        x = x.dropna()

    return x

def difference_df(df, d=0):
    x = df.copy()
    for _ in range(d):
        x = x.diff()
    if d > 0:
        x = x.dropna()
    return x

def standardize_df(df):
    x = (df - df.mean()) / df.std(ddof=0)
    return x.replace([np.inf, -np.inf], np.nan).dropna()

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def signif_lags_from_confint(values, confint_arr):
    """
    Return lags where 0 lies outside the confint interval.
    values shape: (nlags+1,)
    confint_arr shape: (nlags+1, 2)
    """
    sig = []
    for lag in range(1, len(values)):  # skip lag 0
        lo, hi = confint_arr[lag]
        if lo > 0 or hi < 0:
            sig.append((lag, values[lag], lo, hi))
    return sig

def save_significant_table(sig_list, path, header=("series", "lag", "value", "lo", "hi", "kind")):
    if not sig_list:
        pd.DataFrame(columns=header).to_csv(path, index=False)
        return
    df = pd.DataFrame(sig_list, columns=header)
    df.to_csv(path, index=False)

def plot_and_save_acf_pacf(series_name, x, nlags, outdir):
    fig1 = plot_acf(x, lags=nlags, title=f"ACF: {series_name}")
    fig1.figure.tight_layout()
    fig1.figure.savefig(os.path.join(outdir, f"{series_name}__ACF.png"), dpi=180)
    plt.close(fig1.figure)

    fig2 = plot_pacf(x, lags=nlags, method="ywmle", title=f"PACF: {series_name}")
    fig2.figure.tight_layout()
    fig2.figure.savefig(os.path.join(outdir, f"{series_name}__PACF.png"), dpi=180)
    plt.close(fig2.figure)

def compute_and_collect_significant(x, nlags, series_name):
    # Raw arrays (for table output)
    acf_vals, acf_ci = acf(x, nlags=nlags, alpha=0.05, fft=True)
    pacf_vals, pacf_ci = pacf(x, nlags=nlags, method="ywmle", alpha=0.05)

    acf_sig = signif_lags_from_confint(acf_vals, acf_ci)
    pacf_sig = signif_lags_from_confint(pacf_vals, pacf_ci)

    acf_rows = [(series_name, lag, val, lo, hi, "ACF") for (lag, val, lo, hi) in acf_sig]
    pacf_rows = [(series_name, lag, val, lo, hi, "PACF") for (lag, val, lo, hi) in pacf_sig]
    return acf_rows, pacf_rows

def scan_pairwise_ccf(df, max_lag, outdir):
    """
    For each pair (A,B), compute CCF at lags -max_lag..+max_lag by shifting B.
    We'll report lag* at which |CCF| is maximized, and the value.
    """
    results = []
    cols = df.columns.tolist()
    for a, b in combinations(cols, 2):
        xa = df[a].dropna()
        xb = df[b].dropna()
        # align indices
        x = pd.concat([xa, xb], axis=1, join="inner").dropna()
        if len(x) < max(50, max_lag * 2 + 5):
            continue  # not enough data
        A = (x.iloc[:, 0] - x.iloc[:, 0].mean()) / x.iloc[:, 0].std(ddof=0)
        B = (x.iloc[:, 1] - x.iloc[:, 1].mean()) / x.iloc[:, 1].std(ddof=0)

        # Compute CCF at non-negative lags directly; we'll mirror for negative via swapping/shifts.
        ccf_vals = ccf(A, B)[: max_lag + 1]  # lags 0..+max_lag for A leading B
        # For negative lags, compute ccf(B,A) (equivalent to shifting A)
        ccf_vals_neg = ccf(B, A)[: max_lag + 1]  # lags 0..+max_lag => we will map to -lag
        # Build full lag vector
        lags = list(range(-max_lag, 0)) + list(range(0, max_lag + 1))
        vals = list(reversed(ccf_vals_neg[1:])) + list(ccf_vals)  # skip duplicate zero from neg side

        vals = np.array(vals)
        lags = np.array(lags)
        k = int(np.argmax(np.abs(vals)))
        results.append({
            "series_A": a,
            "series_B": b,
            "best_lag": int(lags[k]),
            "ccf_at_best_lag": float(vals[k]),
        })

        # Optional: save a quick plot
        plt.figure()
        plt.stem(lags, vals, use_line_collection=True)
        plt.title(f"CCF {a} vs {b} (best lag={lags[k]}, val={vals[k]:.3f})")
        plt.xlabel("Lag (A leads +)")
        plt.ylabel("CCF")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"CCF__{a}__vs__{b}.png"), dpi=160)
        plt.close()

    if results:
        ccf_df = pd.DataFrame(results)
    else:
        ccf_df = pd.DataFrame(columns=["series_A", "series_B", "best_lag", "ccf_at_best_lag"])
    ccf_path = os.path.join(outdir, "ccf_scan_summary.csv")
    ccf_df.to_csv(ccf_path, index=False)

    # Heatmap of |best CCF| by pair (symmetric)
    if not ccf_df.empty:
        labels = sorted(set(ccf_df["series_A"]).union(set(ccf_df["series_B"])))
        mat = pd.DataFrame(0.0, index=labels, columns=labels)
        for _, r in ccf_df.iterrows():
            a, b, v = r["series_A"], r["series_B"], abs(r["ccf_at_best_lag"])
            mat.loc[a, b] = v
            mat.loc[b, a] = v
        plt.figure(figsize=(max(4, len(labels) * 0.6), max(4, len(labels) * 0.6)))
        plt.imshow(mat.values, aspect='auto')
        plt.colorbar(label="|best CCF|")
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha="right")
        plt.yticks(ticks=np.arange(len(labels)), labels=labels)
        plt.title("Pairwise |best CCF| heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "ccf_best_abs_heatmap.png"), dpi=160)
        plt.close()

def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    df = load_csv(args.csv, time_col=args.time_col)
    # df = df[['OT', 'date']]
    if df.empty:
        raise ValueError("No numeric columns found in the CSV.")

    # Handle missing data
    df = apply_missing_handling(df, interpolate=args.interpolate, dropna=args.dropna)

    # Differencing (helps stationarity if needed)
    if args.diff > 0:
        df = difference_df(df, d=args.diff)

    # Optional standardization (useful across mixed-scale variables)
    if args.standardize:
        df = standardize_df(df)

    # Save a clean copy of the processed data
    df.to_csv(os.path.join(args.outdir, "processed_series.csv"))

    # Compute & plot per-series ACF/PACF
    acf_sig_all, pacf_sig_all = [], []
    for col in df.columns:
        x = df[col].dropna().values
        if len(x) < args.nlags + 5:
            print(f"[WARN] Skipping {col}: not enough data for nlags={args.nlags}.")
            continue

        # Plots
        plot_and_save_acf_pacf(col, x, args.nlags, args.outdir)

        # Significant lags tables
        acf_rows, pacf_rows = compute_and_collect_significant(x, args.nlags, col)
        acf_sig_all.extend(acf_rows)
        pacf_sig_all.extend(pacf_rows)

    # Save significance summaries
    save_significant_table(
        acf_sig_all,
        os.path.join(args.outdir, "significant_lags_ACF.csv"),
        header=("series", "lag", "acf", "lo", "hi", "kind"),
    )
    save_significant_table(
        pacf_sig_all,
        os.path.join(args.outdir, "significant_lags_PACF.csv"),
        header=("series", "lag", "pacf", "lo", "hi", "kind"),
    )

    # Optional CCF scan
    if args.ccf_scan and df.shape[1] >= 2:
        scan_pairwise_ccf(df, args.max_ccf_lag, args.outdir)

    print(f"Done. Outputs saved in: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()

