import pandas as pd
# from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
def adf_test(df: pd.DataFrame, column: str, alpha: float = 0.05,
             regression: str = "c", autolag: str | None = "AIC"):
    """
    Run Augmented Dickey–Fuller on a single DataFrame column.

    Parameters
    ----------
    df : pandas.DataFrame
    column : int              # target column index
    alpha : float             # significance level
    regression : {"c","ct","ctt","n"}
    autolag : {"AIC","BIC","t-stat",None}

    Returns
    -------
    dict with keys:
      test_stat, pvalue, usedlag, nobs, critical_values, is_stationary
    """
    s = pd.to_numeric(df.iloc[:, column], errors="coerce").dropna()

    test_stat, pvalue, usedlag, nobs, crit_vals, _ = adfuller(
        s.values, regression=regression, autolag=autolag
    )
    return {
        "test_stat": test_stat,
        "pvalue": pvalue,
        "usedlag": usedlag,
        "nobs": nobs,
        "critical_values": crit_vals,   # dict with "1%","5%","10%"
        "is_stationary": pvalue < alpha
    }

def train_test_split_time_series(df, test_size=0.1):
    """
    Splits a time series dataframe into train and test sets by time order.
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


_DATASETS = [
    # --- Soshianest (target = 'OT', sentiment col = 'Sentiment_textblob') ---
    {"names": ["soshianest_530486"], "path": r"archive\\data\\530486_dataset.csv",
     "target": "OT", "drop_cols": [], "sentiment_cols": ["Sentiment_textblob"]},
    {"names": ["soshianest_530501"], "path": r"archive\\data\\530501_dataset.csv",
     "target": "OT", "drop_cols": [], "sentiment_cols": ["Sentiment_textblob"]},
    {"names": ["soshianest_5627"],  "path": r"archive\\data\\5627_dataset.csv",
     "target": "OT", "drop_cols": [], "sentiment_cols": ["Sentiment_textblob"]},
    {"names": ["soshianest_549324"], "path": r"archive\\data\\549324_dataset.csv",
     "target": "OT", "drop_cols": [], "sentiment_cols": ["Sentiment_textblob"]},

    # --- Finance (target = 'Close', sentiment cols = 'Scaled_sentiment', 'Sentiment_gpt') ---
    {"names": ["fin_aal", "aal.csv"], "path": r"Github\\data\\aal.csv",
     "target": "Close", "drop_cols": ["Date", "News_flag"],
     "sentiment_cols": ["Scaled_sentiment", "Sentiment_gpt"]},

    {"names": ["fin_aapl", "AAPL.csv"], "path": r"Github\\data\\AAPL.csv",
     "target": "Close", "drop_cols": ["Date", "News_flag"],
     "sentiment_cols": ["Scaled_sentiment", "Sentiment_gpt"]},

    {"names": ["fin_abbv", "ABBV.csv"], "path": r"Github\\data\\ABBV.csv",
     "target": "Close", "drop_cols": ["Date", "News_flag"],
     "sentiment_cols": ["Scaled_sentiment", "Sentiment_gpt"]},

    {"names": ["fin_amd", "AMD.csv"], "path": r"Github\\data\\AMD.csv",
     "target": "Close", "drop_cols": ["Date", "News_flag"],
     "sentiment_cols": ["Scaled_sentiment", "Sentiment_gpt"]},

    {"names": ["fin_ko", "KO.csv"], "path": r"Github\\data\\KO.csv",
     "target": "Close", "drop_cols": ["Date", "News_flag"],
     "sentiment_cols": ["Scaled_sentiment", "Sentiment_gpt"]},

    {"names": ["fin_TSM", "TSM.csv"], "path": r"Github\\data\\TSM.csv",
     "target": "Close", "drop_cols": ["Date", "News_flag"],
     "sentiment_cols": ["Scaled_sentiment", "Sentiment_gpt"]},

    {"names": ["goog", "GOOG.csv"], "path": r"Github\\data\\GOOG.csv",
     "target": "Close", "drop_cols": ["Date", "News_flag"],
     "sentiment_cols": ["Scaled_sentiment", "Sentiment_gpt"]},

    {"names": ["fin_wmt", "WMT.csv"], "path": r"Github\\data\\WMT.csv",
     "target": "Close", "drop_cols": ["Date", "News_flag"],
     "sentiment_cols": ["Scaled_sentiment", "Sentiment_gpt"]},

    {"names": ["fin_amzn", "AMZN.csv"], "path": r"Github\\data\\AMZN.csv",
     "target": "Close", "drop_cols": ["Date", "News_flag"],
     "sentiment_cols": ["Scaled_sentiment", "Sentiment_gpt"]},

    {"names": ["fin_baba", "BABA.csv"], "path": r"Github\\data\\BABA.csv",
     "target": "Close", "drop_cols": ["Date", "News_flag"],
     "sentiment_cols": ["Scaled_sentiment", "Sentiment_gpt"]},

    {"names": ["fin_brkb", "BRK-B.csv"], "path": r"Github\\data\\BRK-B.csv",
     "target": "Close", "drop_cols": ["Date", "News_flag"],
     "sentiment_cols": ["Scaled_sentiment", "Sentiment_gpt"]},

    {"names": ["fin_cost", "COST.csv"], "path": r"Github\\data\\COST.csv",
     "target": "Close", "drop_cols": ["Date", "News_flag"],
     "sentiment_cols": ["Scaled_sentiment", "Sentiment_gpt"]},

    {"names": ["fin_ebay", "ebay.csv"], "path": r"Github\\data\\ebay.csv",
     "target": "Close", "drop_cols": ["Date", "News_flag"],
     "sentiment_cols": ["Scaled_sentiment", "Sentiment_gpt"]},
]

def _match_config(dataset: str):
    for cfg in _DATASETS:
        if dataset in cfg["names"]:
            return cfg
    raise ValueError(f"Unknown dataset key: {dataset}")

def load_data(dataset: str, use_sentiment: int = 0):
    """
    Load a dataset, add a time_step, drop date/news cols,
    optionally include/shift sentiment columns, and split.

    Returns
    -------
    train_df, test_df, target_index
    """
    cfg = _match_config(dataset)
    df = pd.read_csv(cfg["path"])

    # Always add a time_step index
    df["time_step"] = range(len(df))

    # Drop columns if they exist
    for c in cfg.get("drop_cols", []):
        if c in df.columns:
            df = df.drop(columns=c)

    # Sentiment handling
    for sc in cfg.get("sentiment_cols", []):
        if sc in df.columns:
            if use_sentiment == 0:
                df = df.drop(columns=sc)
            else:
                df[sc] = df[sc].shift(use_sentiment).fillna(0)

    # Target index
    target_index = df.columns.to_list().index(cfg["target"])

    # Your own splitter (kept identical to your original call)
    train1, test1 = train_test_split_time_series(df)
    return train1, test1, target_index


import matplotlib.pyplot as plt

def plot_one(train1, test1 , target_index,ds_key: str, title: str, use_sentiment: int = 0):
    # train1, test1, target_index = load_data(ds_key, use_sentiment=use_sentiment)
    ax = train1.iloc[:, target_index].plot(figsize=(8,4), linewidth=2.5, label="train")
    test1.iloc[:, target_index].plot(ax=ax, linewidth=2.5, color="k", label="test")
    ax.grid(True)
    ax.tick_params(labelsize=16)      # both x and y
    ax.set_xlabel("Time Step", fontsize=16)
    ax.set_ylabel("Value", fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.legend()

datasets = [
    ("soshianest_530501", "Clarckson 530501"),
    ("soshianest_530486", "Clarckson 530486"),
    ("soshianest_5627",   "Clarckson 5627"),
    ("soshianest_549324", "Clarckson 549324"),
    ("fin_aal",  "FINSPD AAL"),
    ("fin_aapl", "FINSPD AAPL"),
    ("fin_abbv", "FINSPD ABBV"),
    ("fin_amzn", "FINSPD AMZN"),
    ("fin_baba", "FINSPD BABA"),
    ("fin_brkb", "FINSPD BRKB"),
    ("fin_amd",  "FINSPD AMD"),
    ("fin_ko",   "FINSPD KO"),
    # Use the exact keys your load_data expects (e.g., "fin_TSM" vs "fin_tsm", "goog" vs "fin_goog")
    ("fin_TSM",  "FINSPD TSM"),
    ("goog",     "FINSPD GOOG"),
    ("fin_wmt",  "FINSPD WMT"),
]
# vix original series
# for key, title in datasets:
#     train1, test1, target_index = load_data(key, use_sentiment= 0)
#     result = adf_test(train1, target_index, alpha=0.05, regression="c", autolag="AIC")
#     print(result)
#     if result["is_stationary"]:
#         print(f"{title} Stationary at alpha=0.05")
#     else:
#         print(f"{title} Non-stationary at alpha=0.05")
#     plot_one(train1, test1, target_index, key, title)
#     plt.show()




######################################################################################
# Fourier Transform Example
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Generate long and short sine waves
fs = 100  # samples/sec
T_long, T_short = 10, 1  # seconds
f_signal = 5  # Hz

t_long = np.linspace(0, T_long, int(fs*T_long), endpoint=False)
t_short = np.linspace(0, T_short, int(fs*T_short), endpoint=False)
x_long = np.sin(2*np.pi*f_signal*t_long)
x_short = np.sin(2*np.pi*f_signal*t_short)

# Fourier Transforms
for t, x, title in [(t_long, x_long, 'Long Series'), (t_short, x_short, 'Short Series')]:
    N = len(x)
    X = np.abs(fft(x))[:N//2]
    freqs = fftfreq(N, 1/fs)[:N//2]
    plt.plot(freqs, X, label=title)

plt.legend(); plt.title("Effect of Limited Data in Fourier Domain")
plt.xlabel("Frequency (Hz)"); plt.ylabel("|Amplitude|"); plt.grid(True)
plt.show()
#########################plot fft of a column in dataframe#############################################################
import numpy as np
import matplotlib.pyplot as plt

def fft_ordered(values,title = [], dt=1.0, demean=True, plot=True):
    """
    Compute a simple one-sided FFT for an ordered 1D array/Series.
    
    Parameters
    ----------
    values : array-like
        Ordered samples (equally spaced).
    dt : float, default 1.0
        Sampling interval (e.g., seconds per sample). If unknown, leave as 1.
    demean : bool, default True
        Subtract the mean before FFT (often helps).
    plot : bool, default True
        If True, show a basic amplitude spectrum plot.

    Returns
    -------
    freq : np.ndarray
        One-sided frequency axis (1/dt units, e.g., Hz if dt=seconds).
    amp : np.ndarray
        One-sided amplitude spectrum (simple scaling).
    """
    y = np.asarray(values, dtype=float)
    if demean:
        y = y - y.mean()

    N = y.size
    if N < 8:
        raise ValueError("Need at least 8 samples for a meaningful FFT.")

    # Real FFT (one-sided)
    Y = np.fft.rfft(y)
    freq = np.fft.rfftfreq(N, d=dt)

    # Simple amplitude scaling
    amp = (2.0 / N) * np.abs(Y)
    if N % 2 == 0:
        amp[-1] /= 2.0  # Nyquist term not doubled
    res = 1/N
    if plot:
        plt.figure(figsize=(8, 3.5))
        plt.plot(freq, amp, lw=1.2)
        plt.xlim(left=0)
        plt.xlabel(f"Frequency (1/{'dt' if dt==1.0 else 'unit'})", fontsize=16)
        plt.ylabel("Amplitude", fontsize=16)
        plt.title(f"Simple FFT (ordered samples) for {title}", fontdict={"fontsize":20})
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return freq, amp, res

# ---- Examples ----
# 1) Unit-spaced samples (unknown sampling rate)
# freq, amp = fft_ordered(df["my_column"].values, dt=1.0)

# 2) If your samples are every 5 minutes: dt = 5*60 seconds
# freq_hz, amp = fft_ordered(df["my_column"].values, dt=300.0)

# ---------------- Example usage ----------------
# df = pd.read_csv("your_timeseries.csv")
for key, title in datasets:
    train1, test1, target_index = load_data(key,  use_sentiment=0)
    freq, amp, res = fft_ordered(train1.iloc[:, target_index].values, title=title, dt=1.0)
    print(res, '\n')
    # plt.show()
    
    
    # freqs, amp = simple_fft(train1, time_col=train1.columns.to_list().index("time_step"), value_col=target_index, resample="D")