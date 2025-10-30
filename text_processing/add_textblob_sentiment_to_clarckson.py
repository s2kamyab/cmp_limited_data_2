import re
from pathlib import Path
from datetime import datetime
import pandas as pd

# If TextBlob isn't installed: pip install textblob
from textblob import TextBlob

# ----------------- CONFIG -----------------
TXT_FOLDER = Path("C:/Users/shima/OneDrive/Documentos/Postdoc_Uvic/Paper1/Data/Soshianest/news/all_reports_csv")   # folder with .txt files
CSV_IN     = Path("paper_1_git_repo/data_soshianest/549324_dataset.csv")    # CSV with a 'date' column in YYYYMMDD (possibly with spaces)
CSV_OUT    = Path("paper_1_git_repo/data_soshianest/549324_dataset_with_sentiment.csv")
# ------------------------------------------

DATE_RE = re.compile(r"(\d{2})[._-](\d{2})[._-](\d{4})")  # matches DD_MM_YYYY or DD-MM-YYYY or DD.MM.YYYY

def parse_date_from_filename(name: str):
    """
    Extract datetime.date from a filename like '... 25_09_1998.txt'.
    Returns None if no date is found.
    """
    m = DATE_RE.search(name)
    if not m:
        return None
    dd, mm, yyyy = m.groups()
    try:
        return datetime(int(yyyy), int(mm), int(dd)).date()
    except ValueError:
        return None

def textblob_polarity(text: str) -> float:
    try:
        return float(TextBlob(text).sentiment.polarity)  # [-1, 1]
    except Exception:
        return float("nan")

def load_txt_sentiments_by_year(folder: Path):
    """
    Reads all .txt files, parses date, computes sentiment.
    Returns dict: year -> list of dicts with {'date': date, 'sentiment': float}
    (sorted ascending by date).
    """
    by_year = {}
    for p in folder.glob("*.txt"):
        d = parse_date_from_filename(p.name)
        if d is None:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # fallback encoding
            txt = p.read_text(errors="ignore")
        pol = textblob_polarity(txt)
        by_year.setdefault(d.year, []).append({"date": d, "sentiment": pol})
    # sort each year's entries by date
    for y in by_year:
        by_year[y].sort(key=lambda r: r["date"])
    return by_year

def closest_past_same_year(year_entries, query_date):
    """
    Given a sorted list of {'date', 'sentiment'} for one year,
    return the entry with max date <= query_date, or None if none.
    """
    if not year_entries:
        return None
    # Binary search for rightmost date <= query_date
    lo, hi = 0, len(year_entries) - 1
    best_idx = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if year_entries[mid]["date"] <= query_date:
            best_idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return year_entries[best_idx] if best_idx >= 0 else None

def parse_csv_date(s):
    """
    Parse YYYYMMDD (possibly with trailing spaces) to datetime.date.
    """
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return None

def main():
    # 1) Precompute sentiments from TXT files grouped by year
    year_map = load_txt_sentiments_by_year(TXT_FOLDER)

    # 2) Load CSV and ensure 'date' column exists
    df = pd.read_csv(CSV_IN)
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a 'date' column in YYYYMMDD format.")

    # 3) For each row, find closest past txt date in the same year and assign sentiment + txtdate
    sentiments = []
    txtdates = []
    for val in df["date"].tolist():
        d = parse_csv_date(val)
        if d is None:
            sentiments.append(float("nan"))
            txtdates.append("")
            continue

        entries = year_map.get(d.year, [])
        match = closest_past_same_year(entries, d)
        if match is None:
            sentiments.append(float("nan"))
            txtdates.append("")
        else:
            sentiments.append(match["sentiment"])
            txtdates.append(match["date"].strftime("%Y-%m-%d"))

    df["sentiment"] = sentiments
    df["txtdate"] = txtdates

    # 4) Save
    df.to_csv(CSV_OUT, index=False)
    print(f"Saved: {CSV_OUT.resolve()}")

if __name__ == "__main__":
    main()
