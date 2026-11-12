"""Score a file of dated, ticker-tagged headlines and build a daily sentiment signal.

Input CSV needs a date, a ticker and a headline column. Common names are detected automatically,
so the Kaggle "Massive Stock News Analysis DB" files (title/date/stock) work as-is.

Look-ahead protection: a headline published after the 4pm ET close is assigned to the next day.
If the file has dates but no times, every headline is lagged one day to be safe.

Usage:
    python src/score_headlines.py --input data/headlines.csv
    python src/score_headlines.py --input data/headlines.csv --model tfidf   # compare with baseline
"""
import argparse
import pickle

import numpy as np
import pandas as pd

from common import BASE_MODEL, DATA, FT_DIR, MODELS, predict_probs, sentiment_score


def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Need one of {names} in the input columns, found {list(df.columns)}")


def signal_dates(raw: pd.Series) -> pd.Series:
    # decide from the raw text: date-only strings would otherwise parse as midnight UTC (7pm ET)
    has_time = raw.astype(str).str.contains(r"\d:\d").mean() > 0.5
    if not has_time:
        day = pd.to_datetime(raw.astype(str).str[:10], errors="coerce")
        return day + pd.Timedelta(days=1), False
    ts = pd.to_datetime(raw, utc=True, errors="coerce", format="mixed")
    et = ts.dt.tz_convert("America/New_York")
    day = et.dt.tz_localize(None).dt.normalize()
    after_close = (et.dt.hour >= 16).astype(int)
    return day + pd.to_timedelta(after_close, unit="D"), True


def score(texts, model_name):
    if model_name == "tfidf":
        vec, lr = pickle.load(open(MODELS / "tfidf_logreg.pkl", "rb"))
        probs = lr.predict_proba(vec.transform(texts))
        return probs[:, np.argsort(lr.classes_)]
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = str(FT_DIR) if model_name == "finetuned" else BASE_MODEL
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path)
    return predict_probs(texts, mdl, tok, batch=256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", choices=["finetuned", "finbert", "tfidf"], default="finetuned")
    ap.add_argument("--max-rows", type=int, default=None, help="score a sample first to check timing")
    args = ap.parse_args()

    df = pd.read_csv(args.input, nrows=args.max_rows)
    tcol, dcol, kcol = pick(df, ["headline", "title", "text"]), pick(df, ["date", "datetime", "published"]), pick(df, ["ticker", "stock", "symbol"])
    df = df[[dcol, kcol, tcol]].dropna().rename(columns={dcol: "date", kcol: "ticker", tcol: "headline"})
    df["signal_date"], has_time = signal_dates(df["date"])
    df = df.dropna(subset=["signal_date"])
    print(f"{len(df):,} headlines, {df['ticker'].nunique():,} tickers; "
          f"{'timestamps found: after-close news moved to next day' if has_time else 'no timestamps: lagged 1 day'}")

    probs = score(df["headline"].astype(str).tolist(), args.model)
    df[["p_pos", "p_neg", "p_neu"]] = probs
    df["score"] = sentiment_score(probs)
    df.to_csv(DATA / f"scored_headlines_{args.model}.csv", index=False)

    daily = (df.groupby(["signal_date", "ticker"])
             .agg(sentiment=("score", "mean"), n_headlines=("score", "size"))
             .reset_index())
    daily.to_csv(DATA / f"daily_sentiment_{args.model}.csv", index=False)
    print(f"Saved {len(daily):,} ticker-days to data/daily_sentiment_{args.model}.csv")


if __name__ == "__main__":
    main()
