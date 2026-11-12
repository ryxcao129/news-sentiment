"""Daily research tool: score the latest headlines for your watchlist.

Usage:
    python src/live.py --tickers AAPL MSFT NVDA BAC TD
    python src/live.py --watchlist data/watchlist.txt     # one ticker per line
"""
import argparse
from datetime import datetime, timezone

import pandas as pd

from common import FT_DIR, REPORTS, predict_probs, sentiment_score


def headlines(ticker):
    import yfinance as yf
    rows = []
    for item in yf.Ticker(ticker).news or []:
        c = item.get("content", item)  # yfinance changed its news format over versions
        title = c.get("title")
        when = c.get("pubDate") or c.get("providerPublishTime")
        if isinstance(when, (int, float)):
            when = datetime.fromtimestamp(when, tz=timezone.utc).isoformat()
        if title:
            rows.append({"ticker": ticker, "published": when, "headline": title})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=[])
    ap.add_argument("--watchlist", default=None)
    args = ap.parse_args()
    tickers = list(args.tickers)
    if args.watchlist:
        tickers += [t.strip().upper() for t in open(args.watchlist) if t.strip()]
    if not tickers:
        ap.error("give --tickers or --watchlist")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(FT_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(FT_DIR)

    df = pd.DataFrame([r for t in tickers for r in headlines(t)])
    if df.empty:
        print("No headlines returned.")
        return
    df["score"] = sentiment_score(predict_probs(df["headline"].tolist(), mdl, tok))
    summary = (df.groupby("ticker")
               .agg(sentiment=("score", "mean"), headlines=("score", "size"), most_positive=("score", "max"),
                    most_negative=("score", "min"))
               .sort_values("sentiment", ascending=False).round(2))
    print("\n== Watchlist sentiment (latest headlines, -1 to +1)\n" + summary.to_string())
    for t in summary.index:
        g = df[df.ticker == t].sort_values("score")
        print(f"\n{t}")
        for _, r in pd.concat([g.head(2), g.tail(2)]).drop_duplicates().iterrows():
            print(f"  {r['score']:+.2f}  {r['headline']}")
    stamp = datetime.now().strftime("%Y-%m-%d")
    df.to_csv(REPORTS / f"live_{stamp}.csv", index=False)


if __name__ == "__main__":
    main()
