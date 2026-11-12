"""Does headline sentiment predict next-day stock returns?

For each trading day t, stocks with news are ranked by sentiment known before t's close, and
compared with their return from close t to close t+1. Reported:

- Rank IC: daily Spearman correlation between sentiment and next-day return, its mean and t-stat
- Quintile portfolios: average next-day return of the most negative to most positive fifth
- Long-short (top minus bottom quintile): annual return, volatility, Sharpe, max drawdown,
  gross and after transaction costs

Usage:
    python src/backtest.py --model finetuned
    python src/backtest.py --model tfidf --top-n 500 --cost-bps 5
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DATA, REPORTS


def get_prices(tickers, start, end, cache):
    if cache.exists():
        px = pd.read_csv(cache, index_col=0, parse_dates=True)
        if set(tickers) <= set(px.columns) and px.index.min() <= start and px.index.max() >= end:
            return px[list(tickers)]
    import yfinance as yf
    print(f"Downloading prices for {len(tickers)} tickers ...")
    px = yf.download(list(tickers), start=start - pd.Timedelta(days=5), end=end + pd.Timedelta(days=10),
                     auto_adjust=True, progress=False, threads=True)["Close"]
    px = px.dropna(axis=1, how="all")
    px.to_csv(cache)
    return px


def perf(r: pd.Series, periods=252):
    r = r.dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    return {"ann_return": (1 + r.mean()) ** periods - 1, "ann_vol": r.std() * np.sqrt(periods),
            "sharpe": r.mean() / r.std() * np.sqrt(periods) if r.std() > 0 else np.nan,
            "max_drawdown": (eq / eq.cummax() - 1).min(), "hit_rate": (r > 0).mean(), "days": len(r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="finetuned")
    ap.add_argument("--prices", default=None, help="optional wide CSV of closes (date x ticker)")
    ap.add_argument("--top-n", type=int, default=300, help="most-covered tickers to include")
    ap.add_argument("--min-names", type=int, default=10, help="min stocks with news per day")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="cost per side, basis points")
    args = ap.parse_args()

    sig = pd.read_csv(DATA / f"daily_sentiment_{args.model}.csv", parse_dates=["signal_date"])
    top = sig.groupby("ticker")["n_headlines"].sum().nlargest(args.top_n).index
    sig = sig[sig["ticker"].isin(top)]
    start, end = sig["signal_date"].min(), sig["signal_date"].max()
    px = (pd.read_csv(args.prices, index_col=0, parse_dates=True) if args.prices
          else get_prices(sorted(top), start, end, DATA / "prices.csv"))
    px = px.sort_index()
    fwd = px.shift(-1) / px - 1                        # close t -> close t+1
    fwd = fwd.sub(fwd.mean(axis=1), axis=0)            # market-neutral: excess over the cross-section

    # move each signal to the first trading day on/after its signal date
    days = px.index
    pos = days.searchsorted(sig["signal_date"])
    sig = sig[pos < len(days)].copy()
    sig["trade_date"] = days[pos[pos < len(days)]]
    sig = sig.groupby(["trade_date", "ticker"], as_index=False)["sentiment"].mean()
    long = fwd.stack().rename("fwd_ret").reset_index()
    long.columns = ["trade_date", "ticker", "fwd_ret"]
    panel = sig.merge(long, on=["trade_date", "ticker"]).dropna()
    print(f"{len(panel):,} stock-days with news and a next-day return, {panel['trade_date'].nunique():,} days")

    ics, qrets = [], []
    for d, g in panel.groupby("trade_date"):
        if len(g) < args.min_names or g["sentiment"].nunique() < 5:
            continue
        ics.append((d, spearmanr(g["sentiment"], g["fwd_ret"]).statistic))
        q = pd.qcut(g["sentiment"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        qrets.append(g.groupby(q, observed=True)["fwd_ret"].mean().rename(d))
    ic = pd.Series(dict(ics)).dropna()
    Q = pd.DataFrame(qrets)
    ls = Q[5] - Q[1]
    ls_net = ls - 2 * 2 * args.cost_bps / 1e4          # both legs, in and out each day

    ic_stats = {"mean_IC": ic.mean(), "IC_std": ic.std(), "IC_IR_annual": ic.mean() / ic.std() * np.sqrt(252),
                "IC_t_stat": ic.mean() / ic.std() * np.sqrt(len(ic)), "pct_days_IC_positive": (ic > 0).mean(),
                "days": len(ic)}
    rows = pd.DataFrame({"long_short_gross": perf(ls), f"long_short_net_{args.cost_bps:g}bps": perf(ls_net)}).T
    print("\n== Rank IC\n" + pd.Series(ic_stats).round(4).to_string())
    print("\n== Mean next-day excess return by sentiment quintile (bps)\n" + (Q.mean() * 1e4).round(2).to_string())
    print("\n== Long-short portfolio\n" + rows.round(3).to_string())

    out = f"{args.model}"
    pd.Series(ic_stats).to_csv(REPORTS / f"ic_{out}.csv")
    rows.to_csv(REPORTS / f"long_short_{out}.csv")
    (Q.mean() * 1e4).rename("bps").to_csv(REPORTS / f"quintiles_{out}.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    (1 + ls).cumprod().plot(ax=ax[0], label="gross")
    (1 + ls_net).cumprod().plot(ax=ax[0], label=f"net of {args.cost_bps:g} bps")
    ax[0].set(title="Sentiment long-short (top vs bottom quintile)", ylabel="Growth of $1")
    ax[0].legend()
    (Q.mean() * 1e4).plot.bar(ax=ax[1], color="steelblue")
    ax[1].set(title="Next-day excess return by sentiment quintile", xlabel="1 = most negative, 5 = most positive", ylabel="bps")
    fig.tight_layout()
    fig.savefig(REPORTS / f"backtest_{out}.png", dpi=150)
    print(f"\nSaved charts and tables to {REPORTS}")


if __name__ == "__main__":
    main()
