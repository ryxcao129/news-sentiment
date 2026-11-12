# Financial News Sentiment Signal

Fine-tunes a transformer (FinBERT) to classify financial headlines as positive, negative or neutral,
turns those predictions into a daily sentiment score per stock, and tests whether that score
predicts next-day returns. Also includes a daily watchlist tool that scores today's headlines.

## What's inside (the ML)

| Piece | What it does |
| --- | --- |
| Three-model comparison (`train_classifier.py`) | TF-IDF + logistic regression baseline vs FinBERT zero-shot vs FinBERT fine-tuned. Reports accuracy, macro-F1, per-class precision/recall and a confusion matrix on a held-out test set. |
| GPU fine-tuning | Custom PyTorch loop: AdamW, linear warmup/decay, mixed-precision (fp16) training, gradient clipping, class-weighted loss for the neutral-heavy label mix, best epoch picked on validation macro-F1. |
| Signal construction (`score_headlines.py`) | Batch GPU inference, score = P(positive) − P(negative), averaged per stock per day. Headlines after the 4pm ET close move to the next day to avoid look-ahead bias. |
| Backtest (`backtest.py`) | Daily cross-sectional rank IC with t-stat, sentiment-quintile returns, and a market-neutral long-short portfolio with Sharpe, drawdown and transaction costs. |
| Live tool (`live.py`) | Scores the latest Yahoo Finance headlines for your watchlist. |

## Setup (once, in Anaconda Prompt)

```bash
conda create -n sent python=3.11 -y
conda activate sent
cd %USERPROFILE%\Documents\ml_projects\news_sentiment
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 1. Train and compare the classifiers

```bash
python src/train_classifier.py
```

Downloads FinBERT and the labelled headline dataset from Hugging Face automatically. Fine-tuning
takes a few minutes on the RTX 4050 (3 epochs, ~10k headlines). Outputs `reports/classifier_results.csv`
and saves the model to `models/finbert_finetuned`. If you run out of GPU memory, add `--batch 16`.

## 2. Backtest the signal

You need historical headlines with dates and tickers. A free option is the Kaggle dataset
**"Massive Stock News Analysis DB for NLP/Backtests"** (about 1.4M headlines, 2009-2020).
Download it, put `analyst_ratings_processed.csv` (or `raw_partner_headlines.csv`) in `data/`, then:

```bash
python src/score_headlines.py --input data/analyst_ratings_processed.csv --max-rows 50000   # quick test
python src/score_headlines.py --input data/analyst_ratings_processed.csv                    # full run
python src/backtest.py --model finetuned
```

Repeat with `--model tfidf` (and `--model finbert`) in both commands to compare against the
baselines. Prices download from Yahoo Finance and are cached in `data/prices.csv`. Results go to
`reports/`: `ic_*.csv`, `quintiles_*.csv`, `long_short_*.csv`, `backtest_*.png`.

**What to expect:** on real data a daily IC of 0.01-0.03 is typical and already meaningful. Most of
the effect usually shows up on the day of the news, so after-close timing and costs matter a lot.
A result that looks too good is more likely a timing leak than alpha.

## 3. Daily use

```bash
python src/live.py --tickers AAPL NVDA BAC TD JPM
```

Ranks your watchlist by headline sentiment and shows the most positive and negative headlines
for each name.

## Ideas to extend

- Weight headlines by source or by how surprising they are (sentiment change vs the stock's average).
- Test longer horizons (2-5 days) and whether negative news has a bigger effect than positive news.
- Compare against a local LLM classifier (e.g. Llama 3 8B via Ollama) on the same test set.
