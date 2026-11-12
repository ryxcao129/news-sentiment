"""Train and compare three headline sentiment classifiers.

1. TF-IDF + logistic regression (classic NLP baseline)
2. FinBERT out of the box (transformer pre-trained on financial text, zero-shot here)
3. FinBERT fine-tuned on financial news headlines (on your GPU, mixed precision)

Data: zeroshot/twitter-financial-news-sentiment on Hugging Face (~12k labelled finance
headlines: bearish / bullish / neutral). Its 'validation' split is held out as the test set.

Usage:
    python src/train_classifier.py
    python src/train_classifier.py --epochs 4 --batch 32 --lr 2e-5
"""
import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import BASE_MODEL, DEVICE, FT_DIR, LABELS, MODELS, REPORTS, predict_probs

# dataset labels: 0 bearish, 1 bullish, 2 neutral  ->  our order: positive, negative, neutral
REMAP = {0: 1, 1: 0, 2: 2}


def load_data(local_csv=None):
    if local_csv:  # columns: text,label (label already in LABELS order)
        df = pd.read_csv(local_csv)
        tr, te = train_test_split(df, test_size=0.2, random_state=0, stratify=df["label"])
        return tr.reset_index(drop=True), te.reset_index(drop=True)
    from datasets import load_dataset
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    tr, te = ds["train"].to_pandas(), ds["validation"].to_pandas()
    for d in (tr, te):
        d["label"] = d["label"].map(REMAP)
    return tr, te


def metrics(name, y, pred, secs=None):
    r = {"model": name, "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}
    if secs is not None:
        r["train_seconds"] = round(secs, 1)
    print(f"\n{name}: acc {r['accuracy']:.3f}, macro-F1 {r['macro_f1']:.3f}")
    print(classification_report(y, pred, target_names=LABELS, digits=3))
    return r


def finetune(model, tok, tr, va, epochs, batch, lr, max_len=64):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(tr) / batch))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    # class weights counter the neutral-heavy label mix
    counts = np.bincount(tr["label"], minlength=3)
    w = torch.tensor(len(tr) / (3 * np.maximum(counts, 1)), dtype=torch.float32, device=DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)

    best_f1, best_state, history = -1, None, []
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(tr))
        total = 0.0
        for i in range(0, len(idx), batch):
            b = tr.iloc[idx[i:i + batch]]
            enc = tok(list(b["text"]), padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(DEVICE)
            y = torch.tensor(b["label"].to_numpy(), device=DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = loss_fn(model(**enc).logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(b)
        vp = predict_probs(va["text"].tolist(), model, tok).argmax(1)
        vf1 = f1_score(va["label"], vp, average="macro")
        history.append({"epoch": ep + 1, "train_loss": total / len(tr), "val_macro_f1": vf1})
        print(f"  epoch {ep + 1}: train loss {total / len(tr):.4f}, val macro-F1 {vf1:.3f}")
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--local-csv", default=None, help="offline testing: CSV with text,label")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    train_full, test = load_data(args.local_csv)
    tr, va = train_test_split(train_full, test_size=0.15, random_state=0, stratify=train_full["label"])
    print(f"train {len(tr):,} / val {len(va):,} / test {len(test):,}")
    results = []

    # 1. TF-IDF baseline
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    lr = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    lr.fit(vec.fit_transform(tr["text"]), tr["label"])
    results.append(metrics("TF-IDF + LogReg", test["label"], lr.predict(vec.transform(test["text"])), time.time() - t0))
    pickle.dump((vec, lr), open(MODELS / "tfidf_logreg.pkl", "wb"))

    # 2. FinBERT zero-shot
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=3)
    p0 = predict_probs(test["text"].tolist(), model, tok)
    results.append(metrics("FinBERT (no fine-tuning)", test["label"], p0.argmax(1)))

    # 3. FinBERT fine-tuned
    t0 = time.time()
    model, hist = finetune(model, tok, tr, va, args.epochs, args.batch, args.lr)
    secs = time.time() - t0
    p1 = predict_probs(test["text"].tolist(), model, tok)
    results.append(metrics("FinBERT fine-tuned", test["label"], p1.argmax(1), secs))
    model.save_pretrained(FT_DIR); tok.save_pretrained(FT_DIR)

    res = pd.DataFrame(results)
    res.to_csv(REPORTS / "classifier_results.csv", index=False)
    pd.DataFrame(hist).to_csv(REPORTS / "finetune_history.csv", index=False)
    cm = pd.DataFrame(confusion_matrix(test["label"], p1.argmax(1)), index=LABELS, columns=LABELS)
    cm.to_csv(REPORTS / "confusion_matrix_finetuned.csv")
    json.dump({"device": DEVICE, "epochs": args.epochs, "base_model": args.base_model},
              open(REPORTS / "run_info.json", "w"), indent=2)
    print("\n== Test-set comparison\n" + res.round(3).to_string(index=False))
    print(f"\nFine-tuned model saved to {FT_DIR}")


if __name__ == "__main__":
    main()
