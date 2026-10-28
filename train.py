"""Compare TF-IDF + LR/SVM and PyTorch RNN/LSTM on the same IMDB splits."""

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.svm import LinearSVC
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from data import build_vocab, load_imdb, make_loader, tokenize


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, kind, embedding=128, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding, padding_idx=0)
        self.recurrent = {"RNN": nn.RNN, "LSTM": nn.LSTM}[kind](
            embedding, hidden, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 2))

    def forward(self, tokens, lengths):
        packed = pack_padded_sequence(self.embedding(tokens), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, hidden = self.recurrent(packed)
        if isinstance(hidden, tuple):  # LSTM trả về (h_n, c_n).
            hidden = hidden[0]
        return self.classifier(hidden[-1])


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, targets, predictions = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for tokens, lengths, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = model(tokens, lengths)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
    return total_loss / len(targets), targets, predictions


def train_neural(kind, loaders, vocab_size, args, device):
    torch.manual_seed(args.seed)
    model = SentimentRNN(vocab_size, kind).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, best_state, stale, history = float("inf"), None, 0, []
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, loaders[0], device, optimizer)
        val_loss, truth, pred = run_epoch(model, loaders[1], device)
        val_acc = accuracy_score(truth, pred)
        history.append(dict(epoch=epoch, train_loss=train_loss,
                            val_loss=val_loss, val_accuracy=val_acc))
        print(f"{kind} {epoch:02d}: loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_acc, stale = val_loss, val_acc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 2:
                break
    elapsed = perf_counter() - start
    model.load_state_dict(best_state)
    _, truth, pred = run_epoch(model, loaders[2], device)
    return truth, pred, best_acc, elapsed, history


def evaluate(name, truth, pred, val_accuracy, seconds):
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, pred, average="binary", zero_division=0)
    row = dict(model=name, accuracy=accuracy_score(truth, pred), precision=precision,
               recall=recall, f1=f1, val_accuracy=val_accuracy, train_seconds=seconds)
    details = dict(confusion_matrix=confusion_matrix(truth, pred, labels=[0, 1]).tolist(),
                   report=classification_report(truth, pred, labels=[0, 1],
                       target_names=["negative", "positive"], output_dict=True,
                       zero_division=0))
    print(f"{name}: test_acc={row['accuracy']:.4f}, F1={f1:.4f}", flush=True)
    return row, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--limit", type=int, default=0, help="Samples per original split; 0 = all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_len) < 1 or args.vocab_size < 2:
        parser.error("epochs, batch-size, max-len must be > 0; vocab-size must be >= 2")
    if args.limit < 0 or 0 < args.limit < 20:
        parser.error("limit must be 0 or >= 20")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Device: {device}", flush=True)
    train, val, test = load_imdb(args.data_dir, args.limit, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), actual_device=device,
                  split_sizes=dict(train=len(train[0]), validation=len(val[0]), test=len(test[0])))
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Split sizes: {config['split_sizes']}", flush=True)

    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                 ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
    x_train = vectorizer.fit_transform(train[0])
    x_val, x_test = vectorizer.transform(val[0]), vectorizer.transform(test[0])
    rows, reports, histories = [], {}, {}
    for name, cls in [("LogisticRegression", LogisticRegression), ("SVM", LinearSVC)]:
        start, best_acc, best_model, best_c = perf_counter(), -1, None, None
        for c in (0.1, 1.0, 10.0):
            model = cls(C=c, max_iter=5000, random_state=args.seed).fit(x_train, train[1])
            acc = accuracy_score(val[1], model.predict(x_val))
            if acc > best_acc:
                best_acc, best_model, best_c = acc, model, c
        seconds = perf_counter() - start
        row, report = evaluate(name, test[1], best_model.predict(x_test), best_acc, seconds)
        report["selected_C"] = best_c
        rows.append(row)
        reports[name] = report

    vocab = build_vocab(train[0], args.vocab_size)
    loaders = [make_loader(split, vocab, args.max_len, args.batch_size, shuffle=i == 0)
               for i, split in enumerate((train, val, test))]
    for kind in ("RNN", "LSTM"):
        truth, pred, val_acc, seconds, histories[kind] = train_neural(
            kind, loaders, len(vocab), args, device)
        row, reports[kind] = evaluate(kind, truth, pred, val_acc, seconds)
        rows.append(row)

    with (output / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for filename, content in [("reports.json", reports), ("history.json", histories)]:
        (output / filename).write_text(json.dumps(content, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("accuracy", "f1")):
        bars = ax.bar([r["model"] for r in rows], [r[metric] for r in rows])
        ax.bar_label(bars, fmt="%.3f")
        ax.set(title=f"IMDB test {metric}", ylim=(0, 1.1))
        ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
