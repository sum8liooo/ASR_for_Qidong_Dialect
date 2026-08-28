# -*- coding: utf-8 -*-
"""Step 6 — Phoneme-level error analysis via G2P + edit-distance alignment (§4.4.2).

Pipeline: per-utterance ref/hyp (from evaluate_model.py) -> pinyin initial/final
sequences (pypinyin, extended by a Qidong override lexicon you curate) ->
Levenshtein alignment -> substitution confusion matrix -> heatmap.

IMPORTANT caveat to state in the dissertation: pypinyin gives MANDARIN readings;
mapping Qidong Wu phonology through Mandarin pinyin is an approximation. The
qidong_lexicon.tsv override file (char<TAB>initial<TAB>final) is where your own
dialectological knowledge enters — document every override (Appendix material).

Usage:
    python src/g2p_confusion.py --csv runs/qlora_8h/per_utterance.csv \
        --out runs/qlora_8h/confusion [--lexicon data/qidong_lexicon.tsv]
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pypinyin import Style, lazy_pinyin


def load_lexicon(path):
    lex = {}
    if path and Path(path).exists():
        for line in open(path, encoding="utf-8"):
            if line.strip() and not line.startswith("#"):
                ch, ini, fin = line.rstrip("\n").split("\t")
                lex[ch] = (ini, fin)
    return lex


def to_phonemes(text, lex):
    """char string -> list of (initial, final); lexicon overrides pypinyin."""
    out = []
    inis = lazy_pinyin(text, style=Style.INITIALS, strict=False)
    fins = lazy_pinyin(text, style=Style.FINALS, strict=False)
    for ch, ini, fin in zip(text, inis, fins):
        out.append(lex.get(ch, (ini or "-", fin or "-")))
    return out


def align(ref, hyp):
    """Levenshtein alignment returning ops: list of (tag, r_item, h_item)."""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + cost)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("sub" if ref[i - 1] != hyp[j - 1] else "ok",
                        ref[i - 1], hyp[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None)); i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1])); j -= 1
    return ops[::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lexicon", default=None)
    ap.add_argument("--unit", choices=["initial", "final"], default="initial")
    args = ap.parse_args()

    lex = load_lexicon(args.lexicon)
    df = pd.read_csv(args.csv).dropna(subset=["ref", "hyp"])
    idx = 0 if args.unit == "initial" else 1

    conf = Counter()
    for _, row in df.iterrows():
        r = [p[idx] for p in to_phonemes(str(row.ref), lex)]
        h = [p[idx] for p in to_phonemes(str(row.hyp), lex)]
        for tag, ri, hi in align(r, h):
            if tag == "sub":
                conf[(ri, hi)] += 1

    labels = sorted({k for pair in conf for k in pair})
    mat = pd.DataFrame(0, index=labels, columns=labels)
    for (r, h), c in conf.items():
        mat.loc[r, h] = c

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mat.to_csv(out / f"confusion_{args.unit}.csv")
    top = conf.most_common(15)
    pd.DataFrame(top, columns=["ref->hyp", "count"]).to_csv(
        out / f"top_confusions_{args.unit}.csv", index=False)

    plt.figure(figsize=(max(6, len(labels) * 0.45),) * 2)
    sns.heatmap(mat, annot=len(labels) <= 20, fmt="d", cmap="Blues",
                cbar_kws={"label": "substitution count"})
    plt.xlabel("hypothesis"); plt.ylabel("reference")
    plt.title(f"{args.unit.title()} substitution confusion matrix")
    plt.tight_layout()
    plt.savefig(out / f"confusion_{args.unit}.png", dpi=200)
    print(f"Top confusions ({args.unit}):")
    for (r, h), c in top:
        print(f"  {r} -> {h}: {c}")
    print(f"Saved matrix + heatmap -> {out}")


if __name__ == "__main__":
    main()
