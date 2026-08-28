# -*- coding: utf-8 -*-
"""Step 5 — Test-set evaluation (§4.4.1).

Computes CER (primary metric) and WER on the held-out test split, and writes a
per-utterance CSV (ref/hyp/cer) that feeds both the results tables (Ch.5) and
the G2P confusion-matrix analysis (Step 6).

Usage:
    # baseline zero-shot:
    python src/evaluate_model.py --dataset data/processed/hf_dataset --out runs/zeroshot
    # fine-tuned adapter:
    python src/evaluate_model.py --dataset data/processed/hf_dataset \
        --adapter runs/qlora_8h/best_adapter --out runs/qlora_8h
"""
import argparse
import re
import sys
from pathlib import Path

import evaluate as hf_evaluate
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from text_norm import normalize


def main() -> None:
    # Windows consoles default stdout to GBK; printing a Whisper U+FFFD then
    # crashes with UnicodeEncodeError. Force UTF-8 + replace so logging is safe.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-model", default="openai/whisper-large-v3")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-partition", action="store_true",
                    help="skip sentence/wordlist partition (datasets w/o uttNNN naming)")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(
        args.base_model, language="zh", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    _d = load_from_disk(args.dataset)
    ds = _d[args.split] if hasattr(_d, "keys") else _d
    cer_m, wer_m = hf_evaluate.load("cer"), hf_evaluate.load("wer")

    rows = []
    for ex in ds:
        a = ex["audio"]
        feats = processor(
            a["array"], sampling_rate=a["sampling_rate"], return_tensors="pt"
        ).input_features.to(device, torch.float16)
        with torch.no_grad():
            ids = model.generate(feats, language="zh", task="transcribe",
                                 max_new_tokens=225)
        hyp = normalize(processor.batch_decode(ids, skip_special_tokens=True)[0])
        ref = normalize(ex["text"])
        u_cer = cer_m.compute(predictions=[hyp], references=[ref]) if ref else None
        m = re.search(r"utt(\d+)", Path(a["path"]).name) if a.get("path") else None
        utt = int(m.group(1)) if m else None
        region = "sentence" if (utt is not None and utt <= 155) else \
                 "wordlist" if utt is not None else "unknown"
        rows.append({"speaker": ex["speaker"], "utt": utt, "region": region,
                     "ref": ref, "hyp": hyp, "cer": u_cer})
        print(f"[{len(rows)}/{len(ds)}] cer={u_cer:.3f}  {ref[:20]} | {hyp[:20]}")

    df = pd.DataFrame(rows)
    corpus_cer = cer_m.compute(predictions=df.hyp.tolist(), references=df.ref.tolist())

    def _subset_cer(mask):
        sub = df[mask]
        if len(sub) == 0:
            return None, 0
        return round(cer_m.compute(predictions=sub.hyp.tolist(),
                                   references=sub.ref.tolist()), 4), len(sub)
    if args.no_partition:
        sentence_cer = wordlist_cer = None
        n_sentence = n_wordlist = 0
    else:
        sentence_cer, n_sentence = _subset_cer(df.region == "sentence")  # utt001-155
        wordlist_cer, n_wordlist = _subset_cer(df.region == "wordlist")  # utt156-235

    # WER on space-joined characters == CER for Chinese, so report char-level as
    # primary; keep a WER over whitespace tokens only if code-switching present.
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_utterance.csv", index=False)
    summary = {
        "condition": args.adapter or "zero-shot",
        "corpus_CER": round(corpus_cer, 4),
        "mean_utt_CER": round(df.cer.mean(), 4),
        "per_speaker_CER": df.groupby("speaker").cer.mean().round(4).to_dict(),
        "n_utts": len(df),
    }
    if not args.no_partition:
        summary.update({
            "sentence_CER": sentence_cer,
            "wordlist_CER": wordlist_cer,
            "n_sentence": n_sentence,
            "n_wordlist": n_wordlist,
        })
    pd.Series(summary).to_json(out / "summary.json", force_ascii=False, indent=2)
    print("\n", summary)


if __name__ == "__main__":
    main()
