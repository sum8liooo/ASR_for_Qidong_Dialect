# -*- coding: utf-8 -*-
"""Step 3 — Build HF dataset with a SPEAKER-INDEPENDENT split (§4.2.3).

Input: Label Studio JSON export + wav dir.
File naming convention assumed: spkXX_uttYYY.wav  (speaker id = prefix before '_').

Speaker independence is non-negotiable for the dissertation's validity claim:
test-set speakers must never appear in train/val, otherwise CER is inflated by
speaker memorisation rather than dialect learning (discuss in §3.4 / §6.8).

Usage:
    python src/build_dataset.py --export data/labelstudio/export.json \
        --wav data/processed/wav --out data/processed/hf_dataset \
        --test-speakers 3 --val-speakers 2 --seed 42
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict


def speaker_of(fname: str) -> str:
    return Path(fname).stem.split("_")[0]


def load_export(path: str) -> list[dict]:
    rows = []
    for item in json.load(open(path, encoding="utf-8")):
        audio = Path(item["data"]["audio"]).name
        # last annotation wins (post-correction)
        ann = item.get("annotations") or []
        if not ann:
            continue
        texts = [
            r["value"]["text"][0]
            for r in ann[-1]["result"]
            if r.get("type") == "textarea"
        ]
        if texts and texts[0].strip():
            rows.append({"file": audio, "text": texts[0].strip()})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-speakers", type=int, default=3)
    ap.add_argument("--val-speakers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-ids", nargs="*", default=None,
                    help="explicit test speaker ids (e.g. spk01 spk08); overrides --test-speakers")
    ap.add_argument("--val-ids", nargs="*", default=None,
                    help="explicit val speaker ids (e.g. spk03 spk09); overrides --val-speakers")
    args = ap.parse_args()

    rows = load_export(args.export)
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[speaker_of(r["file"])].append(r)

    if args.test_ids or args.val_ids:          # explicit speaker selection
        test_s = set(args.test_ids or [])
        val_s = set(args.val_ids or [])
        known = set(by_spk)
        unknown = (test_s | val_s) - known
        if unknown:
            raise SystemExit(
                f"Unknown/empty speaker ids: {sorted(unknown)}. "
                f"Available (have annotations): {sorted(known)}")
        if test_s & val_s:
            raise SystemExit(f"test/val overlap: {sorted(test_s & val_s)}")
    else:                                       # seeded-random fallback (by count)
        spks = sorted(by_spk)
        random.Random(args.seed).shuffle(spks)
        test_s = set(spks[: args.test_speakers])
        val_s = set(spks[args.test_speakers : args.test_speakers + args.val_speakers])

    def rows_for(sset):
        out = []
        for s in sset:
            for r in by_spk[s]:
                out.append(
                    {
                        "audio": str(Path(args.wav) / r["file"]),
                        "text": r["text"],
                        "speaker": s,
                    }
                )
        return out

    train_s = sorted(set(by_spk) - test_s - val_s)
    if not train_s:
        raise SystemExit("no speakers left for train — check --test-ids/--val-ids")
    splits = {
        "train": rows_for(train_s),
        "validation": rows_for(val_s),
        "test": rows_for(test_s),
    }
    ds = DatasetDict(
        {k: Dataset.from_list(v).cast_column("audio", Audio(sampling_rate=16_000))
         for k, v in splits.items()}
    )
    ds.save_to_disk(args.out)
    for k in ds:
        spk = sorted({r["speaker"] for r in splits[k]})
        print(f"{k:>10}: {len(ds[k]):4d} utts, speakers={spk}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
