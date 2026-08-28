# -*- coding: utf-8 -*-
"""Step 2 — Draft transcription with Whisper-medium + Label Studio task export (§4.2.2).

Produces drafts that human annotators CORRECT (never trust drafts as ground truth).

Usage:
    python src/draft_transcribe.py --wav data/processed/wav \
        --out data/labelstudio/tasks.json [--audio-url-prefix /data/local-files/?d=wav]

Then in Label Studio:
    1. Create project -> Labelling setup -> Audio transcription template.
    2. Import tasks.json (drafts appear as pre-annotations to edit, not retype).
    3. Annotators fix the text; export as JSON when done.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/whisper-medium")
    ap.add_argument("--audio-url-prefix", default="/data/local-files/?d=wav")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    # Force Mandarin transcription task; drafts will be imperfect for Qidong —
    # that is expected and is exactly what annotators correct.
    gen_kwargs = {"language": "zh", "task": "transcribe"}

    tasks = []
    wavs = sorted(Path(args.wav).rglob("*.wav"))
    for i, w in enumerate(wavs, 1):
        text = asr(str(w), generate_kwargs=gen_kwargs)["text"].strip()
        tasks.append(
            {
                "data": {"audio": f"{args.audio_url_prefix}/{w.name}"},
                "predictions": [
                    {
                        "model_version": args.model,
                        "result": [
                            {
                                "from_name": "transcription",
                                "to_name": "audio",
                                "type": "textarea",
                                "value": {"text": [text]},
                            }
                        ],
                    }
                ],
            }
        )
        print(f"[{i}/{len(wavs)}] {w.name}: {text[:40]}...")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(tasks)} Label Studio tasks -> {args.out}")


if __name__ == "__main__":
    main()
