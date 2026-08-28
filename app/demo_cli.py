# -*- coding: utf-8 -*-
"""CLI demo (§4.5): transcribe ONE audio file with the fine-tuned Qidong model.

A dependency-light alternative to app/demo.py, whose Gradio web stack is broken
by a gradio 4.44 <-> starlette 1.3.1 / fastapi 0.140 version mismatch in this
venv. Same model/adapter path and same normalisation as the eval pipeline.

Usage:
    python app/demo_cli.py --audio path/to/clip.wav
    python app/demo_cli.py --audio clip.m4a --adapter runs/qlora_full/best_adapter
"""
import argparse
import sys
import time
from pathlib import Path

# reuse the pipeline's normalisation (src/text_norm.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from text_norm import normalize

import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="path to an audio file")
    ap.add_argument("--adapter", default="runs/qlora_full/best_adapter",
                    help="LoRA adapter dir (empty string = zero-shot base model)")
    ap.add_argument("--base-model", default="openai/whisper-large-v3")
    args = ap.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"audio not found: {args.audio}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ---- load model (timed) ----
    t0 = time.time()
    processor = WhisperProcessor.from_pretrained(args.base_model, language="zh", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()
    load_s = time.time() - t0

    # ---- transcribe (timed) ----
    y, _ = librosa.load(args.audio, sr=16_000, mono=True)
    audio_s = len(y) / 16_000
    t1 = time.time()
    feats = processor(y, sampling_rate=16_000, return_tensors="pt")\
        .input_features.to(device, model.dtype)
    with torch.no_grad():
        ids = model.generate(feats, language="zh", task="transcribe", max_new_tokens=225)
    raw = processor.batch_decode(ids, skip_special_tokens=True)[0]
    infer_s = time.time() - t1

    print("=" * 64)
    print(f"audio          : {args.audio}  ({audio_s:.2f}s)")
    print(f"adapter        : {args.adapter or '(none / zero-shot base)'}")
    print(f"device         : {device}")
    print(f"model load time: {load_s:.1f} s")
    print(f"inference time : {infer_s:.2f} s")
    print("-" * 64)
    print(f"raw            : {raw}")
    print(f"normalized     : {normalize(raw)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
