# -*- coding: utf-8 -*-
"""Step 1 — Audio preprocessing (§4.2.1).

raw recordings (any format) -> 16 kHz mono WAV, peak-normalised, silence-trimmed.

Usage:
    python src/preprocess_audio.py --in data/raw --out data/processed/wav

Design notes for the dissertation:
- 16 kHz mono is Whisper's expected input; resampling with librosa (soxr_hq).
- Peak normalisation to -1 dBFS equalises loudness variance across phones/mics.
- Leading/trailing silence trimmed at 30 dB below peak; internal pauses kept
  (natural prosody matters for dialect ASR).
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16_000


def process_one(src: Path, dst: Path, trim_db: float = 30.0) -> float:
    y, _ = librosa.load(src, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 10 ** (-1 / 20)  # -1 dBFS
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, y, SR, subtype="PCM_16")
    return len(y) / SR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    exts = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}
    total = 0.0
    files = [p for p in src_root.rglob("*") if p.suffix.lower() in exts]
    for i, p in enumerate(sorted(files), 1):
        rel = p.relative_to(src_root).with_suffix(".wav")
        dur = process_one(p, dst_root / rel)
        total += dur
        print(f"[{i}/{len(files)}] {rel}  {dur:6.1f}s")
    print(f"\nTotal: {total/3600:.2f} h across {len(files)} files")


if __name__ == "__main__":
    main()
