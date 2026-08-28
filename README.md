# **Extremely Low-Resource ASR for the Qidong Dialect:A Whisper-large-v3 QLoRA Adaptation Study**

Pipeline, configurations, and aggregate results for an MSc dissertation project
(7CS041, University of Wolverhampton, 2026): parameter-efficient fine-tuning
(QLoRA) of Whisper-large-v3 for the **Qidong (启东) Wu dialect**, with a
data-scaling ablation and a cross-dialect evaluation on WenetSpeech-Wu-Bench.

> **Note on data:** the Qidong speech corpus and all participant materials are
> **not** included in this repository for ethics reasons — see
> [Data availability](#data-availability). This repository provides the complete
> pipeline and the aggregate (non-identifying) results only.

## Results summary

Character Error Rate (CER), speaker-independent test set (lower is better).

| Evaluation | Zero-shot | Fine-tuned (qlora_full) |
|---|---:|---:|
| **Qidong** dialect test (n=470) | 0.8056 | **0.5886** |
| **WenetSpeech-Wu-Bench** (Mandarin ref, n=3000) | 0.8493 | 1.0883 |

Fine-tuning on Qidong strongly improves in-dialect ASR, but the adapter does
**not** transfer to other Wu dialects — on the public Wu benchmark it regresses
below the base model (decoder instability / repetition on out-of-domain audio).
Full numbers: `runs/results_master.csv`, `data/paper_table.csv`,
`wu_eval_summary.md`; per-condition metrics in `runs/*/summary.json`.

## Environment

- Windows 11, NVIDIA RTX 4070 Ti (12 GB), CUDA 12.1
- Python 3.10/3.11

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`ffmpeg` is required for non-WAV audio decoding. Training uses 4-bit QLoRA
(`bitsandbytes`) with gradient checkpointing; adapter = LoRA r=8 on
`q_proj`/`v_proj` (3.93 M trainable params, 0.25 % of the 1.55 B base). Peak VRAM
≈ 4.9 GB; full-corpus training ≈ 4.9 h wall-clock.

## Reproduction (six steps)

Requires the (ethics-restricted) corpus in `data/raw/`. Steps 1–4 build the
dataset; 5–6 train and evaluate.

```bash
# 1. Preprocess raw recordings -> 16 kHz mono WAV
python src/preprocess_audio.py --in data/raw --out data/processed/wav
# 2. Whisper-medium drafts -> Label Studio tasks
python src/draft_transcribe.py --wav data/processed/wav --out data/labelstudio/tasks.json
# 3. Correct drafts in Label Studio (see data/transcription_norms.md) -> export.json
# 4. Build HF dataset with a speaker-independent split
python src/build_dataset.py --export data/labelstudio/export.json \
    --wav data/processed/wav --out data/processed/hf_dataset \
    --test-ids spk01 spk08 --val-ids spk03 spk09
# 5. Train one condition (or the whole queue via start_ablation.bat)
python src/train.py --config configs/qlora_full.yaml
# 6. Evaluate (partitioned sentence/word-list CER)
python src/evaluate_model.py --dataset data/processed/hf_dataset \
    --adapter runs/qlora_full/best_adapter --out runs/qlora_full
python src/g2p_confusion.py --csv runs/qlora_full/per_utterance.csv \
    --out runs/qlora_full/confusion
```

Batch helpers (independent console, resume-safe): `start_ablation.bat`
(full data-scaling + seed queue), `start_wu_eval.bat` (cross-dialect eval),
`start_demo.bat` (CLI transcription demo).

## Demo

The Gradio app (`app/demo.py`) is retained but **non-functional in this
environment** due to a `gradio 4.44 ↔ starlette 1.3 / fastapi 0.140` version
mismatch (documented in-file). Use the CLI demo instead:

```bash
python app/demo_cli.py --audio path/to/clip.wav          # fine-tuned
python app/demo_cli.py --audio path/to/clip.wav --adapter ""   # zero-shot base
```

## Directory layout

```
src/            pipeline scripts (preprocess, draft, build, train, evaluate,
                g2p_confusion, run_experiments, text_norm, record_assistant)
app/            demo.py (Gradio; non-functional here, see Demo), demo_cli.py (CLI demo)
configs/        QLoRA configs: data-scaling tiers (005h/01h/025h/05h/full)
                + reproducibility seeds (qlora_01h_s{42,1,2})
data/           prompts.txt, transcription_norms.md, qc_report.md,
                paper_table.csv, assets_manifest.md  (NO corpus audio/text)
runs/           results_master.csv; per-condition summary.json;
                confusion/ matrices (csv + heatmap png)  (NO weights, NO
                per-utterance transcripts)
figures/        publication figures (data-scaling curve, partition, confusion)
wu_eval_summary.md   cross-dialect (RQ4) report
```

## Data availability

All participant data — audio recordings, questionnaires, consent records and
transcripts — are held on the researcher's personal computer under full-disk
encryption and were never uploaded to any cloud service or third-party platform.
None of this material is released. The corpus, questionnaires and consent records
are not published with this repository, are not deposited in any archive, and are
not available on request: the consent under which participants took part covers
use for this research only. What is released is the tooling and documentation —
the pipeline code, experimental configurations, elicitation script and
transcription protocol — all of which are reproducible without access to the
recordings. Fine-tuned adapter weights are available through a gated repository.

## License

Code released under the MIT License (see `LICENSE`). This license covers the
code only; it does **not** grant any rights to the corpus, participant
materials, or model weights, which are governed separately as above.
