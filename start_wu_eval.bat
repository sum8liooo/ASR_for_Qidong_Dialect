@echo off
REM ============================================================
REM  Cross-dialect (RQ4) inference eval on WenetSpeech-Wu-Bench.
REM  qlora_full adapter vs zero-shot. INFERENCE ONLY, no training.
REM  Double-click; runs OUTSIDE any Claude session. Keep window open.
REM
REM  Order (per priority): 500-preview -> full mandarin -> dialect.
REM  Uses --no-partition (wu_bench has no uttNNN naming).
REM  Outputs to runs\wu_eval\ (all NEW dirs; existing results untouched).
REM ============================================================

set "PATH=C:\Users\11819\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin;%PATH%"
set "PYTHONUTF8=1"
cd /d "D:\qidong-asr\qidong-asr-pipeline"
set "PY=D:\qidong-asr\qidong-asr-pipeline\.venv\Scripts\python.exe"
set "ADAPTER=runs\qlora_full\best_adapter"

echo ===== PREVIEW a) zeroshot x mandarin_500 =====
"%PY%" src\evaluate_model.py --dataset data\processed\wu_bench_mandarin_500 --out runs\wu_eval\mandarin500_zeroshot --no-partition
echo ===== PREVIEW b) qlora_full x mandarin_500 =====
"%PY%" src\evaluate_model.py --dataset data\processed\wu_bench_mandarin_500 --adapter %ADAPTER% --out runs\wu_eval\mandarin500_full --no-partition
echo ***** PREVIEW DONE -- see runs\wu_eval\mandarin500_*\summary.json *****

echo ===== FULL a) zeroshot x mandarin (3000) =====
"%PY%" src\evaluate_model.py --dataset data\processed\wu_bench_mandarin --out runs\wu_eval\mandarin_zeroshot --no-partition
echo ===== FULL b) qlora_full x mandarin (3000) =====
"%PY%" src\evaluate_model.py --dataset data\processed\wu_bench_mandarin --adapter %ADAPTER% --out runs\wu_eval\mandarin_full --no-partition

echo ===== P2 c) zeroshot x dialect (4851) =====
"%PY%" src\evaluate_model.py --dataset data\processed\wu_bench_dialect --out runs\wu_eval\dialect_zeroshot --no-partition
echo ===== P2 d) qlora_full x dialect (4851) =====
"%PY%" src\evaluate_model.py --dataset data\processed\wu_bench_dialect --adapter %ADAPTER% --out runs\wu_eval\dialect_full --no-partition

echo ***** ALL DONE. Summaries: runs\wu_eval\*\summary.json *****
pause
