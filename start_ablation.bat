@echo off
REM ============================================================
REM  Overnight QLoRA data-scaling ablation for Qidong ASR.
REM  Double-click to run OUTSIDE any Claude session (survives independently).
REM  Keep this window open overnight. Ctrl+C or closing the window stops it.
REM
REM  Resume-safe: conditions that finished (have runs\<cond>\DONE.flag)
REM  are skipped, so you can re-run this after any interruption.
REM  Runs: zeroshot baseline, then qlora_005h / 01h / 025h / 05h / full.
REM ============================================================

set "PATH=C:\Users\11819\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin;%PATH%"
cd /d "D:\qidong-asr\qidong-asr-pipeline"

echo Starting ablation queue (zeroshot + 5 conditions)...
echo Live per-condition logs: runs\<condition>\train.log  /  eval.log
echo.
"D:\qidong-asr\qidong-asr-pipeline\.venv\Scripts\python.exe" src\run_experiments.py
echo.
echo === Queue finished. Master table: runs\results_master.csv  Failures: runs\failures.log ===
pause
