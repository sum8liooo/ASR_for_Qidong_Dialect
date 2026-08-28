@echo off
REM ============================================================
REM  Qidong ASR CLI demo (whisper-large-v3 + qlora_full adapter).
REM  Double-click. When prompted, DRAG an audio file into this window
REM  (or paste its full path) and press Enter. Leave blank + Enter to quit.
REM  Prints raw + normalized transcription, plus load/inference times.
REM ============================================================
setlocal EnableDelayedExpansion

set "PATH=C:\Users\11819\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin;%PATH%"
set "PYTHONUTF8=1"
cd /d "D:\qidong-asr\qidong-asr-pipeline"
set "PY=D:\qidong-asr\qidong-asr-pipeline\.venv\Scripts\python.exe"

echo Qidong Dialect ASR - CLI demo (adapter: runs\qlora_full\best_adapter)
echo.

:loop
set "AUDIO="
set /p "AUDIO=Drag an audio file here (blank + Enter to quit): "
if "!AUDIO!"=="" goto end
set "AUDIO=!AUDIO:"=!"
echo.
"%PY%" app\demo_cli.py --audio "!AUDIO!"
echo.
goto loop

:end
echo Bye.
endlocal
