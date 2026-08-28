# -*- coding: utf-8 -*-
"""Recording assistant for corpus collection sessions (§4.2.1 field tool).

Shows each prompt in large text; the participant reads it; you control with:
    SPACE  start recording / stop & save & next
    R      discard current take and re-record this item
    B      go back one item (to re-do the previous)
    S      skip this item (marks skipped; no file saved)
    Q/Esc  quit (progress is safe: files already saved stay on disk)

Files are written directly as 16 kHz mono PCM16 WAV named spkXX_uttYYY.wav —
matching the pipeline convention, so preprocessing/trim runs on them as-is.
Resume is automatic: on start, jumps to the first item with no saved file.

Usage:
    python src/record_assistant.py --speaker spk01
    # optional: --prompts data/prompts.txt --out data/raw --device N
    # list input devices: python -m sounddevice
"""
import argparse
import queue
import sys
import time
import tkinter as tk
from tkinter import simpledialog
from pathlib import Path

# When bundled as an exe (PyInstaller), resources live next to the exe.
FROZEN = getattr(sys, "frozen", False)
BASE = Path(sys.executable).parent if FROZEN else Path(".")

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16_000


class Recorder:
    def __init__(self, device=None):
        self.q = queue.Queue()
        self.frames = []
        self.stream = None
        self.device = device

    def start(self):
        self.frames = []
        self.q = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=SR, channels=1, dtype="int16", device=self.device,
            callback=lambda indata, *_: self.q.put(indata.copy()))
        self.stream.start()

    def stop(self):
        self.stream.stop(); self.stream.close(); self.stream = None
        while not self.q.empty():
            self.frames.append(self.q.get())
        return np.concatenate(self.frames) if self.frames else np.zeros((0, 1), np.int16)

    def drain(self):
        while not self.q.empty():
            self.frames.append(self.q.get())


class App:
    def __init__(self, root, prompts, speaker, outdir, device):
        self.root, self.prompts, self.speaker = root, prompts, speaker
        self.outdir = Path(outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        self.rec = Recorder(device)
        self.recording = False
        self.t0 = 0.0
        self.idx = self.first_missing()

        root.title(f"Qidong Recording — {speaker}")
        root.configure(bg="#101418")
        root.geometry("1000x560")
        self.lbl_prog = tk.Label(root, font=("Microsoft YaHei", 16), fg="#8aa", bg="#101418")
        self.lbl_prog.pack(pady=(18, 4))
        self.lbl_text = tk.Label(root, font=("Microsoft YaHei", 44, "bold"),
                                 fg="#f2f2f2", bg="#101418", wraplength=920, justify="center")
        self.lbl_text.pack(expand=True, fill="both", padx=30)
        self.lbl_stat = tk.Label(root, font=("Microsoft YaHei", 18), fg="#7f7", bg="#101418")
        self.lbl_stat.pack(pady=(0, 6))
        self.lbl_help = tk.Label(root, text="空格=开始/停止保存    R=重录    B=上一句    S=跳过    Q=退出",
                                 font=("Microsoft YaHei", 13), fg="#667", bg="#101418")
        self.lbl_help.pack(pady=(0, 14))
        root.bind("<space>", self.toggle)
        root.bind("r", self.redo); root.bind("R", self.redo)
        root.bind("b", self.back); root.bind("B", self.back)
        root.bind("s", self.skip); root.bind("S", self.skip)
        root.bind("q", self.quit); root.bind("Q", self.quit); root.bind("<Escape>", self.quit)
        self.tick()
        self.render()

    # ---------- helpers ----------
    def fname(self, i):
        return self.outdir / f"{self.speaker}_utt{i+1:03d}.wav"

    def first_missing(self):
        for i in range(len(self.prompts)):
            if not self.fname(i).exists():
                return i
        return len(self.prompts)

    def render(self, msg=None):
        if self.idx >= len(self.prompts):
            self.lbl_prog.config(text="")
            self.lbl_text.config(text="\u5168\u90e8\u5b8c\u6210 \U0001F389\n\u8f9b\u82e6\u4e86\uff01")
            self.lbl_stat.config(text=f"{self.speaker}: {len(self.prompts)} items done")
            return
        done = self.fname(self.idx).exists()
        self.lbl_prog.config(text=f"{self.speaker}   \u7b2c {self.idx+1} / {len(self.prompts)} \u53e5"
                                  + ("   (\u5df2\u5f55\u8fc7,\u7a7a\u683c\u91cd\u5f55)" if done and not self.recording else ""))
        self.lbl_text.config(text=self.prompts[self.idx])
        if msg:
            self.lbl_stat.config(text=msg)
        elif self.recording:
            pass  # tick() updates
        else:
            self.lbl_stat.config(text="\u6309\u7a7a\u683c\u5f00\u59cb\u5f55\u97f3")

    def tick(self):
        if self.recording:
            self.rec.drain()
            self.lbl_stat.config(text=f"\u25cf \u5f55\u97f3\u4e2d\u2026 {time.time()-self.t0:4.1f}s   (\u7a7a\u683c=\u505c\u6b62\u5e76\u4fdd\u5b58)", fg="#f66")
        else:
            self.lbl_stat.config(fg="#7f7")
        self.root.after(100, self.tick)

    # ---------- key handlers ----------
    def toggle(self, _=None):
        if self.idx >= len(self.prompts):
            return
        if not self.recording:
            self.rec.start(); self.recording = True; self.t0 = time.time()
        else:
            audio = self.rec.stop(); self.recording = False
            dur = len(audio) / SR
            if dur < 0.3:
                self.render("\u592a\u77ed,\u672a\u4fdd\u5b58 \u2014 \u6309\u7a7a\u683c\u91cd\u5f55")
                return
            sf.write(self.fname(self.idx), audio, SR, subtype="PCM_16")
            self.idx += 1
            self.render(f"\u5df2\u4fdd\u5b58 ({dur:.1f}s) \u2192 \u4e0b\u4e00\u53e5")

    def redo(self, _=None):
        if self.recording:
            self.rec.stop(); self.recording = False
        f = self.fname(self.idx)
        if f.exists():
            f.unlink()
        self.render("\u5df2\u4e22\u5f03,\u6309\u7a7a\u683c\u91cd\u5f55\u672c\u53e5")

    def back(self, _=None):
        if self.recording:
            return
        if self.idx > 0:
            self.idx -= 1
            self.render("\u56de\u5230\u4e0a\u4e00\u53e5 (\u7a7a\u683c=\u91cd\u5f55\u8986\u76d6)")

    def skip(self, _=None):
        if self.recording:
            return
        self.idx += 1
        self.render("\u5df2\u8df3\u8fc7")

    def quit(self, _=None):
        if self.recording:
            self.rec.stop()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", default=None, help="e.g. spk01")
    ap.add_argument("--prompts", default=str(BASE / "prompts.txt") if FROZEN else "data/prompts.txt")
    ap.add_argument("--out", default=str(BASE / "recordings") if FROZEN else "data/raw")
    ap.add_argument("--device", type=int, default=None,
                    help="input device index (python -m sounddevice to list)")
    args = ap.parse_args()

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip()]
    root = tk.Tk()
    speaker = args.speaker
    if not speaker:
        root.withdraw()
        speaker = simpledialog.askstring(
            "\u8bf4\u8bdd\u4eba\u7f16\u53f7",
            "\u8bf7\u8f93\u5165\u7814\u7a76\u8005\u544a\u77e5\u4f60\u7684\u7f16\u53f7 (\u5982 spk03):",
            parent=root)
        if not speaker:
            return
        speaker = speaker.strip()
        root.deiconify()
    App(root, prompts, speaker, args.out, args.device)
    root.mainloop()


if __name__ == "__main__":
    main()
