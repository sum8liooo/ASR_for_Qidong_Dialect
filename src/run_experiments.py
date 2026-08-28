# -*- coding: utf-8 -*-
"""Experiment orchestrator — run ALL conditions unattended (overnight).

For each YAML in configs/: train -> evaluate on test -> G2P confusion,
then aggregate every summary.json into results_master.csv (your Chapter 5
Table 5.x comes straight from that file).

Features:
- Resume: a condition whose output dir contains DONE.flag is skipped, so you
  can re-run the same command after a crash/power cut and it continues.
- Fault isolation: one condition failing (e.g. OOM) is logged to failures.log
  and the queue moves on instead of dying at 3am.
- Zero-shot baseline runs first automatically if missing.

Usage:
    python src/run_experiments.py                 # everything in configs/
    python src/run_experiments.py --only qlora_025h qlora_05h
    python src/run_experiments.py --shutdown      # power off when finished (Windows/Linux)
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list[str], log_file: Path) -> int:
    print(f"\n>>> {' '.join(cmd)}\n    (live log: {log_file})")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
        return proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--dataset", default="data/processed/hf_dataset")
    ap.add_argument("--only", nargs="*", default=None,
                    help="config stems to run, e.g. --only qlora_025h")
    ap.add_argument("--shutdown", action="store_true",
                    help="power off the machine when the queue finishes")
    args = ap.parse_args()

    cfg_files = sorted(Path(ROOT, args.configs).glob("*.yaml"))
    if args.only:
        cfg_files = [c for c in cfg_files if c.stem in set(args.only)]
    if not cfg_files:
        sys.exit("No configs matched.")

    failures = Path(ROOT, "runs", "failures.log")
    t0 = time.time()

    # ---- 0. zero-shot baseline (once) ----
    zs = Path(ROOT, "runs", "zeroshot")
    if not (zs / "summary.json").exists():
        rc = run([PY, "src/evaluate_model.py", "--dataset", args.dataset,
                  "--out", str(zs)], zs / "run.log")
        if rc != 0:
            open(failures, "a").write("zeroshot: evaluate failed\n")

    # ---- 1. queue of conditions ----
    for cfg_path in cfg_files:
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        out = Path(ROOT, cfg["output_dir"])
        flag = out / "DONE.flag"
        if flag.exists():
            print(f"[skip] {cfg_path.stem} already complete")
            continue
        print(f"\n{'='*60}\nCONDITION: {cfg_path.stem}\n{'='*60}")

        rc = run([PY, "src/train.py", "--config", str(cfg_path)], out / "train.log")
        if rc != 0:
            open(failures, "a").write(f"{cfg_path.stem}: TRAIN failed rc={rc}\n")
            continue

        adapter = out / "best_adapter"
        rc = run([PY, "src/evaluate_model.py", "--dataset", args.dataset,
                  "--adapter", str(adapter), "--out", str(out)], out / "eval.log")
        if rc != 0:
            open(failures, "a").write(f"{cfg_path.stem}: EVAL failed rc={rc}\n")
            continue

        run([PY, "src/g2p_confusion.py", "--csv", str(out / "per_utterance.csv"),
             "--out", str(out / "confusion")], out / "confusion.log")

        flag.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"[done] {cfg_path.stem}")

    # ---- 2. aggregate master results table ----
    rows = []
    for sj in Path(ROOT, "runs").glob("*/summary.json"):
        d = json.load(open(sj, encoding="utf-8"))
        d["run"] = sj.parent.name
        rows.append(d)
    if rows:
        cols = ["run", "corpus_CER", "sentence_CER", "wordlist_CER", "mean_utt_CER", "n_utts"]
        df = pd.DataFrame(rows)
        df = df[[c for c in cols if c in df.columns]]
        df = df.sort_values("corpus_CER")
        df.to_csv(Path(ROOT, "runs", "results_master.csv"), index=False)
        print("\n===== MASTER RESULTS =====")
        print(df.to_string(index=False))

    hrs = (time.time() - t0) / 3600
    print(f"\nQueue finished in {hrs:.1f} h. Failures (if any): {failures}")

    if args.shutdown:
        print("Shutting down in 60s (Ctrl+C to cancel)...")
        time.sleep(60)
        if platform.system() == "Windows":
            subprocess.run(["shutdown", "/s", "/t", "0"])
        else:
            subprocess.run(["sudo", "shutdown", "-h", "now"])


if __name__ == "__main__":
    main()
