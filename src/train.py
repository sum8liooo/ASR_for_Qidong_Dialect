# -*- coding: utf-8 -*-
"""Step 4 — LoRA/QLoRA fine-tuning of Whisper-large-v3 (§4.3).

Config-driven so every experimental condition (data scale 2/4/6/8h, LoRA vs
QLoRA, with/without WenetSpeech-Wu warm-up checkpoint) is one YAML file —
this gives you a clean audit trail for Chapter 4/5 tables.

Usage:
    python src/train.py --config configs/qlora_8h.yaml

Hardware: QLoRA(4-bit) fits large-v3 in ~14-16 GB VRAM (T4 16GB is tight but
works with batch 1 + grad-accum; L4/A100 comfortable). LoRA fp16 needs ~24 GB+.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

import evaluate as hf_evaluate
from text_norm import normalize


@dataclass
class Collator:
    processor: WhisperProcessor

    def __call__(self, feats):
        inputs = [{"input_features": f["input_features"]} for f in feats]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = [{"input_ids": f["labels"]} for f in feats]
        lab = self.processor.tokenizer.pad(labels, return_tensors="pt")
        lab = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (lab[:, 0] == self.processor.tokenizer.bos_token_id).all():
            lab = lab[:, 1:]
        batch["labels"] = lab
        return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8"))
    set_seed(cfg.get("seed", 42))   # seed torch/numpy/random BEFORE LoRA init + data prep

    processor = WhisperProcessor.from_pretrained(
        cfg["base_model"], language="zh", task="transcribe"
    )

    # ---- model: QLoRA (4-bit) or plain LoRA (fp16) ----
    if cfg.get("qlora", True):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            cfg["base_model"], quantization_config=bnb, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            cfg["base_model"], torch_dtype=torch.float16, device_map="auto"
        )
        if cfg.get("grad_ckpt", False):
            model.enable_input_require_grads()
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "zh"
    model.generation_config.task = "transcribe"

    # ---- optional warm-up checkpoint (WenetSpeech-Wu-adapted adapters) ----
    if cfg.get("init_adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["init_adapter"], is_trainable=True)
        print(f"Initialised adapters from {cfg['init_adapter']}")
    else:
        lora = LoraConfig(
            r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
            target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            bias="none",
        )
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- data ----
    ds = load_from_disk(cfg["dataset"])
    if cfg.get("train_hours"):  # data-scaling condition: subsample train split
        target = cfg["train_hours"] * 3600
        acc, keep = 0.0, []
        for i, ex in enumerate(ds["train"]):
            acc += len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
            keep.append(i)
            if acc >= target:
                break
        ds["train"] = ds["train"].select(keep)
        print(f"Subsampled train to {acc/3600:.2f} h ({len(keep)} utts)")

    def prep(ex):
        a = ex["audio"]
        ex["input_features"] = processor(
            a["array"], sampling_rate=a["sampling_rate"], return_tensors="np"
        ).input_features[0]
        ex["labels"] = processor.tokenizer(ex["text"]).input_ids
        return ex

    ds = ds.map(prep, remove_columns=["audio", "text", "speaker"], num_proc=1)

    cer_metric = hf_evaluate.load("cer")

    def compute_metrics(pred):
        ids = pred.predictions
        lab = pred.label_ids
        lab[lab == -100] = processor.tokenizer.pad_token_id
        hyp = processor.batch_decode(ids, skip_special_tokens=True)
        ref = processor.batch_decode(lab, skip_special_tokens=True)
        hyp = [normalize(h) for h in hyp]
        ref = [normalize(r) for r in ref]
        return {"cer": cer_metric.compute(predictions=hyp, references=ref)}

    args = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"],
        seed=cfg.get("seed", 42),
        data_seed=cfg.get("seed", 42),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        per_device_eval_batch_size=cfg.get("eval_batch_size", 1),
        gradient_accumulation_steps=cfg.get("grad_accum", 16),
        learning_rate=cfg.get("lr", 1e-4),
        warmup_ratio=0.1,
        max_steps=cfg.get("max_steps", 2000),
        fp16=True,
        gradient_checkpointing=cfg.get("grad_ckpt", False),
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 200),
        save_steps=cfg.get("eval_steps", 200),
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=Collator(processor),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(Path(cfg["output_dir"]) / "best_adapter")
    processor.save_pretrained(Path(cfg["output_dir"]) / "best_adapter")
    print("Training complete.")


if __name__ == "__main__":
    main()
