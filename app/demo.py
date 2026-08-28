# -*- coding: utf-8 -*-
"""Gradio demo (§4.5): microphone/file -> Qidong transcription.

Usage:
    python app/demo.py --adapter runs/qlora_8h/best_adapter
Deploy: push this folder + adapter to a Hugging Face Space (hardware: T4 small).
"""
import argparse

import gradio as gr
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def _patch_gradio_client_bool_schema():
    """Work around a gradio_client 1.3.0 (gradio 4.44) bug.

    gradio_client.utils._json_schema_to_python_type -> get_type() runs
    `if "const" in schema` on a BOOLEAN `additionalProperties` (booleans are a
    valid JSON-Schema value) and raises
        TypeError: argument of type 'bool' is not iterable
    while generating the /info API schema. That 500s the info endpoint, so
    Gradio's post-launch localhost self-check fails and .launch() aborts with
        ValueError: When localhost is not accessible, a shareable link must be created
    Fix: short-circuit boolean sub-schemas to "Any" so schema generation
    succeeds. Runtime monkeypatch only -- no dependency/venv change.
    """
    try:
        import gradio_client.utils as gcu
        _orig = gcu._json_schema_to_python_type

        def _safe(schema, defs=None):
            if isinstance(schema, bool):
                return "Any"
            return _orig(schema, defs)

        gcu._json_schema_to_python_type = _safe
    except Exception as e:  # never let the shim itself break the demo
        print(f"[demo] gradio_client shim skipped: {e}")


_patch_gradio_client_bool_schema()


def build(adapter, base="openai/whisper-large-v3"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(base, language="zh", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()

    def transcribe(audio):
        import librosa
        y, _ = librosa.load(audio, sr=16_000, mono=True)
        feats = processor(y, sampling_rate=16_000, return_tensors="pt")\
            .input_features.to(device, model.dtype)
        with torch.no_grad():
            ids = model.generate(feats, language="zh", task="transcribe",
                                 max_new_tokens=225)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

    return gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(sources=["microphone", "upload"], type="filepath",
                        label="启东话语音 / Qidong speech"),
        outputs=gr.Textbox(label="转写结果 / Transcription"),
        title="Qidong Dialect ASR (Whisper-large-v3 + QLoRA)",
        description="MSc dissertation demo — University of Wolverhampton, 2026.",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    build(ap.parse_args().adapter).launch(server_name="127.0.0.1", share=False)
