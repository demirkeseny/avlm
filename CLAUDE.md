# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project that adds an **audio/ASR capability to Qwen2-VL-7B-Instruct** — a vision-language model that doesn't natively understand audio. The core idea: graft a Whisper-large-v3-turbo encoder plus a learned linear projection head onto Qwen2VL so that the same model can handle text, images, and speech in a single forward pass.

The project lives entirely in Jupyter notebooks and a custom fork of `transformers`. There are no CLI entry points, no test suite, and no build step.

## Environment Setup

```bash
# Create and activate the virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies (CPU/macOS wheels are pinned)
pip install -r requirements.txt

# Install the local transformers fork in editable mode (required — upstream transformers lacks Qwen2VL audio support)
pip install -e transformers/

# Set HuggingFace token (copy .env.example → .env and fill in)
cp .env.example .env
# Edit .env: HF_TOKEN=<your token>
```

On GPU machines (CUDA 12.1), swap the PyTorch lines in `requirements.txt` for the `+cu121` wheels and add `--extra-index-url https://download.pytorch.org/whl/cu121`.

## Key Dependencies

| Package | Role |
|---|---|
| `transformers/` (local fork) | Contains `Qwen2VLForConditionalGenerationWithAudio`, `Qwen2VLAudioConfig`, `WhisperAudioEncoder` |
| `trl` | SFT trainer for fine-tuning |
| `peft` | QLoRA adapter training |
| `bitsandbytes` | 4-bit quantization |
| `librosa` / `torchaudio` | Audio loading and resampling |
| `wandb` | Training metrics |

## Architecture

### Custom classes (all in `transformers/src/transformers/models/qwen2_vl/`)

- **`Qwen2VLAudioConfig`** (`configuration_qwen2_vl.py:24`) — config for the audio encoder: Whisper model name, mel spectrogram parameters, projection output size, and special audio token IDs (`<|audio_start|>` 151657, `<|audio_pad|>` 151658, `<|audio_end|>` 151659).

- **`WhisperAudioEncoder`** (`modeling_qwen2_vl.py:2044`) — wraps `openai/whisper-large-v3-turbo` encoder + a single `nn.Linear` projection (`audio_projection`) that maps Whisper hidden states → Qwen2VL hidden size (3584). This is the only component trained in Stage 1.

- **`Qwen2VLForConditionalGenerationWithAudio`** (`modeling_qwen2_vl.py:2190`) — extends `Qwen2VLPreTrainedModel` with `self.audio_module = WhisperAudioEncoder(config.audio_config)`. Audio embeddings are injected at the `<|audio_pad|>` token positions before the LM forward pass (line ~2492).

To enable audio, load a `Qwen2VLConfig` with `use_audio=True` and attach a `Qwen2VLAudioConfig`:
```python
cfg = Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
cfg.use_audio = True
cfg.audio_config = Qwen2VLAudioConfig()
model = Qwen2VLForConditionalGenerationWithAudio.from_pretrained(..., config=cfg)
```

### Processor / tokenizer

The standard `Qwen2VLProcessor` is extended with three new special tokens and a custom Jinja chat template that handles `content[].type == "audio"` by inserting `<|audio_start|><|audio_pad|><|audio_end|>`. The extended processor is saved to HF as `mdmy/whisper_asr_finetuning` (subfolder `qwen_w_audio_processor`).

### Audio pipeline (`src/utils.py`)

- `convert_np_to_log_mel_spectrogram` — converts raw 1D audio to a Whisper-style log-mel spectrogram (80 bins, 3000 frames for 30 s), including chunking via `pad_and_split`.
- `format_data(sample)` — converts a `speechbrain/LargeScaleASR` row into the OpenAI conversation format (system + user with audio bytes + assistant with transcription).
- Hard-coded constants match Whisper's defaults: `SAMPLE_RATE=16000`, `N_FFT=400`, `HOP_LENGTH=160`, `N_MELS=80`.

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_audio_image_processor.ipynb` | Builds and tests the extended processor; pushes it to HF |
| `notebooks/02_finetuning.ipynb` | Two-stage fine-tuning: Stage 1 trains only `audio_projection`; Stage 2 adds QLoRA on `q_proj`/`v_proj` |
| `notebooks/03_testing.ipynb` | End-to-end inference on `reading_test.wav` using the fine-tuned model |

## Fine-Tuning Stages

**Stage 1 (projection-only):** Freeze all parameters except `audio_module.audio_projection`. Train with `trl.SFTTrainer` on `speechbrain/LargeScaleASR` (small shards). Checkpoint: `mdmy/qwen2-vl-audio-projection-sft` / local `e2e_sft_post_audio_projection_sft/`.

**Stage 2 (QLoRA e2e):** Load Stage 1 checkpoint, apply 4-bit quantization (`BitsAndBytesConfig`) and LoRA (`r=8`, `lora_alpha=16`, targets `q_proj`/`v_proj`). Final checkpoint: `mdmy/audio-capable-qwen2-vl`.

## HF Hub

Models and processors live under `mdmy/whisper_asr_finetuning` and `mdmy/audio-capable-qwen2-vl`. Local artifacts are cached under `hf/`. The `.env` file must supply `HF_TOKEN` before any `api.upload_folder` or `from_pretrained` calls to private/gated repos.

## Audio Input Format

Audio must be resampled to 16 kHz before passing to the processor. In `03_testing.ipynb`:
```python
waveform, sr = torchaudio.load("reading_test.wav")
waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16_000)
waveform = waveform.clamp(-1.0, 1.0).to(torch.float32)
# Serialize to BytesIO WAV and pass as audio_bytes in the conversation dict
```

## Important Notes

- The local `transformers/` fork **must** be installed (`pip install -e transformers/`) — the upstream package does not contain `Qwen2VLForConditionalGenerationWithAudio` or `Qwen2VLAudioConfig`.
- `format_data` is duplicated across `src/utils.py` and several notebooks; `src/utils.py` is the canonical location.
- The `Qwen2VLAudioConfig._name_or_path` field is hardcoded to an absolute local path (`configuration_qwen2_vl.py:48`) — this is a known issue that may cause warnings when loading configs on other machines.
- On macOS, `DEVICE` falls back to `"cpu"` (`src/utils.py:24`); GPU training requires a CUDA machine (Lambda Labs / cloud).
