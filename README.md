# Hindi to Haryanvi TTS

FastAPI app for Hindi to Haryanvi translation plus Haryanvi speech synthesis.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`.

## Docker Deployment

The Docker image does not include `model_weights/`. Deployments should pull
custom model artifacts from Hugging Face at container startup.

1. Create an env file:

```bash
cp .env.example .env
```

2. Edit `.env` and set:

```bash
HF_TOKEN=hf_xxx
LLM_BACKEND=gguf
LLM_GGUF_REPO_ID=your-org/hindi-haryanvi-gguf
LLM_GGUF_FILENAME=gemma-4-E4B-it-Q4_K_S.gguf
TTS_BACKEND=xtts
XTTS_REPO_ID=your-org/haryanvi-xtts-v2
XTTS_CHECKPOINT_FILENAME=model.pth
XTTS_CONFIG_FILENAME=config.json
XTTS_SPEAKER_WAV_FILENAME=reference.wav
XTTS_LANGUAGE=hi
# COQUI_TOS_AGREED=1  # set after reviewing the XTTS-v2 license terms
```

For the original VITS model instead of XTTS:

```bash
TTS_BACKEND=vits
TTS_REPO_ID=your-org/haryanvi-vits
TTS_CHECKPOINT_FILENAME=best_model_16731.pth
TTS_CONFIG_FILENAME=config.json
```

3. Build and run:

```bash
docker compose up --build
```

Open `http://localhost:8080`.

To run without Compose:

```bash
docker build -t hindi-haryanvi-tts .
docker run --rm -p 8080:8080 --env-file .env \
  -v hf-cache:/cache/huggingface \
  -v audio-cache:/tmp/audio_outputs \
  hindi-haryanvi-tts
```

For a LoRA deployment instead of GGUF:

```bash
pip install -r requirements-lora.txt

LLM_BACKEND=hf_lora
LLM_HF_BASE_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
LLM_HF_ADAPTER_ID=your-org/hindi-haryanvi-lora
```

## Modal GPU Deployment

For limited live demos with free Modal credits, use Modal for the heavy model
container. This repo includes [modal_app.py](modal_app.py), which serves the
existing FastAPI app on an L4 GPU and caches Hugging Face downloads in a Modal
Volume.

Recommended architecture:

```text
Modal = model inference endpoint
Railway = optional public frontend / lightweight wrapper
```

For the fastest college demo, deploying the full app on Modal is fine:

```bash
pip install modal
modal setup
```

Create the Modal secret used by `modal_app.py`:

```bash
modal secret create hindi-haryanvi-secrets \
  HF_TOKEN=hf_xxx \
  TRANSLATION_MODELS=gemma_gguf,llama_lora \
  DEFAULT_TRANSLATION_MODEL=gemma_gguf \
  LLM_BACKEND=gguf \
  LLM_GGUF_REPO_ID=your-org/hindi-haryanvi-gguf \
  LLM_GGUF_FILENAME=your-model.Q4_K_S.gguf \
  LLM_GGUF_CHAT_FORMAT=gemma \
  LLM_GGUF_N_GPU_LAYERS=-1 \
  LLM_MAX_NEW_TOKENS=64 \
  LLM_LORA_BASE_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct \
  LLM_LORA_ADAPTER_ID=Satyam-Srivastava/hindi-to-haryanvi-translation-llama3-qlora \
  TTS_BACKEND=vits \
  TTS_REPO_ID=your-org/haryanvi-vits \
  TTS_CHECKPOINT_FILENAME=best_model_16731.pth \
  TTS_CONFIG_FILENAME=config.json
```

The frontend includes a translation model dropdown:

```text
gemma_gguf -> your uploaded Gemma GGUF model
llama_lora -> meta-llama/Meta-Llama-3.1-8B-Instruct + Satyam-Srivastava adapter
```

The LoRA option is loaded lazily on first use. Because the base LLaMA model is
gated, add `HF_TOKEN` to the Modal secret and make sure the token has access to
`meta-llama/Meta-Llama-3.1-8B-Instruct`.

After you fine-tune XTTS-v2 and upload it, switch the TTS settings to:

```bash
modal secret create hindi-haryanvi-secrets \
  HF_TOKEN=hf_xxx \
  TRANSLATION_MODELS=gemma_gguf,llama_lora \
  DEFAULT_TRANSLATION_MODEL=gemma_gguf \
  LLM_BACKEND=gguf \
  LLM_GGUF_REPO_ID=your-org/hindi-haryanvi-gguf \
  LLM_GGUF_FILENAME=your-model.Q4_K_S.gguf \
  LLM_GGUF_CHAT_FORMAT=gemma \
  LLM_GGUF_N_GPU_LAYERS=-1 \
  LLM_MAX_NEW_TOKENS=64 \
  TTS_BACKEND=xtts \
  XTTS_REPO_ID=your-org/haryanvi-xtts-v2 \
  XTTS_CHECKPOINT_FILENAME=model.pth \
  XTTS_CONFIG_FILENAME=config.json \
  XTTS_VOCAB_FILENAME=vocab.json \
  XTTS_SPEAKER_WAV_FILENAME=reference.wav \
  XTTS_LANGUAGE=hi \
  COQUI_TOS_AGREED=1
```

To upload the current local GGUF and VITS files to Hugging Face:

```bash
export HF_TOKEN=hf_xxx
python scripts/upload_current_models_to_hf.py
```

The script uploads:

```text
model_weights/llm/base/gemma-4-E4B-it-Q4_K_S.gguf
model_weights/tts/best_model_16731.pth
model_weights/tts/config.json
```

Set `HF_NAMESPACE`, `LLM_GGUF_REPO_ID`, `TTS_REPO_ID`, or `HF_PRIVATE=true` if
you want custom repo names or private repos.

Deploy:

```bash
modal deploy modal_app.py
```

Modal prints a URL like:

```text
https://<workspace>--hindi-haryanvi-tts-api.modal.run
```

Useful endpoints:

```bash
curl https://<workspace>--hindi-haryanvi-tts-api.modal.run/health

curl -X POST https://<workspace>--hindi-haryanvi-tts-api.modal.run/api/pipeline/base64 \
  -H "Content-Type: application/json" \
  -d '{"text":"आज बहुत गर्मी है"}'
```

Credit-saving defaults in `modal_app.py`:

```text
GPU: L4
min_containers: 0
max_containers: 1
scaledown_window: 5 minutes
```

For lower cost but slower inference, change `gpu="L4"` to `gpu="T4"` in
`modal_app.py`. For faster inference that uses credits faster, use `gpu="A10G"`.

## Model Loading

Model selection is controlled by environment variables. The default mode uses
the bundled local GGUF file if it exists.

### Local GGUF

```bash
export LLM_BACKEND=gguf
export LLM_GGUF_MODEL_PATH=./model_weights/llm/base/gemma-4-E4B-it-Q4_K_S.gguf
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

You can also set `LLM_GGUF_DIR` and `LLM_GGUF_FILENAME` instead of a full path.

### Hugging Face GGUF

Upload the `.gguf` file to a Hugging Face model repo, then run:

```bash
export HF_TOKEN=hf_xxx
export LLM_BACKEND=gguf
export LLM_GGUF_REPO_ID=your-org/hindi-haryanvi-gguf
export LLM_GGUF_FILENAME=gemma-4-E4B-it-Q4_K_S.gguf
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Hugging Face Base Model + LoRA Adapter

Use this when you want the fine-tuned LLaMA/PEFT adapter path:

```bash
export HF_TOKEN=hf_xxx
export LLM_BACKEND=hf_lora
export LLM_HF_BASE_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
export LLM_HF_ADAPTER_ID=your-org/hindi-haryanvi-lora
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

If `LLM_HF_ADAPTER_ID` is not set, the app uses the local
`LLM_ADAPTER_DIR` path when it exists.

### Hugging Face VITS TTS

Upload `best_model_16731.pth` and `config.json` to a Hugging Face model repo:

```bash
export HF_TOKEN=hf_xxx
export TTS_BACKEND=vits
export TTS_REPO_ID=your-org/haryanvi-vits
export TTS_CHECKPOINT_FILENAME=best_model_16731.pth
export TTS_CONFIG_FILENAME=config.json
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

If `TTS_REPO_ID` is not set, the app uses `TTS_CHECKPOINT` and `TTS_CONFIG`
from local disk.

### Hugging Face XTTS-v2 TTS

For a college project demo, upload the fine-tuned XTTS-v2 files and one speaker
reference WAV to a private Hugging Face model repo:

```bash
export HF_TOKEN=hf_xxx
export TTS_BACKEND=xtts
export XTTS_REPO_ID=your-org/haryanvi-xtts-v2
export XTTS_CHECKPOINT_FILENAME=model.pth
export XTTS_CONFIG_FILENAME=config.json
export XTTS_VOCAB_FILENAME=vocab.json
export XTTS_SPEAKERS_FILENAME=speakers_xtts.pth
export XTTS_SPEAKER_WAV_FILENAME=reference.wav
export XTTS_LANGUAGE=hi
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

If the speaker WAV is in a separate repo, use:

```bash
export XTTS_SPEAKER_WAV_REPO_ID=your-org/haryanvi-speaker-refs
export XTTS_SPEAKER_WAV_FILENAME=reference.wav
```

You can also use the stock Coqui XTTS model for comparison:

```bash
export TTS_BACKEND=xtts
export XTTS_MODEL_NAME=tts_models/multilingual/multi-dataset/xtts_v2
export XTTS_SPEAKER_WAV_REPO_ID=your-org/haryanvi-speaker-refs
export XTTS_SPEAKER_WAV_FILENAME=reference.wav
export XTTS_LANGUAGE=hi
```

## Useful Env Vars

```bash
export DEVICE=auto                  # auto, cpu, or cuda
export LLM_MAX_NEW_TOKENS=256
export LLM_TEMPERATURE=0.1
export TTS_BACKEND=xtts             # vits or xtts
export TMP_AUDIO_DIR=/tmp/audio_outputs
export TMP_AUDIO_MAX_AGE_SECONDS=3600
```

## Uploading Model Files to Hugging Face

Use private repos unless the model licenses and dataset permissions allow public
release.

```bash
huggingface-cli login
huggingface-cli repo create your-org/hindi-haryanvi-gguf --type model --private
huggingface-cli upload your-org/hindi-haryanvi-gguf ./model_weights/llm/base/gemma-4-E4B-it-Q4_K_S.gguf

huggingface-cli repo create your-org/haryanvi-vits --type model --private
huggingface-cli upload your-org/haryanvi-vits ./model_weights/tts/best_model_16731.pth
huggingface-cli upload your-org/haryanvi-vits ./model_weights/tts/config.json

huggingface-cli repo create your-org/haryanvi-xtts-v2 --type model --private
huggingface-cli upload your-org/haryanvi-xtts-v2 ./xtts_finetuned/model.pth
huggingface-cli upload your-org/haryanvi-xtts-v2 ./xtts_finetuned/config.json
huggingface-cli upload your-org/haryanvi-xtts-v2 ./xtts_finetuned/vocab.json
huggingface-cli upload your-org/haryanvi-xtts-v2 ./xtts_finetuned/reference.wav
```
