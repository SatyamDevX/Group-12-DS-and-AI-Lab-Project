# 📘 Regional Dialect Synthesis Pipeline  
**Hindi → Haryanvi (Bangru) → Speech**  
*Group 12 • DS & AI Lab Project • Milestone 6*

---

# 🅰️ A. Overview

## 🔹 Purpose
This project presents a complete AI-powered pipeline that:
- Translates **Hindi text into Haryanvi (Bangru dialect)**
- Converts the translated text into **natural speech audio**

### 🎯 Objectives
- Support **low-resource regional dialects**
- Enable **speech synthesis for Indian languages**
- Provide a **real-world deployable ML system**

---

## 🔹 Architecture Summary

```
Hindi Input
   ↓
LLM Translator (Gemma / LLaMA)
   ↓
Haryanvi Text (Validated)
   ↓
TTS Model (Coqui VITS)
   ↓
Speech Output (WAV)
```

### Key Design
- Two-stage modular pipeline
- Decoupled translation and speech synthesis
- Stateless API design

---

## 🔹 Deployed Components

| Component | Description |
|----------|------------|
| Frontend | Web UI (HTML, JS) |
| Backend | FastAPI server |
| Models | LLM + VITS |
| Hosting | Google Cloud VM |
| API | REST endpoints |

---

## B. Technical Documentation — Summary

The table below maps each required section to a concise description of its contents, followed by expanded detail sections.

| Section | Description |
|---|---|
| **1. Environment Setup** | Python 3.12, PyTorch 2.4, FastAPI backend, Uvicorn server. Full pinned dependency list in `requirements.txt`. Hardware: NVIDIA A100 / T4 GPU (16–40 GB VRAM) for training; CPU deployment supported for inference. |
| **2. Data Pipeline** | Translation corpus: ~5,594 parallel Hindi–Haryanvi sentence pairs (filtered 3–100 words). Audio corpus: ~5,500 utterances filtered to 2,714 valid samples. Multi-script transliteration (Bengali, Urdu → Devanagari) with NFC normalisation. Datasets stored locally; see `docs/licenses.md`. |
| **3. Model Architecture** | Two-stage pipeline: (A) LLM translator — LLaMA 3.1 8B with QLoRA (r=16, α=32) for research; Gemma 4 E4B Q4_K_S GGUF for deployment. (B) TTS — Coqui VITS (GAN + Flow-based decoder), 22,050 Hz output, 173-char vocab. |
| **4. Training Summary** | LLM (QLoRA): 4 epochs, cosine LR 1e-4, NF4 4-bit quantisation, BF16 compute, assistant-only loss masking. TTS (VITS): 300 epochs / ≈50,700 steps, generator & discriminator LR 1e-5, FP16, batch size 16. Warm-started from SYSPIN Hindi Female. |
| **5. Evaluation Summary** | TTS: Mel loss 32.0 → 13.36, KL divergence 3.70 → 1.50 (stable by epoch 25), discriminator loss stable at ≈2.75 (no mode collapse). LLM BLEU / Exact-Match invalidated by a LoRA adapter loading bug; qualitative dialect quality judged acceptable for deployment. |
| **6. Inference Pipeline** | Input Hindi text → Gemma GGUF (prompt-engineered, temperature 0.1) → Devanagari validation loop → Haryanvi text → VITS synthesiser → 22,050 Hz WAV. Exposed as async FastAPI endpoints with thread-pool executors. |
| **7. Deployment Details** | Platform: GCP Virtual Machine. Hosting: FastAPI + Uvicorn (port 8080, single worker). Weights loaded locally from `./model_weights/`. UI: static HTML served at `/`. Example: `curl -X POST http://<host>:8080/api/pipeline -H 'Content-Type: application/json' -d '{"text":"तुम कहाँ जा रहे हो?"}'` |
| **8. System Design Considerations** | Modular two-stage design — translator and TTS are decoupled and independently callable. Models load on a background thread so the server accepts requests immediately. Async handlers + thread-pool executors prevent blocking. Designed for horizontal scale-out behind a load balancer. |
| **9. Error Handling & Monitoring** | HTTP 503 while models load, 400 for empty input, 404 for missing audio IDs, 500 on model failure. `/health` endpoint reports readiness and loading errors. Structured logging via Python `logging` module; stdout captured by systemd / screen session on the VM. |
| **10. Reproducibility Checklist** | Pinned `requirements.txt`, fixed VITS checkpoint (`best_model_16731.pth`), fixed GGUF weights, deterministic decoding (temperature 0.1, top_p 0.95), seed-controlled training notebooks, sample config files under `app/config.py`, and documented env vars (`LLM_BASE_MODEL_ID`, `TTS_CHECKPOINT`, `TTS_CONFIG`, `DEVICE`). |

---

## 1. Environment Setup

### Runtime requirements

- Python 3.12 (64-bit)
- PyTorch 2.4.0 with CUDA 12.1 build (CPU inference also supported)
- FastAPI 0.111, Uvicorn 0.29 — async HTTP server
- `coqui-tts` 0.27.5 — VITS training & inference
- `llama-cpp-python` 0.3.20 — GGUF quantised LLM runtime
- `transformers` 5.5.4, `peft` 0.19.1, `bitsandbytes` 0.44.1 — LLaMA QLoRA stack

### Hardware

- **Training:** NVIDIA A100 / T4 GPU, 16–40 GB VRAM
- **Inference (deployed):** GCP VM, CPU is sufficient for Gemma GGUF + VITS; GPU optional for lower latency

### Install & run

```bash
# Clone the repository
git clone <repo-url> hindi-haryanvi-tts-vm
cd hindi-haryanvi-tts-vm

# Create a Python 3.12 virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt

# Launch the server
uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

*A complete pinned dependency list is available in `requirements.txt`. The server binds to `0.0.0.0:8080` and serves both the JSON API and the static HTML frontend.*

---

## 2. Data Pipeline

### Translation corpus (Module A)

- 5,594 parallel Hindi–Haryanvi sentence pairs (Devanagari script)
- Length filter: 3–100 words per sentence to eliminate padding inefficiency
- Formatted as a causal-LM chat template (system / user / assistant) with assistant-only loss masking

### Audio corpus (Module B)

- ~5,500 raw Haryanvi utterances → 2,714 valid samples after de-duplication and length filtering (3–25 words)
- Multi-script clean-up: Bengali transliterated via `indic-transliteration`; Urdu via a hand-tuned mapping
- Unicode NFC normalisation applied globally to prevent silent tokenizer-attention breakage
- Phonemisers bypassed (`espeak-ng` Hindi mispronounces Bangru retroflexes); input fed through `basic_cleaners` into a 173-character vocabulary

### Storage & licensing

- Datasets stored locally under the project; license and attribution details are documented in `docs/licenses.md`
- No personally identifiable data is included in the training sets

---

## 3. Model Architecture

The system is a decoupled two-stage pipeline. Stage A is an LLM that maps Hindi → Haryanvi in Devanagari. Stage B is a VITS TTS that maps Haryanvi → 22,050 Hz waveform. The output of Stage A is plain UTF-8 text, which becomes the direct input to Stage B — no intermediate phonemiser.

```
          ┌───────────────────────────┐
          │      Hindi Text Input     │
          └─────────────┬─────────────┘
                        ▼
          ┌───────────────────────────┐
          │  Gemma 4 E4B (GGUF)       │
          │  Prompt + few-shot + low- │
          │  entropy decoding         │
          └─────────────┬─────────────┘
                        ▼
          ┌───────────────────────────┐
          │  Haryanvi (Devanagari)    │
          │  — validation retry loop  │
          └─────────────┬─────────────┘
                        ▼
          ┌───────────────────────────┐
          │  Coqui VITS (GAN TTS)     │
          │  warm-started from SYSPIN │
          │  Hindi Female             │
          └─────────────┬─────────────┘
                        ▼
          ┌───────────────────────────┐
          │  22,050 Hz WAV audio      │
          └───────────────────────────┘
```

### Chosen hyperparameters

| Parameter | LLM (QLoRA) | TTS (VITS) |
|---|---|---|
| Base model | LLaMA 3.1 8B Instruct | SYSPIN/vits_Hindi_Female |
| Precision | 4-bit NF4 + BF16 compute | FP16 mixed-precision |
| Learning rate | 1e-4 (cosine) | 1e-5 (G & D) |
| Epochs / Steps | 4 epochs | 300 epochs (≈50,700 steps) |
| Batch size | Per-device 4, grad-accum 4 | 16 |
| LoRA / adapter rank | r=16, α=32 | n/a |
| Vocabulary size | Base tokenizer | 173 characters |

---

## 4. Training Summary

*Full training procedures are in Milestone 4. Summary below:*

### LLM — LLaMA 3.1 8B Instruct (QLoRA)

- 4 epochs, cosine learning-rate schedule, peak LR 1e-4
- 4-bit NormalFloat (NF4) weights with double quantisation; BF16 compute dtype
- LoRA adapters injected into `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Assistant-only causal cross-entropy loss (system / padding tokens masked with `-100`)
- Optimiser: paged AdamW 8-bit

### TTS — Coqui VITS

- 300 epochs, approximately 50,700 training steps
- Warm-started from `SYSPIN/vits_Hindi_Female` — waveform decoder, flow and duration predictor weights transferred at 100%; text-encoder embeddings partially re-initialised for the new 173-char vocabulary
- Generator and discriminator learning rates both 1e-5 to maintain adversarial equilibrium
- FP16 mixed precision, batch size 16 on 16 GB VRAM
- AdamW optimiser, β=(0.8, 0.99)

---

## 5. Evaluation Summary

*See Milestone 5 for full evaluation. Headline metrics:*

### TTS convergence (VITS)

- **Mel loss:** 32.0 → 13.36 (monotonic decrease, no plateau)
- **KL divergence:** 3.70 → 1.50 within 25 epochs (rapid posterior alignment)
- **Discriminator loss:** stabilised at ≈2.75 — no mode collapse observed across 300 epochs
- **Feature loss:** saturated around epoch 165 at 6.31 — no further optimisation signal

### LLM translation

- Generative metrics (BLEU, Exact Match) were invalidated by an inference-script bug: dynamic LoRA adapters failed to reload between evaluation batches, producing identical static scores across divergent runs
- Qualitative evaluation confirmed acceptable dialect quality for deployment; formal re-evaluation requires an updated `PeftModel` inference script and is marked as future work

### Production model

- Gemma 4 E4B Q4_K_S GGUF selected for deployment — dialect quality acceptable at a fraction of LLaMA 3.1 8B latency and memory footprint

---

## 6. Inference Pipeline

Data flow: client → FastAPI → thread-pool executor → translator (`llama-cpp-python`) → Devanagari validation loop → TTS (Coqui VITS) → WAV file → response. The translation and synthesis steps are both CPU-friendly and run on dedicated executor threads to keep the asyncio event loop unblocked.

### Reference inference function

```python
# app/main.py — excerpt
@app.post('/api/pipeline')
async def pipeline(req: TextIn):
    _require_models()
    if not req.text.strip():
        raise HTTPException(400, 'Input text is empty.')

    loop = asyncio.get_running_loop()

    # Step 1 — translate Hindi → Haryanvi
    haryanvi = await loop.run_in_executor(
        None, translate, req.text, _llm, _tokenizer
    )

    # Step 2 — synthesize speech
    audio_id = str(uuid.uuid4())
    path = ModelConfig.TMP_AUDIO_DIR / f'{audio_id}.wav'
    await loop.run_in_executor(None, synthesize, _tts, haryanvi, str(path))

    return {
        'hindi':     req.text,
        'haryanvi':  haryanvi,
        'audio_id':  audio_id,
        'audio_url': f'/api/audio/{audio_id}',
    }
```

---

## 7. Deployment Details

### Platform

- Google Cloud Platform — single Virtual Machine instance
- VITS TTS model is additionally published as a Hugging Face Space: `huggingface.co/spaces/Satyam-Srivastava/tts-haryanvi-bangru-text-to-audio`

### Hosting

- FastAPI application (`app/main.py`) served by Uvicorn on port 8080, `--workers 1` to match the single-process model-loading model
- Weights loaded locally from `./model_weights/` — no network dependency at request time (`HF_HUB_OFFLINE=1`)
- Static HTML frontend mounted at `/` via `StaticFiles`

### How to interact

- **Browser UI:** `http://<VM-IP>:8080/`
- **REST API:** `POST /api/translate`, `/api/tts`, `/api/pipeline`, `/api/pipeline/base64`, `GET /health`, `GET /api/audio/{id}`

### Example request

```bash
curl -X POST http://<VM-IP>:8080/api/pipeline \
     -H 'Content-Type: application/json' \
     -d '{"text":"तुम कहाँ जा रहे हो?"}'

# Response
{
  "hindi":     "तुम कहाँ जा रहे हो?",
  "haryanvi":  "तू कड़े जा रया सै?",
  "audio_id":  "123e4567-e89b-12d3-a456-426614174000",
  "audio_url": "/api/audio/123e4567-e89b-12d3-a456-426614174000"
}
```

---

## 8. System Design Considerations

### Architecture

- Clean separation of translator and synthesiser — either module can be replaced or scaled independently
- Thin FastAPI layer that only handles request validation, async dispatch, and response serialisation
- Static frontend served from the same process — zero extra infrastructure

### Scalability

- Stateless HTTP handlers — server instances can be placed behind a load balancer and scaled horizontally
- Temporary audio files are written to a per-process directory and cleaned up on shutdown
- Model loading happens on a background thread so the server accepts connections immediately; readiness is reported via `/health`

### Modularity

- Translator lives in `app/models/translator.py` — swap Gemma GGUF for another LLM by changing `load_llm()` and `translate()`
- Synthesiser lives in `app/models/tts_model.py` — swap VITS for another TTS by changing `load_tts()` and `synthesize()`
- All configuration (weight paths, temp dir) centralised in `app/config.py` → `ModelConfig`

### Data flow

- One-way, stateless flow: input text → translator output → TTS output. No persistent database — audio files are kept only until retrieval.

---

## 9. Error Handling & Monitoring

### HTTP error semantics

- **400 Bad Request** — empty input text or malformed audio UUID
- **404 Not Found** — audio file already cleaned up / never existed
- **500 Internal Server Error** — background model loading failed (details in `/health`)
- **503 Service Unavailable** — models still loading; clients should poll `/health` then retry

### Latency controls

- Async endpoints + `run_in_executor` ensure the event loop never blocks on CPU-heavy generation
- Low-entropy decoding (temperature 0.1, top_p 0.95) bounds LLM latency; VITS is a single forward pass

### Observability

- Python `logging` configured at `INFO` level with timestamps, levels, and logger names
- `/health` returns `{status, models_ready, error}` for external probes
- `stdout` captured by the hosting process (screen session or systemd journal on the VM)

---

## 10. Reproducibility Checklist

| Artifact | Location / Value |
|---|---|
| Requirements | `requirements.txt` (fully pinned, Python 3.12) |
| LLM weights | `model_weights/llm/` (Gemma 4 E4B Q4_K_S GGUF) |
| TTS checkpoint | `model_weights/tts/best_model_16731.pth` |
| TTS config | `model_weights/tts/config.json` |
| Config module | `app/config.py` — `ModelConfig` (paths, temp dir) |
| Env variables | `LLM_BASE_MODEL_ID`, `TTS_CHECKPOINT`, `TTS_CONFIG`, `DEVICE`, `HF_HUB_OFFLINE` |
| Decoding seed | Temperature 0.1, top_p 0.95, repeat_penalty 1.05 (deterministic) |
| Training notebooks | `notebooks/llama-text-pipe-final.ipynb`, `notebooks/coqui-tts-pipe-final.ipynb` |
| Training logs | `logs/trainer_0_log_2.txt`, `logs/experiment_summary.csv` |

*Following the table above, a new engineer can replicate the deployed system end-to-end: clone the repo, create a Python 3.12 virtualenv, install `requirements.txt`, drop the checkpoints into `model_weights/`, export the env vars, and launch Uvicorn.*


# 🅲 C. User Documentation

## Regional Dialect Synthesis Pipeline — Hindi → Haryanvi (Bangru) → Speech

*A guide for non-technical users • Group 12 • Milestone 6*

---

## App Overview

This application turns standard Hindi text into spoken Haryanvi (Bangru dialect). You type a sentence in Hindi, and the system does two things automatically:

1. It rewrites the sentence in Haryanvi.
2. It speaks that Haryanvi sentence out loud in a natural-sounding voice.

It is intended for anyone curious about regional Indian dialects — language learners, teachers, content creators, localisation teams, and researchers working on low-resource languages. No machine-learning background is required.

> **💡 What you can do with it**
>
> Enter a Hindi sentence, instantly read the Haryanvi version, and listen to or download the spoken audio. You can also use only the translator, or only the text-to-speech, depending on what you need.

---

## How You Enter Text

All interaction happens inside your web browser. The home page shows three tabs at the top:

- **▶ Full Pipeline:** Type Hindi — the app shows the Haryanvi translation and plays the audio.
- **🔤 Translate Only:** Type Hindi — get only the Haryanvi translation (no audio).
- **🔊 TTS Only:** Type Haryanvi directly — get only the spoken audio.

Below the text box you will see ready-made example sentences. Clicking any of these chips fills the input box automatically so you can try the system without typing.

Inputs are limited to 1,000 characters. You can press `Ctrl + Enter` (or `Cmd + Enter` on a Mac) to submit without reaching for the mouse.

---

## What You See And Hear

- **Haryanvi text:** Appears in a result card under your input. A **Copy** button next to it copies the text to your clipboard in one click.
- **Audio:** A built-in audio player appears once the speech is ready. Press play to listen, or click **Download WAV** to save the file to your computer.
- **Progress indicators:** While the Full Pipeline is running, you will see small status ticks for the translation step and the speech step, so you always know what is happening.
- **Error banner:** If anything goes wrong, a red banner drops down under the input box with a short explanation.

![Translation result after entering a Hindi sentence](../assets/translate.jpg)
*Figure 1 — Translation result displayed after the user enters a Hindi sentence.*

---

## Step-By-Step Instructions

### How to launch the app

1. Open a modern web browser (Chrome, Firefox, Safari, or Edge).
2. Enter the live URL in the address bar: `http://34.123.144.143:8080/` and press Enter.
3. Wait a few seconds on the first visit — if the banner says "Models loading…", leave the tab open and it will clear automatically.

> **💡 Tip**
>
> You do not need to install anything. Everything runs on the server — your browser is the only tool you need.

### How to interact with it

1. Choose the tab you want — **Full Pipeline**, **Translate Only**, or **TTS Only**.
2. Type (or paste) your text in the large input box, or click one of the example chips to autofill.
3. Click the **Run Pipeline** button (or press `Ctrl + Enter`).
4. Watch the progress indicators. The Haryanvi text appears first, then the audio player shows up once the voice is ready.
5. Press ▶ on the audio player to listen, or click **Download WAV** to save the file.
6. To try another sentence, either click **Clear** or simply overwrite the text and press **Run Pipeline** again.

### Example queries

| Field | Example |
|---|---|
| **Hindi input** | तुम कहाँ जा रहे हो? |
| **Haryanvi output** | तू कड़े जा रया सै? |
| **Audio output** | Downloadable 22,050 Hz `.wav` file spoken in Haryanvi (Bangru) |

*Other sentences you can try:*

- आज बहुत गर्मी है।
- क्या तुमने खाना खा लिया?
- हम कल गाँव जाएँगे।
- मुझे नहीं पता वो कहाँ गया।

---

## Screenshots — Example Interactions

The two screenshots below show the application in use.

![Translate Only tab](../assets/translate.jpg)
*Figure 2 — Translate Only tab. The user enters a Hindi sentence and receives the Haryanvi version instantly.*

![Full Pipeline on mobile](../assets/full-pipeline-working.jpg)
*Figure 3 — Full Pipeline view on a mobile-width browser. Translation appears first, followed by the audio player.*

---

## Troubleshooting

If something does not behave as expected, try the suggested fix below before reporting a bug.

| Symptom | What to try |
|---|---|
| Page does not load at all | Confirm the server is running and the port (8080) is open. Try `http://<VM-IP>:8080/` in a different browser. Make sure you are on a network that can reach the VM. |
| "Models are still loading" banner | The server starts immediately but models take up to a minute to load in the background. Wait ~60 seconds and press **Run** again, or refresh the page. |
| First request is very slow | This is expected. The first inference warms up the models; subsequent requests are noticeably faster. |
| No audio plays | Check your browser's sound / tab-mute icon. Click the **Download WAV** button to verify the file plays locally. If playback still fails, refresh the page. |
| Translation looks too much like Hindi | The deployed Gemma model is a fast approximation. Simplify or shorten the sentence, or try a variant phrasing. Very formal Hindi tends to leak into the output. |
| "Input text is empty" error | The input box has only whitespace. Type actual Devanagari text and try again. |
| Request returns 503 | The server restarted and is reloading models. Wait a minute and retry; no action needed from your side. |
| Error banner with HTTP 500 | A model failed to load on the server. Contact the project maintainer and share the timestamp shown in the banner. |

> **💡 Before asking for help**
>
> Refresh the page once, wait 60 seconds for the models to be ready, and try the example chips — they are known to work and are a quick way to confirm the server is healthy.

---

## Frequently Asked Questions

### Is my input stored anywhere?

The application does not keep your text or audio permanently. Generated audio files live only long enough for you to download them, and are cleaned up automatically.

### Can I use this on my phone?

Yes. The interface adapts to mobile screens. All you need is a browser and an internet connection to reach the server.

### Why does it sometimes sound a little like Hindi?

The voice model was warm-started from a Hindi speaker because the available Haryanvi audio dataset is small. For very common or formal words the voice can slip towards Hindi pronunciation. This is a known limitation and is being improved with more audio data.

### Can I use this in my own app?

Yes — the same server exposes a simple REST API (see the API Documentation). Developers can send JSON with their Hindi text and receive back the Haryanvi text and the audio.

# 🅳 D. API Documentation

## D. API Documentation — Summary

The table below answers the required **D. API Documentation** fields. Detailed endpoint reference follows in the sections after.

| Section | Example |
|---|---|
| **Base URL** | `http://34.123.144.143:8080` |
| **POST /api/pipeline** | **Body:** `{"text":"तुम कहाँ जा रहे हो?"}`<br>**Response:** `{"hindi":…,"haryanvi":…,"audio_id":…,"audio_url":"/api/audio/<id>"}` |
| **GET /health** | Health-check endpoint — returns `{"status":"ok","models_ready":true,"error":null}` |
| **Example curl request** | `curl -X POST -H "Content-Type: application/json" -d '{"text":"तुम कहाँ जा रहे हो?"}' http://34.123.144.143:8080/api/pipeline` |
| **Response format** | JSON keys and meanings documented per endpoint below. Audio returned either as a URL (`GET /api/audio/{id}`) or inline base64 via `/api/pipeline/base64` and `/api/tts/base64`. |

---

## 1. Introduction

This API exposes the deployed dialect-synthesis pipeline over HTTP. Clients send Hindi text and receive back Haryanvi text and/or synthesised speech. All requests are JSON; audio responses are returned either as downloadable WAV files or base64-encoded JSON payloads.

- **Base URL (live server):** `http://34.123.144.143:8080`
- **Alternative base URL (TTS Hugging Face Space):** `https://huggingface.co/spaces/Satyam-Srivastava/tts-haryanvi-bangru-text-to-audio`
- **Authentication:** none (public read-only demo endpoint)
- **Content type:** `application/json` for all POST bodies
- **CORS:** open — any origin may call the API

---

## 2. Server Startup Behaviour

Models are loaded on a background thread at server boot. The process accepts TCP connections immediately, but any endpoint that needs the models will respond with `503 Service Unavailable` until loading finishes. Always check `/health` first if you are starting a fresh client.

---

## 3. Endpoints At A Glance

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Server and model readiness probe |
| `POST` | `/api/translate` | Hindi → Haryanvi text translation |
| `POST` | `/api/tts` | Haryanvi text → downloadable `.wav` |
| `POST` | `/api/tts/base64` | Haryanvi text → base64 audio JSON |
| `POST` | `/api/pipeline` | Full pipeline — translation + audio URL |
| `POST` | `/api/pipeline/base64` | Full pipeline — translation + inline base64 audio |
| `GET` | `/api/audio/{audio_id}` | Fetch a previously generated `.wav` by UUID |

---

## 4. Endpoint Reference

### `GET /health`

Returns the current status of the server and the background model loader. Safe to poll; no side effects.

**Response (200 OK):**

```json
{
  "status":       "ok",        // "ok" when ready, "loading" otherwise
  "models_ready": true,
  "error":        null          // string with traceback if load failed
}
```

---

### `POST /api/translate`

Translates Hindi text into Haryanvi. Audio is not generated.

**Request body:**

```json
{ "text": "तुम कहाँ जा रहे हो?" }
```

**Response (200 OK):**

```json
{
  "hindi":    "तुम कहाँ जा रहे हो?",
  "haryanvi": "तू कड़े जा रया सै?"
}
```

---

### `POST /api/tts`

Synthesises spoken audio from Haryanvi text. The body must already be Haryanvi — translation is not performed here.

**Request body:**

```json
{ "text": "म्हने बेरा कोन्या के वो कड़े गया" }
```

**Response (200 OK):** a downloadable `audio/wav` file named `haryanvi_tts.wav`.

---

### `POST /api/tts/base64`

Same as `/api/tts`, but returns the audio inside a JSON envelope. Convenient for browser clients that do not want to handle binary responses.

**Request body:**

```json
{ "text": "म्हने बेरा कोन्या के वो कड़े गया" }
```

**Response (200 OK):**

```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA…",
  "sample_rate":  22050
}
```

---

### `POST /api/pipeline`

Runs the complete flow — translate Hindi to Haryanvi, synthesise Haryanvi audio, and return a URL to fetch the audio.

**Request body:**

```json
{ "text": "आज बहुत गर्मी है" }
```

**Response (200 OK):**

```json
{
  "hindi":     "आज बहुत गर्मी है",
  "haryanvi":  "आज भोत गर्मी सै",
  "audio_id":  "123e4567-e89b-12d3-a456-426614174000",
  "audio_url": "/api/audio/123e4567-e89b-12d3-a456-426614174000"
}
```

**Response field reference:**

| Field | Type | Meaning |
|---|---|---|
| `hindi` | string | Original Hindi input (echoed back for convenience). |
| `haryanvi` | string | Translated Devanagari Haryanvi text. |
| `audio_id` | string | UUID v4 identifier for the generated WAV file. |
| `audio_url` | string | Relative URL — GET it to download the WAV. |

---

### `POST /api/pipeline/base64`

Same as `/api/pipeline` but returns the audio inline so clients can get everything in a single round trip.

**Response (200 OK):**

```json
{
  "hindi":        "आज बहुत गर्मी है",
  "haryanvi":     "आज भोत गर्मी सै",
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA…",
  "sample_rate":  22050
}
```

**Response field reference:**

| Field | Type | Meaning |
|---|---|---|
| `hindi` | string | Original Hindi input. |
| `haryanvi` | string | Translated Haryanvi text. |
| `audio_base64` | string | WAV file contents, base64-encoded. |
| `sample_rate` | integer | Audio sample rate — always `22050`. |

---

### `GET /api/audio/{audio_id}`

Downloads a previously generated WAV file using the UUID returned by `/api/pipeline`.

- **Path parameter:** `audio_id` — a valid UUID v4 string
- **Success (200 OK):** `audio/wav` file named `haryanvi_speech.wav`
- **Errors:** `400` if the ID is not a valid UUID, `404` if the file has already been cleaned up

---

## 5. Example: End-To-End With curl

Minimal full-pipeline call as specified in the **D. API Documentation** template:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"text":"तुम कहाँ जा रहे हो?"}' \
     http://34.123.144.143:8080/api/pipeline
```

**Translate only:**

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"text":"तुम कहाँ जा रहे हो?"}' \
     http://34.123.144.143:8080/api/translate
```

**Health probe:**

```bash
curl http://34.123.144.143:8080/health
```

**Fetch a generated audio file** (after `/api/pipeline` returns an `audio_url`):

```bash
curl -o out.wav http://34.123.144.143:8080/api/audio/<audio_id>
```

---

## 6. Example: Python Client

```python
import base64, requests

BASE = "http://34.123.144.143:8080"

# Wait for models to be ready
while requests.get(f"{BASE}/health").json()["status"] != "ok":
    pass

# Full pipeline (inline base64 — one request, no second fetch)
resp = requests.post(
    f"{BASE}/api/pipeline/base64",
    json={"text": "तुम कहाँ जा रहे हो?"},
    timeout=60,
).json()

print("Hindi:   ", resp["hindi"])
print("Haryanvi:", resp["haryanvi"])

with open("out.wav", "wb") as f:
    f.write(base64.b64decode(resp["audio_base64"]))
```

---

## 7. Example: Browser JavaScript

```javascript
const BASE = 'http://34.123.144.143:8080';

const res  = await fetch(`${BASE}/api/pipeline/base64`, {
  method:  'POST',
  headers: { 'Content-Type': 'application/json' },
  body:    JSON.stringify({ text: 'तुम कहाँ जा रहे हो?' }),
});

const { hindi, haryanvi, audio_base64, sample_rate } = await res.json();
const audio = new Audio(`data:audio/wav;base64,${audio_base64}`);
audio.play();
```

---

## 8. Error Codes

| Status | When it occurs |
|---|---|
| `200 OK` | Normal success. |
| `400 Bad Request` | Input text is empty or whitespace only, or the supplied `audio_id` is not a valid UUID. |
| `404 Not Found` | Requested audio file does not exist or has already been cleaned up. |
| `500 Internal Server Error` | Background model loading failed completely. Call `GET /health` for the error trace. |
| `503 Service Unavailable` | Models are still loading in the background. Poll `/health` and retry once status becomes `"ok"`. |

**Error payloads follow the FastAPI convention:**

```json
{ "detail": "Input text is empty." }
```

---

## 9. Versioning And Limits

- Request body: max 1,000 characters of input text (enforced by the frontend; backend is tolerant but performance degrades).
- Sample rate: always 22,050 Hz, mono, 16-bit PCM WAV.
- No rate-limiting is currently enforced; avoid hammering the public server.
- Audio files are cleaned up when the server process terminates — fetch them promptly after a `/api/pipeline` call.

---

## 10. Changelog

- **v1.1** — added `/api/pipeline/base64` and `/api/tts/base64` for single-round-trip clients.
- **v1.0** — initial deployment with `/health`, `/api/translate`, `/api/tts`, `/api/pipeline`, `/api/audio/{id}`.


# 🅴 E. Licensing & Dataset References

- Code License: MIT (recommended)
- Datasets: Hindi–Haryanvi corpus, speech dataset
- Models: LLaMA, Gemma, Coqui TTS

---

# 🅵 F. Future Work

## Improvements
- Larger dataset
- Better dialect accuracy
- Improved evaluation metrics
- Multilingual expansion

## Limitations
- Small dataset
- Hindi influence in speech
- CPU latency

## Retraining Steps
1. Update dataset
2. Retrain LLM
3. Retrain TTS
4. Replace weights

---

## Maintainers
Group 12 — DS & AI Lab Project

---

# ✅ Final Note
This document provides complete details to understand, reproduce, deploy, and extend the system.



