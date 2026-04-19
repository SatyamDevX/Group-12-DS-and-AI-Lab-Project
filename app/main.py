# CHANGED: monkey-patch kept — still needed because coqui-tts imports
# isin_mps_friendly from transformers.pytorch_utils which was removed in
# transformers 5.x. This runs before any TTS import.
from transformers import pytorch_utils as _pu
import torch
if not hasattr(_pu, "isin_mps_friendly"):
    _pu.isin_mps_friendly = lambda e, t: torch.isin(e, t.unsqueeze(0) if t.ndim == 0 else t)

import asyncio
import base64
import logging
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager

import soundfile as sf
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask

from app.config import ModelConfig

from app.models.translator import translate, load_llm, load_tokenizer
from app.models.tts_model import load_tts, synthesize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_tokenizer = _llm = _tts = None
_models_ready = False
_loading_error = None
_llm_lock = threading.Lock()
_tts_lock = threading.Lock()


def _load_models_background():
    """Runs in a background thread — server starts immediately without waiting.

    Model objects are written once here and shared by request handlers. Request
    handlers use locks around inference because these model runtimes are not
    guaranteed to be thread-safe.
    """
    global _tokenizer, _llm, _tts, _models_ready, _loading_error
    try:
        logger.info("=== Background model loading started ===")
        _tokenizer = load_tokenizer()
        _llm = load_llm()
        _tts = load_tts()
        ModelConfig.TMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        _models_ready = True
        logger.info("=== All models ready ===")
    except Exception as exc:
        _loading_error = str(exc)
        logger.error("Model loading failed: %s", exc, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_load_models_background, daemon=True)
    t.start()
    logger.info("Server started. Models loading in background thread.")
    yield
    shutil.rmtree(ModelConfig.TMP_AUDIO_DIR, ignore_errors=True)


app = FastAPI(title="Hindi → Haryanvi TTS Pipeline", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _require_models():
    if _loading_error:
        raise HTTPException(500, f"Model loading failed: {_loading_error}")
    if not _models_ready:
        raise HTTPException(503, "Models are still loading. Retry in a moment.")


def _cleanup_old_audio():
    max_age = ModelConfig.TMP_AUDIO_MAX_AGE_SECONDS
    if max_age <= 0:
        return

    ModelConfig.TMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - max_age
    for path in ModelConfig.TMP_AUDIO_DIR.glob("*.wav"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to clean temp audio file: %s", path, exc_info=True)


def _translate_locked(text: str) -> str:
    with _llm_lock:
        return translate(text, _llm, _tokenizer)


def _synthesize_locked(text: str, output_path: str) -> str:
    with _tts_lock:
        return synthesize(_tts, text, output_path)


# ── Request / Response models ──────────────────────────────────────────────────

class TextIn(BaseModel):
    text: str

# ADDED: separate request model for TTS-only endpoint
# Accepts Haryanvi text directly so you can test TTS without going through translation
class HaryanviIn(BaseModel):
    text: str  # expects Haryanvi text directly

# ADDED: base64 audio response model for pipeline and tts endpoints
class AudioResponse(BaseModel):
    audio_base64: str
    sample_rate: int


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health(response: Response):
    if _loading_error:
        response.status_code = 500
    return {
        "status": "error" if _loading_error else "ok" if _models_ready else "loading",
        "models_ready": _models_ready,
        "llm_backend": ModelConfig.LLM_BACKEND,
        "tts_backend": ModelConfig.TTS_BACKEND,
        "error": _loading_error,
    }


# ── Translation only ───────────────────────────────────────────────────────────

@app.post("/api/translate")
async def translate_endpoint(req: TextIn):
    """
    Hindi → Haryanvi text translation using the configured LLM backend.
    Test this independently to verify translation quality.

    Example:
        {"text": "मुझे पता नहीं कि वह कहाँ गया"}
    Expected:
        {"hindi": "...", "haryanvi": "म्हने बेरा कोन्या के वो कड़े गया"}
    """
    _require_models()
    if not req.text.strip():
        raise HTTPException(400, "Input text is empty.")
    loop = asyncio.get_running_loop()

    haryanvi = await loop.run_in_executor(None, _translate_locked, req.text)
    return {"hindi": req.text, "haryanvi": haryanvi}


# ── TTS only ───────────────────────────────────────────────────────────────────

@app.post("/api/tts")
async def tts_endpoint(req: HaryanviIn):
    """
    Haryanvi text → speech (WAV file download).
    Test TTS model independently by passing Haryanvi text directly.

    Example:
        {"text": "म्हने बेरा कोन्या के वो कड़े गया"}
    Returns: WAV audio file
    """
    _require_models()
    if not req.text.strip():
        raise HTTPException(400, "Input text is empty.")
    _cleanup_old_audio()
    audio_id = str(uuid.uuid4())
    path = ModelConfig.TMP_AUDIO_DIR / f"{audio_id}.wav"
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _synthesize_locked, req.text, str(path))
    return FileResponse(
        path,
        media_type="audio/wav",
        filename="haryanvi_tts.wav",
        background=BackgroundTask(path.unlink, missing_ok=True),
    )


# ADDED: /api/tts/base64 — returns audio as base64 JSON instead of file download
# Useful for frontend apps or API clients that can't handle file downloads easily
@app.post("/api/tts/base64")
async def tts_base64_endpoint(req: HaryanviIn):
    """
    Haryanvi text → speech returned as base64-encoded WAV in JSON.
    Useful for web frontends or clients that prefer JSON responses.

    Example:
        {"text": "म्हने बेरा कोन्या के वो कड़े गया"}
    Returns:
        {"audio_base64": "...", "sample_rate": 22050}
    """
    _require_models()
    if not req.text.strip():
        raise HTTPException(400, "Input text is empty.")
    _cleanup_old_audio()
    audio_id = str(uuid.uuid4())
    path = ModelConfig.TMP_AUDIO_DIR / f"{audio_id}.wav"
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _synthesize_locked, req.text, str(path))
    with open(path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    # CHANGED: read sample rate from written file for accuracy
    with sf.SoundFile(str(path)) as sf_file:
        sr = sf_file.samplerate
    path.unlink(missing_ok=True)
    return {"audio_base64": audio_b64, "sample_rate": sr}


# ── Full pipeline ──────────────────────────────────────────────────────────────

@app.post("/api/pipeline")
async def pipeline(req: TextIn):
    """
    Full pipeline: Hindi text → Haryanvi translation → speech audio.
    Returns translation text + audio download URL.

    Example:
        {"text": "आज बहुत गर्मी है"}
    Returns:
        {"hindi": "...", "haryanvi": "...", "audio_id": "...", "audio_url": "..."}
    """
    _require_models()
    if not req.text.strip():
        raise HTTPException(400, "Input text is empty.")
    loop = asyncio.get_running_loop()

    haryanvi = await loop.run_in_executor(None, _translate_locked, req.text)

    _cleanup_old_audio()
    audio_id = str(uuid.uuid4())
    path = ModelConfig.TMP_AUDIO_DIR / f"{audio_id}.wav"
    await loop.run_in_executor(None, _synthesize_locked, haryanvi, str(path))

    return {
        "hindi": req.text,
        "haryanvi": haryanvi,
        "audio_id": audio_id,
        "audio_url": f"/api/audio/{audio_id}",
    }


# ADDED: /api/pipeline/base64 — same as pipeline but returns audio as base64 JSON
# Avoids needing a second request to /api/audio/{id} for clients that want everything in one call
@app.post("/api/pipeline/base64")
async def pipeline_base64(req: TextIn):
    """
    Full pipeline returning everything in one JSON response.
    Hindi text → Haryanvi translation + base64 WAV audio.

    Example:
        {"text": "आज बहुत गर्मी है"}
    Returns:
        {"hindi": "...", "haryanvi": "...", "audio_base64": "...", "sample_rate": 22050}
    """
    _require_models()
    if not req.text.strip():
        raise HTTPException(400, "Input text is empty.")
    loop = asyncio.get_running_loop()

    haryanvi = await loop.run_in_executor(None, _translate_locked, req.text)
    _cleanup_old_audio()
    audio_id = str(uuid.uuid4())
    path = ModelConfig.TMP_AUDIO_DIR / f"{audio_id}.wav"
    await loop.run_in_executor(None, _synthesize_locked, haryanvi, str(path))

    with open(path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    with sf.SoundFile(str(path)) as sf_file:
        sr = sf_file.samplerate
    path.unlink(missing_ok=True)

    return {
        "hindi": req.text,
        "haryanvi": haryanvi,
        "audio_base64": audio_b64,
        "sample_rate": sr,
    }


# ── Audio file serving ─────────────────────────────────────────────────────────

@app.get("/api/audio/{audio_id}")
async def serve_audio(audio_id: str):
    """Serve a previously generated audio file by its ID."""
    try:
        uuid.UUID(audio_id)
    except ValueError:
        raise HTTPException(400, "Invalid audio ID.")
    path = ModelConfig.TMP_AUDIO_DIR / f"{audio_id}.wav"
    if not path.exists():
        raise HTTPException(404, "Audio not found or expired.")
    return FileResponse(path, media_type="audio/wav", filename="haryanvi_speech.wav")


# ── Static frontend (must be last — catches all unmatched routes) ──────────────
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
