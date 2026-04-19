"""
Haryanvi text to WAV with env-selected Coqui TTS backend.

Supported backends:
  TTS_BACKEND=vits -> Coqui VITS checkpoint.
  TTS_BACKEND=xtts -> XTTS-v2 checkpoint with speaker reference audio.
"""
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from app.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadedTTS:
    backend: str
    model: Any
    speaker_wav: str | None = None
    language: str | None = None


def _backend() -> str:
    backend = ModelConfig.TTS_BACKEND.lower()
    if backend not in {"vits", "xtts"}:
        raise ValueError("TTS_BACKEND must be one of: vits, xtts")
    return backend


def _use_gpu() -> bool:
    import torch

    if ModelConfig.DEVICE == "cpu":
        return False
    use_gpu = torch.cuda.is_available()
    if ModelConfig.DEVICE == "cuda" and not use_gpu:
        raise RuntimeError("DEVICE=cuda was requested, but CUDA is not available.")
    return use_gpu


def _download(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def _resolve_vits_artifacts() -> tuple[Path, Path]:
    if ModelConfig.TTS_REPO_ID:
        return (
            _download(ModelConfig.TTS_REPO_ID, ModelConfig.TTS_CHECKPOINT_FILENAME),
            _download(ModelConfig.TTS_REPO_ID, ModelConfig.TTS_CONFIG_FILENAME),
        )

    if not ModelConfig.TTS_CHECKPOINT.exists():
        raise FileNotFoundError(f"TTS checkpoint not found: {ModelConfig.TTS_CHECKPOINT}")
    if not ModelConfig.TTS_CONFIG.exists():
        raise FileNotFoundError(f"TTS config not found: {ModelConfig.TTS_CONFIG}")
    return ModelConfig.TTS_CHECKPOINT, ModelConfig.TTS_CONFIG


def _resolve_xtts_speaker_wav() -> Path:
    if ModelConfig.XTTS_SPEAKER_WAV_PATH:
        path = ModelConfig.XTTS_SPEAKER_WAV_PATH
        if not path.exists():
            raise FileNotFoundError(f"XTTS speaker wav not found: {path}")
        return path

    if ModelConfig.XTTS_SPEAKER_WAV_REPO_ID and ModelConfig.XTTS_SPEAKER_WAV_FILENAME:
        return _download(
            ModelConfig.XTTS_SPEAKER_WAV_REPO_ID,
            ModelConfig.XTTS_SPEAKER_WAV_FILENAME,
        )

    if ModelConfig.XTTS_REPO_ID and ModelConfig.XTTS_SPEAKER_WAV_FILENAME:
        return _download(ModelConfig.XTTS_REPO_ID, ModelConfig.XTTS_SPEAKER_WAV_FILENAME)

    raise ValueError(
        "XTTS requires a speaker reference. Set XTTS_SPEAKER_WAV_PATH, or set "
        "XTTS_SPEAKER_WAV_REPO_ID and XTTS_SPEAKER_WAV_FILENAME, or include "
        "XTTS_SPEAKER_WAV_FILENAME in XTTS_REPO_ID."
    )


def _load_vits(use_gpu: bool) -> LoadedTTS:
    from TTS.api import TTS

    checkpoint, config = _resolve_vits_artifacts()
    logger.info("Loading VITS from %s", checkpoint)
    model = TTS(
        model_path=str(checkpoint),
        config_path=str(config),
        progress_bar=False,
        gpu=use_gpu,
    )
    logger.info("VITS ready")
    return LoadedTTS(backend="vits", model=model)


def _load_xtts(use_gpu: bool) -> LoadedTTS:
    from TTS.api import TTS

    speaker_wav = _resolve_xtts_speaker_wav()

    if ModelConfig.XTTS_MODEL_NAME:
        logger.info("Loading XTTS model name: %s", ModelConfig.XTTS_MODEL_NAME)
        model = TTS(ModelConfig.XTTS_MODEL_NAME, progress_bar=False, gpu=use_gpu)
    elif ModelConfig.XTTS_REPO_ID:
        checkpoint = _download(ModelConfig.XTTS_REPO_ID, ModelConfig.XTTS_CHECKPOINT_FILENAME)
        config = _download(ModelConfig.XTTS_REPO_ID, ModelConfig.XTTS_CONFIG_FILENAME)
        # These files are used by XTTS configs/checkpoints when present. Download
        # them early so the local HF snapshot contains the full model folder.
        for filename in (
            ModelConfig.XTTS_VOCAB_FILENAME,
            ModelConfig.XTTS_SPEAKERS_FILENAME,
        ):
            if filename:
                try:
                    _download(ModelConfig.XTTS_REPO_ID, filename)
                except Exception:
                    logger.info("Optional XTTS artifact not found: %s", filename)

        logger.info("Loading XTTS checkpoint from %s", checkpoint)
        model = TTS(
            model_path=str(checkpoint),
            config_path=str(config),
            progress_bar=False,
            gpu=use_gpu,
        )
    else:
        raise ValueError("Set XTTS_MODEL_NAME or XTTS_REPO_ID when TTS_BACKEND=xtts")

    logger.info("XTTS ready with language=%s and speaker=%s", ModelConfig.XTTS_LANGUAGE, speaker_wav)
    return LoadedTTS(
        backend="xtts",
        model=model,
        speaker_wav=str(speaker_wav),
        language=ModelConfig.XTTS_LANGUAGE,
    )


def load_tts() -> LoadedTTS:
    backend = _backend()
    use_gpu = _use_gpu()
    logger.info("TTS backend=%s device=%s", backend, "GPU" if use_gpu else "CPU")

    if backend == "vits":
        return _load_vits(use_gpu)
    return _load_xtts(use_gpu)


def synthesize(tts_model: LoadedTTS | Any, text: str, output_path: str) -> str:
    if isinstance(tts_model, LoadedTTS) and tts_model.backend == "xtts":
        tts_model.model.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=tts_model.speaker_wav,
            language=tts_model.language,
            split_sentences=True,
        )
        return output_path

    model = tts_model.model if isinstance(tts_model, LoadedTTS) else tts_model
    model.tts_to_file(text=text, file_path=output_path)
    return output_path
