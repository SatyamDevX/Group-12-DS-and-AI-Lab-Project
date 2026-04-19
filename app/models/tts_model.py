"""
Haryanvi text to WAV with env-selected Coqui TTS backend.

Supported backends:
  TTS_BACKEND=vits -> Coqui VITS checkpoint.
  TTS_BACKEND=xtts -> XTTS-v2 checkpoint with speaker reference audio.
"""
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

from app.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadedTTS:
    key: str
    label: str
    backend: str
    model: Any
    config: Any = None
    speaker_wav: str | None = None
    language: str | None = None
    sample_rate: int | None = None
    native_xtts: bool = False


TTS_MODEL_LABELS = {
    "vits": "Fine-tuned VITS",
    "xtts": "XTTS-v2",
}

TTS_MODEL_ALIASES = {
    "vits": "vits",
    "coqui_vits": "vits",
    "haryanvi_vits": "vits",
    "xtts": "xtts",
    "xtts_v2": "xtts",
    "xtts-v2": "xtts",
    "coqui_xtts": "xtts",
    "coqui_xtts_v2": "xtts",
}


def normalize_tts_model(model_key: str | None = None) -> str:
    key = (model_key or ModelConfig.DEFAULT_TTS_MODEL).strip().lower()
    normalized = TTS_MODEL_ALIASES.get(key)
    if not normalized:
        raise ValueError(
            "tts_model must be one of: " + ", ".join(sorted(TTS_MODEL_LABELS))
        )
    return normalized


def available_tts_models() -> list[dict[str, str]]:
    keys = []
    for raw_key in ModelConfig.TTS_MODELS.split(","):
        raw_key = raw_key.strip()
        if not raw_key:
            continue
        key = normalize_tts_model(raw_key)
        if key not in keys:
            keys.append(key)

    if not keys:
        keys = [normalize_tts_model(ModelConfig.DEFAULT_TTS_MODEL)]

    return [{"key": key, "label": TTS_MODEL_LABELS[key]} for key in keys]


def default_tts_model() -> str:
    default = normalize_tts_model(ModelConfig.DEFAULT_TTS_MODEL)
    configured = {item["key"] for item in available_tts_models()}
    if default not in configured:
        raise ValueError(
            f"DEFAULT_TTS_MODEL={default} is not present in TTS_MODELS."
        )
    return default


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


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _resolve_vits_artifacts() -> tuple[Path, Path]:
    if ModelConfig.TTS_REPO_ID:
        return (
            _download(ModelConfig.TTS_REPO_ID, ModelConfig.TTS_CHECKPOINT_FILENAME),
            _download(ModelConfig.TTS_REPO_ID, ModelConfig.TTS_CONFIG_FILENAME),
        )

    return (
        _require_file(ModelConfig.TTS_CHECKPOINT, "TTS checkpoint"),
        _require_file(ModelConfig.TTS_CONFIG, "TTS config"),
    )


def _optional_xtts_file(path: Path, label: str) -> Path | None:
    return _require_file(path, label) if path.exists() else None


def _resolve_xtts_artifacts() -> dict[str, Path | None]:
    if ModelConfig.XTTS_CHECKPOINT_PATH or ModelConfig.XTTS_CONFIG_PATH:
        if not ModelConfig.XTTS_CHECKPOINT_PATH or not ModelConfig.XTTS_CONFIG_PATH:
            raise ValueError(
                "Set both XTTS_CHECKPOINT_PATH and XTTS_CONFIG_PATH, or use "
                "XTTS_MODEL_DIR/XTTS_REPO_ID instead."
            )
        return {
            "checkpoint": _require_file(
                ModelConfig.XTTS_CHECKPOINT_PATH,
                "XTTS checkpoint",
            ),
            "config": _require_file(ModelConfig.XTTS_CONFIG_PATH, "XTTS config"),
            "vocab": (
                _require_file(ModelConfig.XTTS_VOCAB_PATH, "XTTS vocab")
                if ModelConfig.XTTS_VOCAB_PATH
                else None
            ),
            "speakers": (
                _require_file(ModelConfig.XTTS_SPEAKERS_PATH, "XTTS speakers file")
                if ModelConfig.XTTS_SPEAKERS_PATH
                else None
            ),
        }

    if ModelConfig.XTTS_MODEL_DIR:
        model_dir = _require_file(ModelConfig.XTTS_MODEL_DIR, "XTTS model directory")
        return {
            "checkpoint": _require_file(
                model_dir / ModelConfig.XTTS_CHECKPOINT_FILENAME,
                "XTTS checkpoint",
            ),
            "config": _require_file(
                model_dir / ModelConfig.XTTS_CONFIG_FILENAME,
                "XTTS config",
            ),
            "vocab": (
                _optional_xtts_file(model_dir / ModelConfig.XTTS_VOCAB_FILENAME, "XTTS vocab")
                if ModelConfig.XTTS_VOCAB_FILENAME
                else None
            ),
            "speakers": (
                _optional_xtts_file(
                    model_dir / ModelConfig.XTTS_SPEAKERS_FILENAME,
                    "XTTS speakers file",
                )
                if ModelConfig.XTTS_SPEAKERS_FILENAME
                else None
            ),
            "dvae": (
                _optional_xtts_file(model_dir / ModelConfig.XTTS_DVAE_FILENAME, "XTTS DVAE")
                if ModelConfig.XTTS_DVAE_FILENAME
                else None
            ),
            "mel_stats": (
                _optional_xtts_file(
                    model_dir / ModelConfig.XTTS_MEL_STATS_FILENAME,
                    "XTTS mel stats",
                )
                if ModelConfig.XTTS_MEL_STATS_FILENAME
                else None
            ),
        }

    if ModelConfig.XTTS_REPO_ID:
        artifacts: dict[str, Path | None] = {
            "checkpoint": _download(
                ModelConfig.XTTS_REPO_ID,
                ModelConfig.XTTS_CHECKPOINT_FILENAME,
            ),
            "config": _download(
                ModelConfig.XTTS_REPO_ID,
                ModelConfig.XTTS_CONFIG_FILENAME,
            ),
            "vocab": None,
            "speakers": None,
            "dvae": None,
            "mel_stats": None,
        }
        optional_filenames = {
            "vocab": ModelConfig.XTTS_VOCAB_FILENAME,
            "speakers": ModelConfig.XTTS_SPEAKERS_FILENAME,
            "dvae": ModelConfig.XTTS_DVAE_FILENAME,
            "mel_stats": ModelConfig.XTTS_MEL_STATS_FILENAME,
        }
        for key, filename in optional_filenames.items():
            if not filename:
                continue
            try:
                artifacts[key] = _download(ModelConfig.XTTS_REPO_ID, filename)
            except Exception:
                logger.info("Optional XTTS artifact not found: %s", filename)
        return artifacts

    raise ValueError(
        "Set XTTS_MODEL_NAME, XTTS_REPO_ID, XTTS_MODEL_DIR, or explicit "
        "XTTS_CHECKPOINT_PATH/XTTS_CONFIG_PATH when TTS_BACKEND=xtts."
    )


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

    if ModelConfig.XTTS_MODEL_DIR and ModelConfig.XTTS_SPEAKER_WAV_FILENAME:
        return _require_file(
            ModelConfig.XTTS_MODEL_DIR / ModelConfig.XTTS_SPEAKER_WAV_FILENAME,
            "XTTS speaker wav",
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
    return LoadedTTS(
        key="vits",
        label=TTS_MODEL_LABELS["vits"],
        backend="vits",
        model=model,
    )


def _load_xtts(use_gpu: bool) -> LoadedTTS:
    from TTS.api import TTS

    speaker_wav = _resolve_xtts_speaker_wav()

    if ModelConfig.XTTS_LOADER not in {"api", "native"}:
        raise ValueError("XTTS_LOADER must be one of: api, native")

    explicit_xtts_repo = bool(os.getenv("XTTS_REPO_ID"))
    if ModelConfig.XTTS_MODEL_NAME and not (
        explicit_xtts_repo
        or ModelConfig.XTTS_MODEL_DIR
        or ModelConfig.XTTS_CHECKPOINT_PATH
    ):
        logger.info("Loading XTTS model name: %s", ModelConfig.XTTS_MODEL_NAME)
        model = TTS(ModelConfig.XTTS_MODEL_NAME, progress_bar=False, gpu=use_gpu)
        native_xtts = False
        config_obj = None
        sample_rate = None
    else:
        artifacts = _resolve_xtts_artifacts()
        checkpoint = artifacts["checkpoint"]
        config = artifacts["config"]
        vocab = artifacts["vocab"]
        speakers = artifacts["speakers"]
        logger.info("Loading XTTS checkpoint from %s", checkpoint)

        if ModelConfig.XTTS_LOADER == "api":
            model = TTS(
                model_path=str(checkpoint),
                config_path=str(config),
                progress_bar=False,
                gpu=use_gpu,
            )
            native_xtts = False
            config_obj = None
            sample_rate = None
        else:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts

            config_obj = XttsConfig()
            config_obj.load_json(str(config))
            model = Xtts.init_from_config(config_obj)
            load_kwargs = {
                "checkpoint_path": str(checkpoint),
                "eval": True,
                "use_deepspeed": ModelConfig.XTTS_USE_DEEPSPEED,
            }
            if vocab:
                load_kwargs["vocab_path"] = str(vocab)
            if speakers:
                load_kwargs["speaker_file_path"] = str(speakers)
            model.load_checkpoint(config_obj, **load_kwargs)
            if use_gpu:
                model.cuda()
            native_xtts = True
            sample_rate = int(
                getattr(getattr(config_obj, "audio", None), "output_sample_rate", 24000)
            )

    logger.info("XTTS ready with language=%s and speaker=%s", ModelConfig.XTTS_LANGUAGE, speaker_wav)
    return LoadedTTS(
        key="xtts",
        label=TTS_MODEL_LABELS["xtts"],
        backend="xtts",
        model=model,
        config=config_obj,
        speaker_wav=str(speaker_wav),
        language=ModelConfig.XTTS_LANGUAGE,
        sample_rate=sample_rate,
        native_xtts=native_xtts,
    )


def load_tts(model_key: str | None = None) -> LoadedTTS:
    backend = normalize_tts_model(model_key)
    configured = {item["key"] for item in available_tts_models()}
    if backend not in configured:
        raise ValueError(
            f"TTS model '{backend}' is not enabled. Set TTS_MODELS to include it."
        )

    use_gpu = _use_gpu()
    logger.info("TTS backend=%s device=%s", backend, "GPU" if use_gpu else "CPU")

    if backend == "vits":
        return _load_vits(use_gpu)
    return _load_xtts(use_gpu)


def synthesize(tts_model: LoadedTTS | Any, text: str, output_path: str) -> str:
    if isinstance(tts_model, LoadedTTS) and tts_model.backend == "xtts":
        if tts_model.native_xtts:
            import soundfile as sf

            wav = tts_model.model.synthesize(
                text=text,
                config=tts_model.config,
                speaker_wav=tts_model.speaker_wav,
                language=tts_model.language,
                gpt_cond_len=ModelConfig.XTTS_GPT_COND_LEN,
                temperature=ModelConfig.XTTS_TEMPERATURE,
                speed=ModelConfig.XTTS_SPEED,
            )["wav"]
            if hasattr(wav, "detach"):
                wav = wav.detach().cpu().numpy()
            sf.write(output_path, wav, tts_model.sample_rate or 24000)
        else:
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
