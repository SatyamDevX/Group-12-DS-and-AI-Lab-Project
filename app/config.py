import os
from pathlib import Path


def _bool_env(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).lower() in ("1", "true", "yes")


def _int_env(key: str, default: str) -> int:
    return int(os.getenv(key, default))


def _float_env(key: str, default: str) -> float:
    return float(os.getenv(key, default))


class ModelConfig:
    # LLM_BACKEND:
    #   gguf    -> llama-cpp loads a GGUF file from local disk or Hugging Face.
    #   hf_lora -> Transformers loads a base model and optional PEFT/LoRA adapter.
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "gguf").lower()

    # ── GGUF backend ───────────────────────────────────────────────────────────
    LLM_GGUF_MODEL_PATH: Path | None = (
        Path(os.environ["LLM_GGUF_MODEL_PATH"])
        if os.getenv("LLM_GGUF_MODEL_PATH")
        else None
    )
    LLM_GGUF_DIR: Path = Path(os.getenv("LLM_GGUF_DIR", "./model_weights/llm/base"))
    LLM_GGUF_REPO_ID: str | None = os.getenv("LLM_GGUF_REPO_ID")
    LLM_GGUF_FILENAME: str = os.getenv(
        "LLM_GGUF_FILENAME", "gemma-4-E4B-it-Q4_K_S.gguf"
    )
    LLM_GGUF_CHAT_FORMAT: str = os.getenv("LLM_GGUF_CHAT_FORMAT", "gemma")
    LLM_GGUF_N_CTX: int = _int_env("LLM_GGUF_N_CTX", "2048")
    LLM_GGUF_N_THREADS: int = _int_env("LLM_GGUF_N_THREADS", "8")
    LLM_GGUF_N_GPU_LAYERS: int = _int_env("LLM_GGUF_N_GPU_LAYERS", "0")

    # ── Hugging Face Transformers + LoRA backend ───────────────────────────────
    LLM_BASE_MODEL_ID: str = os.getenv(
        "LLM_BASE_MODEL_ID",
        os.getenv("LLM_HF_BASE_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    )
    LLM_HF_BASE_MODEL_ID: str = os.getenv("LLM_HF_BASE_MODEL_ID", LLM_BASE_MODEL_ID)

    LLM_ADAPTER_DIR: Path = Path(
        os.getenv("LLM_ADAPTER_DIR", "./model_weights/llm/adapter")
    )
    LLM_HF_ADAPTER_ID: str | None = os.getenv("LLM_HF_ADAPTER_ID")
    LLM_USE_LOCAL_ADAPTER: bool = _bool_env("LLM_USE_LOCAL_ADAPTER", "true")

    LLM_PROMPT_TEMPLATE: str = os.getenv(
        "LLM_PROMPT_TEMPLATE",
        "Translate the following Hindi sentence to Haryanvi dialect.\n\nHindi: {text}\nHaryanvi:",
    )
    LLM_MAX_NEW_TOKENS: int = _int_env("LLM_MAX_NEW_TOKENS", "256")
    LLM_TEMPERATURE: float = _float_env("LLM_TEMPERATURE", "0.1")
    LLM_LOAD_IN_4BIT: bool = _bool_env("LLM_LOAD_IN_4BIT", "true")

    # DEVICE: auto | cuda | cpu
    DEVICE: str = os.getenv("DEVICE", "auto")

    # TTS_BACKEND:
    #   vits -> single-speaker Coqui VITS checkpoint.
    #   xtts -> XTTS-v2 checkpoint with speaker reference audio.
    TTS_BACKEND: str = os.getenv("TTS_BACKEND", "vits").lower()

    # ── Coqui VITS TTS ──────────────────────────────────────────────────────────
    TTS_CHECKPOINT: Path = Path(
        os.getenv("TTS_CHECKPOINT", "./model_weights/tts/best_model_16731.pth")
    )

    TTS_CONFIG: Path = Path(
        os.getenv("TTS_CONFIG", "./model_weights/tts/config.json")
    )
    TTS_REPO_ID: str | None = os.getenv("TTS_REPO_ID")
    TTS_CHECKPOINT_FILENAME: str = os.getenv(
        "TTS_CHECKPOINT_FILENAME", "best_model_16731.pth"
    )
    TTS_CONFIG_FILENAME: str = os.getenv("TTS_CONFIG_FILENAME", "config.json")

    # ── Coqui XTTS-v2 TTS ───────────────────────────────────────────────────────
    XTTS_MODEL_NAME: str | None = os.getenv("XTTS_MODEL_NAME")
    XTTS_REPO_ID: str | None = os.getenv("XTTS_REPO_ID")
    XTTS_CHECKPOINT_FILENAME: str = os.getenv("XTTS_CHECKPOINT_FILENAME", "model.pth")
    XTTS_CONFIG_FILENAME: str = os.getenv("XTTS_CONFIG_FILENAME", "config.json")
    XTTS_VOCAB_FILENAME: str | None = os.getenv("XTTS_VOCAB_FILENAME", "vocab.json")
    XTTS_SPEAKERS_FILENAME: str | None = os.getenv(
        "XTTS_SPEAKERS_FILENAME", "speakers_xtts.pth"
    )
    XTTS_SPEAKER_WAV_PATH: Path | None = (
        Path(os.environ["XTTS_SPEAKER_WAV_PATH"])
        if os.getenv("XTTS_SPEAKER_WAV_PATH")
        else None
    )
    XTTS_SPEAKER_WAV_REPO_ID: str | None = os.getenv("XTTS_SPEAKER_WAV_REPO_ID")
    XTTS_SPEAKER_WAV_FILENAME: str | None = os.getenv("XTTS_SPEAKER_WAV_FILENAME")
    XTTS_LANGUAGE: str = os.getenv("XTTS_LANGUAGE", "hi")

    # ── Server ──────────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = _int_env("PORT", "8080")
    TMP_AUDIO_DIR: Path = Path(os.getenv("TMP_AUDIO_DIR", "/tmp/audio_outputs"))
    TMP_AUDIO_MAX_AGE_SECONDS: int = _int_env("TMP_AUDIO_MAX_AGE_SECONDS", "3600")
