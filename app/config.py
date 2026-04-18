import os
from pathlib import Path

def _bool_env(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).lower() in ("1", "true", "yes")


class ModelConfig:
    # ── LLaMA 3.1 8B (QLoRA fine-tuned, Hindi → Haryanvi) ─────────────────────
    LLM_BASE_MODEL_ID: str = os.getenv(
        "LLM_BASE_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"
    )

    LLM_ADAPTER_DIR: Path = Path(
        os.getenv("LLM_ADAPTER_DIR", "./model_weights/llm/adapter")
    )

    LLM_PROMPT_TEMPLATE: str = os.getenv(
        "LLM_PROMPT_TEMPLATE",
        "Translate the following Hindi sentence to Haryanvi dialect.\n\nHindi: {text}\nHaryanvi:",
    )

    LLM_MAX_NEW_TOKENS: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    LLM_LOAD_IN_4BIT: bool = _bool_env("LLM_LOAD_IN_4BIT", "true")

    DEVICE: str = os.getenv("DEVICE", "auto")

    # ── Coqui VITS TTS ──────────────────────────────────────────────────────────
    TTS_CHECKPOINT: Path = Path(
        os.getenv("TTS_CHECKPOINT", "./model_weights/tts/best_model_16731.pth")
    )

    TTS_CONFIG: Path = Path(
        os.getenv("TTS_CONFIG", "./model_weights/tts/config.json")
    )

    # ── Server ──────────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8080"))
    TMP_AUDIO_DIR: Path = Path(os.getenv("TMP_AUDIO_DIR", "/tmp/audio_outputs"))
