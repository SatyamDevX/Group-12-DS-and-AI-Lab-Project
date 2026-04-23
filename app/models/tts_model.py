"""
Haryanvi text → WAV using Coqui VITS.
"""
import logging
import torch
from TTS.api import TTS
from app.config import ModelConfig

logger = logging.getLogger(__name__)


def load_tts() -> TTS:
    logger.info("Loading VITS from %s", ModelConfig.TTS_CHECKPOINT)


    if ModelConfig.DEVICE == "cpu":
        use_gpu = False
    else:
        use_gpu = torch.cuda.is_available()

    logger.info("TTS will use %s", "GPU" if use_gpu else "CPU")

    tts = TTS(
        model_path=str(ModelConfig.TTS_CHECKPOINT),
        config_path=str(ModelConfig.TTS_CONFIG),
        progress_bar=False,
        gpu=use_gpu,
    )

    logger.info("TTS ready")
    return tts


def synthesize(tts_model: TTS, text: str, output_path: str) -> str:
    tts_model.tts_to_file(text=text, file_path=output_path)
    return output_path
