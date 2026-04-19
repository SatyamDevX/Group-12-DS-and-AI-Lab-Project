"""Modal deployment entrypoint.

Recommended use for free-credit demos:
  modal secret create hindi-haryanvi-secrets HF_TOKEN=... LLM_GGUF_REPO_ID=... ...
  modal deploy modal_app.py

This serves the existing FastAPI app on Modal GPU. Modal scales to zero when
idle; keep `min_containers=0` to preserve credits.
"""
import modal


APP_NAME = "hindi-haryanvi-tts"
SECRET_NAME = "hindi-haryanvi-secrets"
HF_CACHE_PATH = "/cache/huggingface"


app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("hindi-haryanvi-hf-cache", create_if_missing=True)


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "build-essential",
        "cmake",
        "curl",
        "espeak-ng",
        "ffmpeg",
        "git",
        "libgomp1",
        "libsndfile1",
    )
    .env(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PIP_NO_CACHE_DIR": "1",
            "CC": "gcc",
            "CXX": "g++",
            "HF_HOME": HF_CACHE_PATH,
            "HUGGINGFACE_HUB_CACHE": f"{HF_CACHE_PATH}/hub",
            "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
            "TMP_AUDIO_DIR": "/tmp/audio_outputs",
            "DEVICE": "cuda",
            "LLM_BACKEND": "gguf",
            "TRANSLATION_MODELS": "gemma_gguf,llama_lora",
            "DEFAULT_TRANSLATION_MODEL": "gemma_gguf",
            "LLM_GGUF_N_GPU_LAYERS": "-1",
            "TTS_BACKEND": "vits",
            "TTS_MODELS": "vits,xtts",
            "DEFAULT_TTS_MODEL": "vits",
            "XTTS_LOADER": "native",
            "XTTS_LANGUAGE": "hi",
        }
    )
    .pip_install("modal")
    .pip_install_from_requirements("requirements-modal.txt")
    # Install the prebuilt CUDA 12.4 wheel. Building llama-cpp-python from
    # source on Modal is slow and can fail during nvcc compilation.
    .run_commands(
        "python -m pip install --no-cache-dir --prefer-binary "
        "--only-binary=llama-cpp-python "
        "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 "
        "llama-cpp-python==0.3.20"
    )
    .workdir("/root/app")
    .add_local_dir("app", remote_path="/root/app/app")
)


@app.function(
    image=image,
    gpu="L4",
    timeout=15 * 60,
    startup_timeout=20 * 60,
    scaledown_window=5 * 60,
    min_containers=0,
    max_containers=1,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    volumes={HF_CACHE_PATH: hf_cache},
)
@modal.asgi_app(label="api")
def fastapi_app():
    from app.main import app as fastapi

    return fastapi
