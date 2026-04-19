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
            "HF_HOME": HF_CACHE_PATH,
            "HUGGINGFACE_HUB_CACHE": f"{HF_CACHE_PATH}/hub",
            "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
            "TMP_AUDIO_DIR": "/tmp/audio_outputs",
            "DEVICE": "cuda",
            "LLM_BACKEND": "gguf",
            "LLM_GGUF_N_GPU_LAYERS": "-1",
            "TTS_BACKEND": "vits",
            "XTTS_LANGUAGE": "hi",
        }
    )
    .pip_install("modal")
    .pip_install_from_requirements("requirements.txt")
    # Rebuild llama-cpp-python with CUDA support so GGUF layers can run on GPU.
    .run_commands(
        'CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 '
        "python -m pip install --force-reinstall --no-cache-dir llama-cpp-python"
    )
    .add_local_dir("app", remote_path="/root/app/app")
    .workdir("/root/app")
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
