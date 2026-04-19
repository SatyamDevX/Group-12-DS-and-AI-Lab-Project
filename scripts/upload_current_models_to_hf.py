"""Upload the current local model artifacts to Hugging Face.

This helper publishes the deployable artifacts already present in
model_weights/:
  - GGUF translator model
  - VITS TTS checkpoint/config

It intentionally does not upload model_weights/ wholesale. The LoRA adapter is
not needed by the current Modal deployment because the app loads the GGUF file.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, upload_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GGUF = ROOT / "model_weights" / "llm" / "base" / "gemma-4-E4B-it-Q4_K_S.gguf"
DEFAULT_TTS_CHECKPOINT = ROOT / "model_weights" / "tts" / "best_model_16731.pth"
DEFAULT_TTS_CONFIG = ROOT / "model_weights" / "tts" / "config.json"


def _bool_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a file artifact: {path}")


def _repo_default(namespace: str, suffix: str) -> str:
    return f"{namespace}/{suffix}"


def _resolve_namespace(api: HfApi, token: str | None, namespace: str | None) -> str:
    if namespace:
        return namespace
    whoami = api.whoami(token=token)
    name = whoami.get("name")
    if not name:
        raise RuntimeError("Could not determine Hugging Face username. Set HF_NAMESPACE.")
    return name


def _upload(api: HfApi, token: str | None, repo_id: str, path: Path, repo_path: str) -> None:
    size_gb = path.stat().st_size / 1024**3
    print(f"Uploading {path} -> {repo_id}/{repo_path} ({size_gb:.2f} GiB)")
    upload_file(
        path_or_fileobj=str(path),
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )


def _print_modal_secret_command(args: argparse.Namespace, gguf_repo: str, tts_repo: str) -> None:
    print("\nUpload complete.")
    print("\nCreate/update the Modal secret with:")
    print(
        "modal secret create hindi-haryanvi-secrets "
        "\\\n  HF_TOKEN=$HF_TOKEN "
        "\\\n  LLM_BACKEND=gguf "
        f"\\\n  LLM_GGUF_REPO_ID={gguf_repo} "
        f"\\\n  LLM_GGUF_FILENAME={args.gguf_path.name} "
        "\\\n  LLM_GGUF_CHAT_FORMAT=gemma "
        "\\\n  LLM_GGUF_N_GPU_LAYERS=-1 "
        "\\\n  LLM_MAX_NEW_TOKENS=64 "
        "\\\n  TTS_BACKEND=vits "
        f"\\\n  TTS_REPO_ID={tts_repo} "
        f"\\\n  TTS_CHECKPOINT_FILENAME={args.tts_checkpoint.name} "
        f"\\\n  TTS_CONFIG_FILENAME={args.tts_config.name}"
    )
    print("\nThen deploy:")
    print("modal deploy modal_app.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=os.getenv("HF_NAMESPACE"))
    parser.add_argument("--gguf-repo-id", default=os.getenv("LLM_GGUF_REPO_ID"))
    parser.add_argument("--tts-repo-id", default=os.getenv("TTS_REPO_ID"))
    parser.add_argument("--gguf-path", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--tts-checkpoint", type=Path, default=DEFAULT_TTS_CHECKPOINT)
    parser.add_argument("--tts-config", type=Path, default=DEFAULT_TTS_CONFIG)
    parser.add_argument("--private", action="store_true", default=_bool_env("HF_PRIVATE"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.gguf_path = args.gguf_path.resolve()
    args.tts_checkpoint = args.tts_checkpoint.resolve()
    args.tts_config = args.tts_config.resolve()

    _require_file(args.gguf_path)
    _require_file(args.tts_checkpoint)
    _require_file(args.tts_config)

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required to create repos and upload artifacts. Run "
            "`export HF_TOKEN=hf_xxx` with a Hugging Face write token, then run "
            "this script again."
        )

    api = HfApi()
    namespace = _resolve_namespace(api, token, args.namespace)
    gguf_repo = args.gguf_repo_id or _repo_default(namespace, "hindi-haryanvi-gemma-gguf")
    tts_repo = args.tts_repo_id or _repo_default(namespace, "haryanvi-vits")

    print(f"Using Hugging Face namespace: {namespace}")
    print(f"GGUF repo: {gguf_repo}")
    print(f"TTS repo: {tts_repo}")
    print(f"Private repos: {args.private}")

    api.create_repo(repo_id=gguf_repo, repo_type="model", private=args.private, exist_ok=True, token=token)
    api.create_repo(repo_id=tts_repo, repo_type="model", private=args.private, exist_ok=True, token=token)

    _upload(api, token, gguf_repo, args.gguf_path, args.gguf_path.name)
    _upload(api, token, tts_repo, args.tts_checkpoint, args.tts_checkpoint.name)
    _upload(api, token, tts_repo, args.tts_config, args.tts_config.name)

    _print_modal_secret_command(args, gguf_repo, tts_repo)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
