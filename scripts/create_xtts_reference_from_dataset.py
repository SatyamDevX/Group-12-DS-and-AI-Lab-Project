"""Create an XTTS speaker reference WAV from a Hugging Face audio dataset.

The default source is ankitdhiman/haryanvi-tts. The script selects one clean
8-13 second clip, writes it as reference.wav, and can optionally upload it to a
small Hugging Face model repo for Modal to use.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
import wave

from huggingface_hub import HfApi, get_token, hf_hub_download, upload_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "model_weights" / "xtts_reference" / "reference.wav"


def _bool_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


def _duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / float(rate)
    except (wave.Error, EOFError):
        sf = _require_soundfile()
        info = sf.info(str(path))
        return info.frames / float(info.samplerate)


def _require_soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "This audio format is not readable by Python's built-in wave module. "
            "Install soundfile with `.venv_deploy/bin/python -m pip install soundfile` "
            "and rerun the command."
        ) from exc
    return sf


def _copy_reference(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(source_path), "rb"):
                shutil.copyfile(source_path, output_path)
                return
        except (wave.Error, EOFError):
            pass

    sf = _require_soundfile()

    audio, sample_rate = sf.read(source_path)
    sf.write(output_path, audio, sample_rate, format="WAV")


def _trim_wav(source_path: Path, output_path: Path, start_seconds: float, seconds: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(source_path), "rb") as src:
            params = src.getparams()
            sample_rate = src.getframerate()
            start_frame = int(start_seconds * sample_rate)
            frame_count = int(seconds * sample_rate)
            src.setpos(min(start_frame, src.getnframes()))
            frames = src.readframes(frame_count)

        with wave.open(str(output_path), "wb") as dst:
            dst.setparams(params)
            dst.writeframes(frames)
        return
    except (wave.Error, EOFError):
        pass

    sf = _require_soundfile()
    info = sf.info(str(source_path))
    start_frame = int(start_seconds * info.samplerate)
    frame_count = int(seconds * info.samplerate)
    audio, sample_rate = sf.read(
        source_path,
        start=start_frame,
        frames=frame_count,
    )
    sf.write(output_path, audio, sample_rate, format="WAV")


def _score_row(row: dict, text_column: str) -> tuple[int, int]:
    text = str(row.get(text_column) or "")
    # Prefer plain sentence-like clips with enough text but no noisy metadata.
    punctuation_penalty = sum(text.count(char) for char in "#[]{}<>|/@\\")
    length_score = abs(len(text) - 85)
    return punctuation_penalty, length_score


def _row_text_length(row: dict[str, str], text_column: str) -> int:
    return len(str(row.get(text_column) or "").strip())


def _metadata_rows(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",|\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        return list(reader)


def _row_audio_path(row: dict[str, str], audio_files: set[str]) -> str | None:
    for key in (
        "filepath",
        "file_path",
        "file_name",
        "filename",
        "path",
        "audio_path",
        "audio",
        "speech",
    ):
        value = (row.get(key) or "").strip()
        if value:
            normalized = value.replace("\\", "/")
            if normalized in audio_files:
                return normalized
            basename_matches = [path for path in audio_files if path.endswith("/" + normalized) or path.endswith(normalized)]
            if basename_matches:
                return sorted(basename_matches)[0]
    return None


def _audio_only_rows(audio_files: set[str]) -> list[dict[str, str]]:
    return [
        {"file_name": audio_path, "text": ""}
        for audio_path in sorted(audio_files)
    ]


def create_reference(args: argparse.Namespace) -> Path:
    api = HfApi()
    print(f"Listing files for dataset {args.dataset_id}")
    files = api.list_repo_files(args.dataset_id, repo_type="dataset")
    audio_files = {
        path for path in files if Path(path).suffix.lower() in {".wav", ".flac", ".mp3"}
    }
    if not audio_files:
        raise RuntimeError(f"No audio files found in dataset {args.dataset_id}.")

    metadata_file = args.metadata_filename
    rows = None
    if not metadata_file:
        metadata_matches = [path for path in files if Path(path).name == "metadata.csv"]
        if metadata_matches:
            metadata_file = sorted(metadata_matches)[0]
        elif args.metadata_required:
            raise RuntimeError("Could not find metadata.csv in the dataset.")
        else:
            print("No metadata.csv found; scanning audio files directly.")
            rows = _audio_only_rows(audio_files)

    if rows is None:
        metadata_path = Path(
            hf_hub_download(
                repo_id=args.dataset_id,
                repo_type="dataset",
                filename=metadata_file,
            )
        )
        rows = _metadata_rows(metadata_path)
    if not rows:
        raise RuntimeError(f"No rows found in {metadata_file or 'audio file list'}.")

    candidates = []
    skipped_missing_audio = 0
    for row in rows:
        audio_path = _row_audio_path(row, audio_files)
        if not audio_path:
            skipped_missing_audio += 1
            continue
        text_len = _row_text_length(row, args.text_column)
        if text_len and text_len < args.min_text_chars:
            continue
        candidates.append((_score_row(row, args.text_column), row, audio_path))

    if not candidates:
        raise RuntimeError(
            "No metadata rows matched the text/audio filters. "
            f"Rows without matching audio: {skipped_missing_audio}."
        )

    candidates.sort(key=lambda item: item[0])
    checked = 0
    for _, row, audio_path in candidates[: args.max_scan]:
        checked += 1
        candidate_path = Path(
            hf_hub_download(
                repo_id=args.dataset_id,
                repo_type="dataset",
                filename=audio_path,
            )
        )
        try:
            duration = _duration_seconds(candidate_path)
        except wave.Error:
            continue

        if args.min_seconds <= duration <= args.max_seconds:
            _copy_reference(candidate_path, args.output)
            text = str(row.get(args.text_column) or "").strip()
            print(f"Wrote {args.output}")
            print(f"Source file: {audio_path}")
            print(f"Duration: {duration:.2f}s")
            if text:
                print(f"Text: {text}")
            return args.output

        if args.allow_trim and duration > args.max_seconds:
            trim_seconds = min(args.trim_seconds, duration)
            trim_start = min(args.trim_start_seconds, max(0.0, duration - trim_seconds))
            _trim_wav(candidate_path, args.output, trim_start, trim_seconds)
            trimmed_duration = _duration_seconds(args.output)
            text = str(row.get(args.text_column) or "").strip()
            print(f"Wrote trimmed reference {args.output}")
            print(f"Source file: {audio_path}")
            print(f"Source duration: {duration:.2f}s")
            print(f"Trim: start={trim_start:.2f}s duration={trimmed_duration:.2f}s")
            if text:
                print(f"Text: {text}")
            return args.output

        print(
            f"Skipping {audio_path}: {duration:.2f}s outside "
            f"{args.min_seconds}-{args.max_seconds}s"
        )

    raise RuntimeError(
        f"No clip found between {args.min_seconds} and {args.max_seconds} "
        f"seconds after downloading {checked} top-ranked candidates."
    )


def _resolve_namespace(api: HfApi, token: str | None, namespace: str | None) -> str:
    if namespace:
        return namespace
    whoami = api.whoami(token=token)
    name = whoami.get("name")
    if not name:
        raise RuntimeError("Could not determine Hugging Face username. Set HF_NAMESPACE.")
    return name


def upload_reference(args: argparse.Namespace, reference_path: Path) -> str:
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "A Hugging Face write token is required for --upload. Run "
            "`export HF_TOKEN=hf_xxx`, or run `.venv_deploy/bin/hf auth login`."
        )

    api = HfApi()
    namespace = _resolve_namespace(api, token, args.namespace)
    repo_id = args.repo_id or f"{namespace}/haryanvi-xtts-reference"
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=token,
    )
    upload_file(
        path_or_fileobj=str(reference_path),
        path_in_repo=args.path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"Uploaded {reference_path} -> {repo_id}/{args.path_in_repo}")
    return repo_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="ankitdhiman/haryanvi-tts")
    parser.add_argument("--metadata-filename", default=None)
    parser.add_argument("--metadata-required", action="store_true")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--min-seconds", type=float, default=8.0)
    parser.add_argument("--max-seconds", type=float, default=13.0)
    parser.add_argument("--min-text-chars", type=int, default=110)
    parser.add_argument("--max-scan", type=int, default=40)
    parser.add_argument("--allow-trim", action="store_true")
    parser.add_argument("--trim-seconds", type=float, default=10.0)
    parser.add_argument("--trim-start-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--namespace", default=os.getenv("HF_NAMESPACE"))
    parser.add_argument("--repo-id", default=os.getenv("XTTS_SPEAKER_WAV_REPO_ID"))
    parser.add_argument("--path-in-repo", default="reference.wav")
    parser.add_argument("--private", action="store_true", default=_bool_env("HF_PRIVATE"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    reference_path = create_reference(args)
    repo_id = upload_reference(args, reference_path) if args.upload else None

    print("\nUse these Modal secret values:")
    if repo_id:
        print(f"XTTS_SPEAKER_WAV_REPO_ID={repo_id}")
    else:
        print("XTTS_SPEAKER_WAV_REPO_ID=<upload reference.wav to a HF model repo>")
    print(f"XTTS_SPEAKER_WAV_FILENAME={args.path_in_repo}")
    print("XTTS_LANGUAGE=hi")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
