from pathlib import Path

TTS_DIR = Path("models/tts")
TTS_FILES = [
    (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx",
        TTS_DIR / "en_US-lessac-high.onnx",
    ),
    (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx.json",
        TTS_DIR / "en_US-lessac-high.onnx.json",
    ),
]


def download_file(url: str, path: Path) -> None:
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"Skip existing: {path}")
        return

    print(f"Downloading: {url}")
    print(f"To: {path}")

    urllib.request.urlretrieve(url, path)

    print(f"Saved: {path}")


def main() -> None:
    TTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nDownloading TTS model...")
    for url, path in TTS_FILES:
        download_file(url, path)

    print("\nModel download step complete.")
    print("Next command:")
    print("  python scripts/check_env.py")


if __name__ == "__main__":
    main()
