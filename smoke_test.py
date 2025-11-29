#!/usr/bin/env python3
"""
Offline Silero multi (v2_multi.pt) smoke test for Codex.

Checks:
- v2_multi.pt exists in the repo
- it can be loaded via torch.package.PackageImporter
- the loaded object has an .apply_tts attribute

No network, no actual TTS inference.
"""

import sys
from pathlib import Path

import torch


# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MULTI_MODEL_PATH = (
    BASE_DIR
    / "silero_cache"
    / "snakers4_silero-models_master"
    / "src"
    / "silero"
    / "model"
    / "v2_multi.pt"
)


# --------------------------------------------------------------------
# Loading Silero multi from .pt (no torch.hub, no network)
# --------------------------------------------------------------------

def load_silero_multi(model_path: Path):
    """
    Load Silero TTS multi model from a local .pt file.

    Uses torch.package.PackageImporter as recommended in Silero docs.
    """
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Silero multi model file not found at:\n  {model_path}"
        )

    # Keep CI lightweight on CPU
    torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))

    importer = torch.package.PackageImporter(str(model_path))
    model = importer.load_pickle("tts_models", "model")

    # Some Silero models have .to(), some may not
    if hasattr(model, "to"):
        model.to("cpu")

    return model


def run_smoke(model) -> None:
    """
    Minimal sanity checks without doing full TTS.
    """
    # Just basic structural checks
    if not hasattr(model, "apply_tts"):
        raise RuntimeError("Silero multi model has no .apply_tts attribute")

    print("Silero multi smoke test: model loaded, apply_tts present")


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

def main() -> int:
    print(f"[smoke] Using Silero multi model at: {MULTI_MODEL_PATH}")

    try:
        model = load_silero_multi(MULTI_MODEL_PATH)
    except Exception as e:
        print(f"[smoke] FAILED to load model: {e}", file=sys.stderr)
        return 1

    try:
        run_smoke(model)
    except Exception as e:
        print(f"[smoke] FAILED during smoke checks: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
