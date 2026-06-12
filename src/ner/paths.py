"""Path helpers for saved NER checkpoints."""

from __future__ import annotations

from pathlib import Path


def active_pointer_path(output_dir: str | Path) -> Path:
    """Return the file that stores the active checkpoint directory."""
    output_dir = Path(output_dir)
    return output_dir.parent / "ner_model_active.txt"


def resolve_ner_model_dir(output_dir: str | Path) -> Path:
    """Resolve the checkpoint used by evaluation and inference.

    Older versions saved directly to ``outputs/ner_model``. Newer training runs
    save to ``outputs/ner_model_runs/<run_id>`` to avoid Windows file locks while
    Streamlit is open. This helper keeps both layouts compatible.
    """
    output_dir = Path(output_dir)
    pointer = active_pointer_path(output_dir)
    if pointer.exists():
        pointed_text = pointer.read_text(encoding="utf-8-sig").strip()
        pointed = Path(pointed_text)
        if pointed.exists():
            return pointed
    return output_dir


def write_active_model_pointer(output_dir: str | Path, checkpoint_dir: str | Path) -> None:
    """Mark a checkpoint directory as active."""
    pointer = active_pointer_path(output_dir)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(Path(checkpoint_dir).resolve()), encoding="utf-8")
