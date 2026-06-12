"""Start the Streamlit app with the current Python interpreter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "src" / "app" / "streamlit_app.py")],
        check=True,
        cwd=ROOT,
    )
