import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from himalaya_support.main import app  # noqa: E402

handler = app
