"""ECG model runtime: preprocessing + architecture. Keep byte-identical wherever the
model runs (training in the MCP workspace, the validation study, the serving
container); fingerprint() is recorded in the model card and the app's lock file."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

HERE = Path(__file__).parent


def fingerprint(folder: Optional[Path] = None) -> str:
    """SHA-256 over preprocess.py and model.py (name + bytes, line endings normalised)."""
    folder = Path(folder or HERE)
    h = hashlib.sha256()
    for name in ("model.py", "preprocess.py"):
        h.update(name.encode() + b"\0" + (folder / name).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()
