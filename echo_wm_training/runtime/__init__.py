"""Echo wall-motion model runtime: preprocessing + architecture + segment/territory model + study pipeline.
Keep byte-identical wherever the model runs (training in MCP, validation study, serving container);
fingerprint() is recorded in the model card and pinned in the app's lock file."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

HERE = Path(__file__).parent
FILES = ("model.py", "preprocess.py", "segments.py", "pipeline.py")


def fingerprint(folder: Optional[Path] = None) -> str:
    folder = Path(folder or HERE)
    h = hashlib.sha256()
    for name in FILES:
        h.update(name.encode() + b"\0" + (folder / name).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()
