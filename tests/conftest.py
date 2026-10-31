import os
import tempfile
from pathlib import Path

cache = Path(tempfile.gettempdir()) / "cctv-privacy-test-cache"
cache.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(cache))
os.environ.setdefault("MPLCONFIGDIR", str(cache))
