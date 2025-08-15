import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
PREVIEWS_DIR = DATA_DIR / "previews"

for d in (DATA_DIR, UPLOADS_DIR, OUTPUTS_DIR, PREVIEWS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def saved_path(dirpath: Path, filename: str) -> Path:
    return dirpath / filename
