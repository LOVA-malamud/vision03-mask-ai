from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get('VISION_MODEL_DIR', str(ROOT / 'models')))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
