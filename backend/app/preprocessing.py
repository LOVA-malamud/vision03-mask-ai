"""One authoritative decoder, shared by dataset training and HTTP inference."""
import io
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from ml import IMAGE_SIZE
from .config import MAX_UPLOAD_BYTES

MAX_PIXELS = 20_000_000
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}

class InvalidImage(ValueError):
    pass

def decode_image(data: bytes) -> Image.Image:
    if not data:
        raise InvalidImage("The image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImage("The image exceeds the 10 MB limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                if probe.format not in ALLOWED_FORMATS:
                    raise InvalidImage("Use a JPEG, PNG, or WebP image.")
                if probe.width * probe.height > MAX_PIXELS:
                    raise InvalidImage("Image dimensions exceed 20 megapixels.")
                # MPO is a still JPEG with auxiliary images; use its primary frame.
                if probe.format != "MPO" and getattr(probe, "n_frames", 1) > 1:
                    raise InvalidImage("Animated images are not supported.")
                probe.verify()
            with Image.open(io.BytesIO(data)) as source:
                source.seek(0)
                source.load()
                return ImageOps.exif_transpose(source).convert("RGB")
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImage("The image is damaged or cannot be decoded.") from exc

def image_array(image: Image.Image) -> np.ndarray:
    resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / np.float32(255.0)

def preprocess(data: bytes) -> np.ndarray:
    # (64, 64, 3) -> (1, 64, 64, 3); no second normalization in the CNN.
    return np.expand_dims(image_array(decode_image(data)), axis=0)
