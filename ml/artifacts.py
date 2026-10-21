"""Load the saved model only when its labels, preprocessing and checksum agree."""
import hashlib
import json
from pathlib import Path
from ml import CLASS_MAPPING, PREPROCESSING

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def load_bundle(directory):
    import tensorflow as tf
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text())
    if meta.get('class_mapping') != CLASS_MAPPING or meta.get('preprocessing') != PREPROCESSING:
        raise ValueError('Class mapping or preprocessing does not match this application.')
    if not meta.get('trained') or meta.get('sha256') != sha256(directory / 'mask_cnn.keras'):
        raise ValueError('The trained model is missing or its checksum is incorrect.')
    model = tf.keras.models.load_model(directory / 'mask_cnn.keras', compile=False, safe_mode=True)
    if model.input_shape != (None, 64, 64, 3) or model.output_shape != (None, 1):
        raise ValueError('Model input/output shape is incompatible.')
    if model.layers[-1].activation.__name__ != 'sigmoid':
        raise ValueError('Expected sigmoid output.')
    return model, meta
