"""FastAPI: one startup load, shared decoder, real inference, no stored uploads."""
from contextlib import asynccontextmanager
import logging
import threading
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from .config import ROOT, MODELS_DIR, MAX_UPLOAD_BYTES
from .preprocessing import InvalidImage, preprocess
from ml.artifacts import load_bundle

class Predictor:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.lock = threading.Lock()

    def load(self):
        model, metadata = load_bundle(MODELS_DIR)
        probe = np.asarray(model(np.zeros((1, 64, 64, 3), np.float32), training=False))
        if probe.shape != (1, 1) or not np.isfinite(probe).all():
            raise ValueError('Model startup inference failed.')
        self.model, self.metadata = model, metadata

    def predict(self, data):
        array = preprocess(data)
        with self.lock:
            p = float(self.model(array, training=False).numpy()[0, 0])
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Model returned an invalid probability.')
        return {'prediction': 'without_mask' if p >= 0.5 else 'with_mask',
                'probability_with_mask': 1 - p, 'probability_without_mask': p,
                'model_version': self.metadata['version']}

@asynccontextmanager
async def lifespan(app):
    app.state.predictor = Predictor()
    try:
        await run_in_threadpool(app.state.predictor.load)
    except (OSError, ValueError, KeyError):
        logging.exception('Model unavailable: install a matching model and metadata, then restart.')
    yield

app = FastAPI(title='Project 03 · VISION', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5174', 'http://127.0.0.1:5174'],
                   allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

class UploadLimit:
    """Bound multipart body before the parser can spool an oversized upload."""
    def __init__(self, app): self.app = app
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST':
            return await self.app(scope, receive, send)
        total = 0
        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get('body', b''))
            if total > MAX_UPLOAD_BYTES + 65536:
                raise HTTPException(413, 'Image exceeds 10 MB.')
            return message
        await self.app(scope, bounded_receive, send)

app.add_middleware(UploadLimit)

@app.get('/api/health')
def health():
    predictor = app.state.predictor
    fields = ('version', 'test_accuracy', 'test_count', 'train_count', 'validation_count',
              'validation_accuracy', 'best_epoch')
    return {'model_ready': predictor.model is not None,
            'model': {k: predictor.metadata[k] for k in fields} if predictor.metadata else None}

@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    try:
        if app.state.predictor.model is None:
            raise HTTPException(503, 'No trained model is loaded. Install artifacts and restart.')
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, 'Image exceeds 10 MB.')
        return await run_in_threadpool(app.state.predictor.predict, data)
    except InvalidImage as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        await file.close()

if (ROOT / 'frontend/dist').is_dir():
    app.mount('/', StaticFiles(directory=ROOT / 'frontend/dist', html=True), name='website')
