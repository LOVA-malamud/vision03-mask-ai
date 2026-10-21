import io
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.preprocessing import preprocess, InvalidImage
from backend.app.main import app
from backend.app.config import ROOT
from ml.artifacts import load_bundle


def png(mode='RGB', size=(100, 80)):
    output=io.BytesIO();Image.new(mode,size,0).save(output,format='PNG');return output.getvalue()

@pytest.mark.parametrize('mode',['RGB','L','RGBA'])
def test_decoder_contract(mode):
    x=preprocess(png(mode));assert x.shape==(1,64,64,3);assert x.dtype==np.float32
    assert 0 <= x.min() <= x.max() <= 1

@pytest.mark.parametrize('data',[b'',b'fake.jpg',b'<script>not an image</script>'])
def test_invalid_files(data):
    with pytest.raises(InvalidImage):preprocess(data)

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as client:yield client

def test_real_model_upload_parity_and_startup_load(client):
    assert client.get('/api/health').json()['model_ready']
    model,meta=load_bundle(ROOT/'models')
    instance=app.state.predictor.model
    examples=list((ROOT/'tests/fixtures').glob('*.jpg'))
    assert len(examples)==2, 'Both held-out class fixtures are required.'
    for path in examples:
        data=path.read_bytes()
        expected=float(model(preprocess(data),training=False).numpy()[0,0])
        response=client.post('/api/predict',files={'file':(path.name,data,'image/jpeg')})
        assert response.status_code==200
        result=response.json()
        assert result['probability_without_mask']==pytest.approx(expected,abs=1e-6)
        assert result['prediction']==('without_mask' if expected>=.5 else 'with_mask')
        assert result['probability_with_mask']+result['probability_without_mask']==pytest.approx(1)
        assert result['model_version']==meta['version']
        assert app.state.predictor.model is instance


def test_http_rejects_invalid_images(client):
    assert client.post('/api/predict',files={'file':('fake.jpg',b'bad','image/jpeg')}).status_code==422
    assert client.post('/api/predict',files={'file':('big.jpg',b'x'*(10*1024*1024+1),'image/jpeg')}).status_code==413
    assert client.post('/api/predict').status_code==422


def test_saved_metadata_is_required(tmp_path):
    meta=json.loads((ROOT/'models/metadata.json').read_text())
    meta['class_mapping']={'without_mask':0,'with_mask':1}
    (tmp_path/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='mapping'):load_bundle(tmp_path)


def test_rgb_normalization_and_exif_orientation():
    from backend.app.preprocessing import decode_image
    buffer=io.BytesIO();Image.new('RGB',(7,13),(255,128,0)).save(buffer,format='PNG')
    np.testing.assert_allclose(preprocess(buffer.getvalue())[0,0,0],[1,128/255,0],atol=1e-7)
    buffer=io.BytesIO();im=Image.new('RGB',(7,13));exif=Image.Exif();exif[274]=6
    im.save(buffer,format='JPEG',exif=exif)
    assert decode_image(buffer.getvalue()).size==(13,7)


def test_http_decoder_matches_training_pixels():
    import hashlib
    manifest=json.loads((ROOT/'reports/dataset_manifest.json').read_text())
    sources=json.loads((ROOT/'tests/fixtures/sources.json').read_text())
    rows={r['path']:r for r in manifest['records']}
    for filename,source in sources.items():
        x=np.rint(preprocess((ROOT/'tests/fixtures'/filename).read_bytes())[0]*255).astype(np.uint8)
        assert hashlib.sha256(x.tobytes()).hexdigest()==rows[source]['sha256_pixels']
        assert rows[source]['split']=='test'


def test_mpo_primary_frame_and_http(client):
    from backend.app.preprocessing import decode_image
    primary=Image.new('RGB',(19,31),(235,30,10))
    auxiliary=Image.new('RGB',(19,31),(0,255,0))
    output=io.BytesIO()
    primary.save(output,format='MPO',save_all=True,append_images=[auxiliary])
    data=output.getvalue()
    with Image.open(io.BytesIO(data)) as im:
        assert im.format=='MPO' and im.n_frames==2
        im.seek(0)
        expected=im.convert('RGB')
    np.testing.assert_array_equal(np.asarray(decode_image(data)),np.asarray(expected))
    response=client.post('/api/predict',files={'file':('phone.jpeg',data,'image/jpeg')})
    assert response.status_code==200
    direct=app.state.predictor.predict(data)
    assert response.json()==direct


def test_animated_png_still_rejected():
    output=io.BytesIO()
    Image.new('RGB',(10,10),'red').save(output,format='PNG',save_all=True,
        append_images=[Image.new('RGB',(10,10),'green')],duration=100,loop=0)
    with pytest.raises(InvalidImage,match='Animated'):
        preprocess(output.getvalue())


def test_notebook_decoder_and_saved_loader_match_backend():
    import ast
    nb=json.loads((ROOT/'notebooks/Project03_VISION_HE.ipynb').read_text())
    for cell in nb['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
        assert all(o['output_type']!='error' for o in cell.get('outputs',[]))
    namespace={}
    for marker in ['def decode_image','def load_bundle']:
        source=next(''.join(c['source']) for c in nb['cells']
                    if c['cell_type']=='code' and marker in ''.join(c['source']))
        exec(source,namespace)
    model,meta=namespace['load_bundle'](ROOT/'models')
    for path in (ROOT/'tests/fixtures').glob('*.jpg'):
        data=path.read_bytes()
        np.testing.assert_array_equal(namespace['preprocess'](data),preprocess(data))
        p=float(model(namespace['preprocess'](data),training=False).numpy()[0,0])
        assert 0<=p<=1
    assert meta['version']==json.loads((ROOT/'models/metadata.json').read_text())['version']


def test_large_multipart_body_is_rejected(client):
    response=client.post('/api/predict',files={'file':('large.jpg',b'x'*(11*1024*1024),'image/jpeg')})
    assert response.status_code==413


def test_disguised_unsupported_image_is_rejected(client):
    output=io.BytesIO();Image.new('RGB',(10,10)).save(output,format='TIFF')
    response=client.post('/api/predict',files={'file':('fake.jpg',output.getvalue(),'image/jpeg')})
    assert response.status_code==422
