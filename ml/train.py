"""Train only on training data, select using validation loss, test once after reload."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from ml import CLASS_MAPPING, PREPROCESSING
from ml.model import build_model
from ml.artifacts import load_bundle, sha256, write_json


def train(prepared, output, reports, epochs=20):
    prepared, output, reports = Path(prepared), Path(output), Path(reports)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing submission by accidentally rerunning training.
    if (output / 'mask_cnn.keras').exists():
        raise FileExistsError('Choose an empty output directory for a new training run.')
    data = np.load(prepared / 'images.npz')
    x_train, y_train = data['x_train'].astype(np.float32)/255, data['y_train']
    x_val, y_val = data['x_validation'].astype(np.float32)/255, data['y_validation']
    data.close()
    model = build_model()
    model.summary()
    best = output / 'mask_cnn.keras'
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best), monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-5),
        tf.keras.callbacks.CSVLogger(str(reports / 'history.csv')),
    ]
    def batches(x, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle: ds = ds.shuffle(len(y), seed=42)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        return ds.batch(32).with_options(options).prefetch(1)
    history = model.fit(batches(x_train, y_train, True), validation_data=batches(x_val, y_val),
                        epochs=epochs, callbacks=callbacks, verbose=2)
    write_json(reports / 'history.json', history.history)
    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    # Reload the actual saved artifact, rather than evaluate only the in-memory model.
    reloaded = tf.keras.models.load_model(best, compile=False, safe_mode=True)
    before = model(x_val[:8], training=False).numpy()
    after = reloaded(x_val[:8], training=False).numpy()
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
    with np.load(prepared / 'images.npz') as data:
        x_test, y_test = data['x_test'].astype(np.float32)/255, data['y_test']
    p = np.concatenate([reloaded(x_test[i:i+64], training=False).numpy().reshape(-1)
                        for i in range(0, len(x_test), 64)])
    predicted = (p >= 0.5).astype(int)
    metrics = {'accuracy': float(accuracy_score(y_test, predicted)),
               'loss': float(log_loss(y_test, p, labels=[0, 1])),
               'confusion_matrix': confusion_matrix(y_test, predicted, labels=[0,1]).tolist(),
               'classification_report': classification_report(y_test, predicted,
                    labels=[0,1], target_names=list(CLASS_MAPPING), output_dict=True, zero_division=0)}
    write_json(reports / 'evaluation.json', metrics)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    write_json(reports / 'dataset_manifest.json', manifest)
    meta = {
        'version': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
        'trained': True, 'sha256': sha256(best), 'class_mapping': CLASS_MAPPING,
        'preprocessing': PREPROCESSING, 'tensorflow_version': tf.__version__,
        'seed': 42, 'epochs_trained': len(history.history['loss']), 'best_epoch': best_epoch,
        'parameters': model.count_params(), 'train_count': len(y_train), 'validation_count': len(y_val),
        'test_count': len(y_test), 'validation_accuracy': history.history['val_accuracy'][best_epoch-1],
        'validation_loss': history.history['val_loss'][best_epoch-1],
        'test_accuracy': metrics['accuracy'], 'test_loss': metrics['loss'],
        'dataset': manifest['source'], 'dataset_archive_sha256': manifest['archive_sha256'],
        'reload_max_absolute_difference': float(np.max(np.abs(before-after))),
        'training_environment': 'Local CPU' if not tf.config.list_physical_devices('GPU') else 'GPU',
    }
    write_json(output / 'metadata.json', meta)
    verified, _ = load_bundle(output)
    np.testing.assert_allclose(verified(x_val[:8], training=False).numpy(), after, atol=1e-6)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, metric in zip(axes, ['accuracy','loss']):
        ax.plot(range(1,len(history.history[metric])+1),history.history[metric],label='Train')
        ax.plot(range(1,len(history.history[metric])+1),history.history['val_'+metric],label='Validation')
        ax.axvline(best_epoch,color='gray',linestyle='--',label='Saved epoch')
        ax.set(xlabel='Epoch',ylabel=metric);ax.legend()
    fig.tight_layout();fig.savefig(reports/'training.png',dpi=150);plt.close(fig)
    fig, ax = plt.subplots(figsize=(5,4))
    matrix=np.array(metrics['confusion_matrix']);ax.imshow(matrix,cmap='Greens')
    for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,str(value),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=list(CLASS_MAPPING),yticklabels=list(CLASS_MAPPING),
           xlabel='Predicted',ylabel='Actual',title='Held-out test set')
    fig.tight_layout();fig.savefig(reports/'confusion_matrix.png',dpi=150);plt.close(fig)
    print(json.dumps(meta,indent=2))
    return meta

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared',default='data/prepared')
    parser.add_argument('--output',default='models')
    parser.add_argument('--reports',default='reports')
    parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    train(args.prepared,args.output,args.reports,args.epochs)
