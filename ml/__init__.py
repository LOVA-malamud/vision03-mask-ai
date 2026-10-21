"""Shared training/inference contract. No normalization inside the CNN."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
# NumPy 1.26 on Python 3.12 / Intel macOS needs setuptools' distutils shim.
import setuptools  # noqa: F401

IMAGE_SIZE = (64, 64)
CLASS_MAPPING = {'with_mask': 0, 'without_mask': 1}
PREPROCESSING = {
    'image_size': [64, 64], 'channels': 'RGB', 'exif_transpose': True,
    'resize': 'Pillow.BILINEAR', 'dtype': 'float32', 'normalization': 'divide_by_255',
    'input_shape': [None, 64, 64, 3], 'sigmoid_positive_class': 'without_mask',
    'threshold': 0.5,
}
