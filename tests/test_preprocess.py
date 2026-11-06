import numpy as np
from PIL import Image

from visionqc.preprocess import preprocess, preprocess_batch


def test_preprocess_shape_dtype_and_normalization():
    arr = preprocess(Image.new("RGB", (640, 480), (124, 116, 104)), size=224)
    assert arr.shape == (3, 224, 224)
    assert arr.dtype == np.float32
    assert np.abs(arr).max() < 0.05  # el color es ~la media de ImageNet -> valores cercanos a 0


def test_preprocess_handles_portrait_grayscale_and_batches():
    images = [Image.new("L", (300, 900)), Image.new("RGBA", (500, 500))]
    assert preprocess_batch(images, size=128).shape == (2, 3, 128, 128)
