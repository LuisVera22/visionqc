"""Preprocesamiento ÚNICO para evaluación e inferencia.

Lo usan la evaluación del entrenamiento y, más adelante, la API en Lambda. Que ambos llamen a
esta misma función evita el "training/serving skew": que el modelo vea en producción imágenes
preparadas distinto de como se evaluó. Solo depende de numpy y Pillow (nada de torch), para que
la imagen Docker de la API sea liviana.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_RATIO = 1.14  # redimensiona el lado corto a size*1.14 y luego recorta el centro


def preprocess(image: Image.Image, size: int = 224) -> np.ndarray:
    """PIL -> array float32 (3, size, size) normalizado como ImageNet."""
    image = image.convert("RGB")
    short = round(size * RESIZE_RATIO)
    w, h = image.size
    scale = short / min(w, h)
    image = image.resize((max(short, round(w * scale)), max(short, round(h * scale))), Image.BILINEAR)
    w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    image = image.crop((left, top, left + size, top + size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return arr.transpose(2, 0, 1).copy()


def preprocess_batch(images: list[Image.Image], size: int = 224) -> np.ndarray:
    return np.stack([preprocess(im, size) for im in images])
