from pathlib import Path

import pytest
from PIL import Image

LABELS = ["ok", "moho", "deforme"]


def make_image(path: Path, size=(64, 48), color=(200, 150, 90)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def labels():
    return LABELS
