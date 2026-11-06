import json

from conftest import make_image
from PIL import Image

from visionqc.roboflow import ExternalDataset, adapt_yolo

PRIORITY = ["moho", "deforme", "ok"]


def yolo_export(root):
    root.mkdir(parents=True)
    (root / "data.yaml").write_text("names: ['bread', 'mold_bread', 'plate']\nnc: 3\n")
    make_image(root / "train/images/a.jpg", size=(200, 100))
    (root / "train/labels").mkdir(parents=True)
    (root / "train/labels/a.txt").write_text(
        "0 0.25 0.5 0.4 0.8\n"  # caja de pan sano
        "1 0.1 0.1 0.4 0.1 0.4 0.6 0.1 0.6\n"  # polígono de moho
        "2 0.8 0.5 0.1 0.1\n"  # clase sin mapear
    )
    make_image(root / "valid/images/b.jpg", size=(200, 100))
    (root / "valid/labels").mkdir(parents=True)
    (root / "valid/labels/b.txt").write_text("1 0.5 0.5 0.01 0.01\n")  # caja diminuta


def ds(mode):
    return ExternalDataset(
        name="roboflow-test", workspace="w", project="p", version=1, license="CC BY 4.0",
        class_map={"bread": "ok", "mold_bread": "moho"}, mode=mode,
    )


def test_crop_mode(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    yolo_export(raw)

    result = adapt_yolo(raw, out, ds("crop"), PRIORITY)

    assert dict(result.counts) == {"ok": 1, "moho": 1}
    assert dict(result.unmapped) == {"plate": 1}
    assert result.skipped_small == 1
    with Image.open(out / "moho/a_1.jpg") as im:
        assert im.size == (72, 60)  # caja 60x50 px + 10 % de margen por lado
    assert json.loads((out / "source.json").read_text())["license"] == "CC BY 4.0"


def test_full_mode_uses_defect_priority(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    yolo_export(raw)

    result = adapt_yolo(raw, out, ds("full"), PRIORITY)

    assert dict(result.counts) == {"moho": 2}
    assert (out / "moho/a.jpg").exists() and not (out / "ok").exists()
