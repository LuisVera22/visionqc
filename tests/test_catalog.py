import json

import pytest
from conftest import make_image

from visionqc.catalog import read_catalog, scan_external, scan_own_photos, write_catalog


def test_scan_own_photos_reads_label_and_panetone(tmp_path, labels):
    make_image(tmp_path / "raw/propia/ok/p01/a.jpg")
    make_image(tmp_path / "raw/propia/moho/P01/b.png")
    make_image(tmp_path / "raw/propia/moho/P02/sub/c.jpg")

    records = scan_own_photos(tmp_path, labels)

    assert [(r.label, r.panetone_id, r.image_path) for r in records] == [
        ("moho", "P01", "raw/propia/moho/P01/b.png"),
        ("moho", "P02", "raw/propia/moho/P02/sub/c.jpg"),
        ("ok", "P01", "raw/propia/ok/p01/a.jpg"),
    ]
    assert all(r.source == "propia" and r.captured_at for r in records)


def test_scan_own_photos_rejects_loose_photos_and_unknown_labels(tmp_path, labels):
    make_image(tmp_path / "raw/propia/ok/suelta.jpg")
    make_image(tmp_path / "raw/propia/quemado/P01/a.jpg")

    with pytest.raises(ValueError) as exc:
        scan_own_photos(tmp_path, labels)
    assert "sueltas" in str(exc.value)
    assert "quemado" in str(exc.value)


def test_scan_external_uses_source_json(tmp_path, labels):
    src = tmp_path / "external/roboflow-x"
    make_image(src / "moho/a.jpg")
    make_image(src / "otra_clase/b.jpg")
    (src / "source.json").write_text(json.dumps({"name": "roboflow-x", "license": "CC BY 4.0"}))
    make_image(tmp_path / "external/_descargas/roboflow-x/train/images/c.jpg")  # se ignora

    records = scan_external(tmp_path, labels)

    assert len(records) == 1
    assert (records[0].source, records[0].license, records[0].label) == ("roboflow-x", "CC BY 4.0", "moho")


def test_catalog_roundtrip(tmp_path, labels):
    make_image(tmp_path / "raw/propia/ok/P01/a.jpg")
    records = scan_own_photos(tmp_path, labels)
    write_catalog(records, tmp_path / "catalog.csv")
    assert read_catalog(tmp_path / "catalog.csv") == records
