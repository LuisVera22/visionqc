"""Catálogo de imágenes: una fila por imagen con su panetón, clase, fuente, licencia y split.

Estructura esperada en disco (relativa a data_root):

    raw/propia/<clase>/<panetone_id>/*.jpg     fotos propias (p. ej. raw/propia/moho/P07/IMG_001.jpg)
    external/<fuente>/<clase>/*.jpg            datasets públicos ya adaptados (ver roboflow.py)
    external/<fuente>/source.json              licencia y URL de la fuente

Un mismo panetón puede aparecer en varias clases (p. ej. P07 en ok/ los primeros días y en
moho/ después): el split por panetón mantiene todas sus fotos juntas.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OWN_SOURCE = "propia"
OWN_DIR = Path("raw") / OWN_SOURCE
EXTERNAL_DIR = Path("external")

_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 36867
_DATETIME = 306


@dataclass
class Record:
    image_path: str  # relativa a data_root, con "/"
    panetone_id: str  # vacío en fuentes externas
    label: str
    source: str  # "propia" o el nombre de la fuente externa
    license: str
    captured_at: str  # ISO 8601
    split: str = ""  # train | val | test


FIELDS = [f.name for f in fields(Record)]


def _captured_at(path: Path) -> str:
    """Fecha de captura desde EXIF; si no hay, la fecha de modificación del archivo."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            raw = exif.get_ifd(_EXIF_IFD).get(_DATETIME_ORIGINAL) or exif.get(_DATETIME)
        if raw:
            return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S").isoformat()
    except (OSError, ValueError):
        pass
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def scan_own_photos(data_root: Path, labels: list[str]) -> list[Record]:
    """Lee raw/propia/<clase>/<panetone_id>/. Falla si algo no sigue la estructura."""
    own_dir = data_root / OWN_DIR
    if not own_dir.exists():
        return []
    records: list[Record] = []
    problems: list[str] = []
    for label_dir in sorted(p for p in own_dir.iterdir() if p.is_dir()):
        if label_dir.name not in labels:
            problems.append(f"clase desconocida '{label_dir.name}' (válidas: {', '.join(labels)})")
            continue
        loose = [p.name for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if loose:
            problems.append(
                f"{len(loose)} foto(s) sueltas en {label_dir.relative_to(data_root).as_posix()}/: "
                "deben ir dentro de una carpeta con el ID del panetón (p. ej. P07/)"
            )
        for pid_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
            for img in _images(pid_dir):
                records.append(
                    Record(
                        image_path=img.relative_to(data_root).as_posix(),
                        panetone_id=pid_dir.name.upper(),
                        label=label_dir.name,
                        source=OWN_SOURCE,
                        license=OWN_SOURCE,
                        captured_at=_captured_at(img),
                    )
                )
    if problems:
        raise ValueError("Estructura de fotos propias inválida:\n  - " + "\n  - ".join(problems))
    return records


def scan_external(data_root: Path, labels: list[str]) -> list[Record]:
    """Lee external/<fuente>/<clase>/*.jpg (generado por `visionqc externos`)."""
    ext_dir = data_root / EXTERNAL_DIR
    if not ext_dir.exists():
        return []
    records: list[Record] = []
    for src_dir in sorted(p for p in ext_dir.iterdir() if p.is_dir()):
        meta_file = src_dir / "source.json"
        if not meta_file.exists():  # carpeta de descarga cruda u otra cosa: se ignora
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        for label_dir in sorted(p for p in src_dir.iterdir() if p.is_dir() and p.name in labels):
            for img in _images(label_dir):
                records.append(
                    Record(
                        image_path=img.relative_to(data_root).as_posix(),
                        panetone_id="",
                        label=label_dir.name,
                        source=meta["name"],
                        license=meta.get("license", ""),
                        captured_at="",
                    )
                )
    return records


def write_catalog(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)


def read_catalog(path: Path) -> list[Record]:
    with path.open(newline="", encoding="utf-8") as f:
        return [Record(**row) for row in csv.DictReader(f)]
