"""Descarga datasets de Roboflow Universe y los adapta a external/<fuente>/<clase>/*.jpg.

Los datasets públicos de pan con moho son de *detección* (cajas o polígonos en formato YOLO).
Para el clasificador se adaptan de dos formas:
  - crop: cada caja etiquetada se recorta como una imagen de su clase.
  - full: la imagen completa toma la clase de mayor prioridad presente (defectos antes que ok).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

from visionqc.catalog import IMAGE_EXTS


@dataclass
class ExternalDataset:
    name: str
    workspace: str
    project: str
    version: int
    license: str
    class_map: dict[str, str]
    mode: str = "crop"
    url: str = ""


@dataclass
class AdaptResult:
    counts: Counter = field(default_factory=Counter)
    unmapped: Counter = field(default_factory=Counter)
    skipped_small: int = 0


def load_config(path: Path) -> list[ExternalDataset]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [ExternalDataset(**d) for d in cfg.get("datasets", [])]


def download(ds: ExternalDataset, dest: Path) -> Path:
    """Descarga el dataset en formato YOLO. Requiere la variable ROBOFLOW_API_KEY."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Define ROBOFLOW_API_KEY (ver .env.example) para descargar de Roboflow.")
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError('Instala el extra: pip install -e ".[roboflow]"') from exc
    dest.mkdir(parents=True, exist_ok=True)
    project = Roboflow(api_key=api_key).workspace(ds.workspace).project(ds.project)
    project.version(ds.version).download("yolov8", location=str(dest), overwrite=False)
    return dest


def _class_names(raw_dir: Path) -> list[str]:
    data_yaml = raw_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"No se encontró {data_yaml}: ¿la descarga terminó bien?")
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):  # formato {0: 'bread', 1: 'mold'}
        names = [names[k] for k in sorted(names, key=int)]
    return list(names)


def _parse_label_line(line: str) -> tuple[int, tuple[float, float, float, float]] | None:
    """Devuelve (clase, caja normalizada x0,y0,x1,y1). Acepta cajas YOLO o polígonos."""
    parts = line.split()
    if len(parts) < 5:
        return None
    cls, coords = int(float(parts[0])), [float(x) for x in parts[1:]]
    if len(coords) == 4:  # cx cy w h
        cx, cy, w, h = coords
        return cls, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    xs, ys = coords[0::2], coords[1::2]  # polígono de segmentación
    return cls, (min(xs), min(ys), max(xs), max(ys))


def _iter_samples(raw_dir: Path):
    """Pares (imagen, archivo de etiquetas) en train/valid/test del export YOLO."""
    for img in sorted(raw_dir.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS or img.parent.name != "images":
            continue
        yield img, img.parent.parent / "labels" / f"{img.stem}.txt"


def adapt_yolo(
    raw_dir: Path,
    out_dir: Path,
    ds: ExternalDataset,
    priority: list[str],
    pad: float = 0.1,
    min_size: int = 32,
) -> AdaptResult:
    """Convierte un export YOLO en carpetas por clase dentro de out_dir."""
    names = _class_names(raw_dir)
    result = AdaptResult()
    rank = {label: i for i, label in enumerate(priority)}

    for img_path, label_path in _iter_samples(raw_dir):
        lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
        boxes = []
        for line in lines:
            parsed = _parse_label_line(line)
            if parsed is None:
                continue
            cls, box = parsed
            name = names[cls] if cls < len(names) else str(cls)
            target = ds.class_map.get(name)
            if target is None:
                result.unmapped[name] += 1
                continue
            boxes.append((target, box))
        if not boxes:
            continue

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            if ds.mode == "full":
                label = min((t for t, _ in boxes), key=lambda t: rank.get(t, len(rank)))
                _save(im, out_dir / label / f"{img_path.stem}.jpg")
                result.counts[label] += 1
                continue
            width, height = im.size
            for k, (label, (x0, y0, x1, y1)) in enumerate(boxes):
                bw, bh = (x1 - x0) * width, (y1 - y0) * height
                left = max(0, int(x0 * width - pad * bw))
                top = max(0, int(y0 * height - pad * bh))
                right = min(width, int(x1 * width + pad * bw))
                bottom = min(height, int(y1 * height + pad * bh))
                if right - left < min_size or bottom - top < min_size:
                    result.skipped_small += 1
                    continue
                _save(im.crop((left, top, right, bottom)), out_dir / label / f"{img_path.stem}_{k}.jpg")
                result.counts[label] += 1

    meta = {"name": ds.name, "license": ds.license, "url": ds.url, "mode": ds.mode}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _save(im: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=92)
