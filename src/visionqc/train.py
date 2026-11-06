"""Entrenamiento en dos etapas (externo -> propio) con transfer learning y export a ONNX.

Salida en runs/<fecha-hora>/:
    model.pt       pesos del mejor modelo (según `train.monitor` sobre val)
    model.onnx     el mismo modelo para inferencia con ONNX Runtime (lo usará la Lambda)
    labels.json    clases y parámetros de preprocesamiento
    metrics.json   historial por época, métricas de test y verificación ONNX
    config.yaml    copia de la configuración usada
"""

from __future__ import annotations

import fnmatch
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from visionqc.catalog import Record, read_catalog
from visionqc.preprocess import IMAGENET_MEAN, IMAGENET_STD, RESIZE_RATIO, preprocess

ARCHS = {
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "efficientnet_b0": models.efficientnet_b0,
}


class CatalogDataset(Dataset):
    def __init__(self, records: list[Record], data_root: Path, labels: list[str], transform):
        self.records, self.data_root, self.transform = records, data_root, transform
        self.index = {label: i for i, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        with Image.open(self.data_root / r.image_path) as im:
            x = self.transform(im.convert("RGB"))
        return x, self.index[r.label]


def train_transform(size: int):
    """Aumentado de datos: solo en entrenamiento."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int):
    """El mismo preprocesamiento que usará la API (visionqc.preprocess)."""
    return lambda im: torch.from_numpy(preprocess(im, size))


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"arquitectura '{arch}' no soportada: {', '.join(ARCHS)}")
    model = ARCHS[arch](weights="DEFAULT" if pretrained else None)
    head = model.classifier[-1]
    model.classifier[-1] = nn.Linear(head.in_features, num_classes)
    # MobileNetV3/EfficientNet traen BatchNorm con momentum 0.01: con datasets chicos las estadísticas
    # casi no se actualizan y el modelo en modo eval() queda descalibrado. 0.1 es el estándar de PyTorch.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1
    return model


def compute_metrics(y_true: list[int], y_pred: list[int], labels: list[str]) -> dict:
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    per_class, f1s = {}, []
    for i, label in enumerate(labels):
        tp, support, predicted = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
        if support == 0:
            per_class[label] = {"precision": None, "recall": None, "f1": None, "support": 0}
            continue
        recall = tp / support
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1s.append(f1)
    total = int(cm.sum())
    return {
        "accuracy": float(np.trace(cm) / total) if total else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def monitor_value(metrics: dict, monitor: str) -> float:
    if monitor == "macro_f1":
        return metrics["macro_f1"] or 0.0
    if monitor.startswith("recall:"):
        return metrics["per_class"][monitor.split(":", 1)[1]]["recall"] or 0.0
    raise ValueError(f"monitor desconocido: {monitor}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, labels: list[str]) -> dict:
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        y_pred += model(x.to(device)).argmax(1).cpu().tolist()
        y_true += y.tolist()
    return compute_metrics(y_true, y_pred, labels)


def _fmt(m: dict, labels: list[str]) -> str:
    parts = [f"macro-F1 {m['macro_f1']:.3f}" if m["macro_f1"] is not None else "macro-F1 -"]
    for label in labels:
        r = m["per_class"][label]["recall"]
        if r is not None:
            parts.append(f"recall {label} {r:.2f}")
    return " · ".join(parts)


def export_onnx(model: nn.Module, size: int, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(2, 3, size, size)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=256)},),  # tamaño de lote variable
        external_data=False,  # un solo archivo .onnx, más simple de empaquetar en la Lambda
        verbose=False,
    )


def verify_onnx(model: nn.Module, onnx_path: Path, sample: np.ndarray) -> float | None:
    """Compara la salida de PyTorch y de ONNX Runtime sobre las mismas imágenes."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"image": sample})[0]
    with torch.no_grad():
        torch_out = model.eval()(torch.from_numpy(sample)).numpy()
    return float(np.abs(onnx_out - torch_out).max())


def _loader(records, data_root, labels, transform, tc, shuffle):
    return DataLoader(
        CatalogDataset(records, data_root, labels, transform),
        batch_size=tc["batch_size"],
        shuffle=shuffle,
        num_workers=tc["num_workers"],
    )


def train(cfg: dict, epochs_override: int | None = None, pretrained: bool = True) -> Path:
    labels, data_root = cfg["labels"], Path(cfg["data_root"])
    size, tc, monitor = cfg["model"]["image_size"], cfg["train"], cfg["train"]["monitor"]
    records = read_catalog(Path(cfg["catalog"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["split"]["seed"])

    run_dir = Path(cfg["output_dir"]) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), "utf-8")

    use_weights = pretrained and cfg["model"]["pretrained"]
    model = build_model(cfg["model"]["arch"], len(labels), use_weights).to(device)
    val = [r for r in records if r.split == "val"]
    test = [r for r in records if r.split == "test"]
    val_loader = _loader(val, data_root, labels, eval_transform(size), tc, shuffle=False)
    best_path, best_score, history = run_dir / "model.pt", -1.0, []

    for stage in cfg["stages"]:
        stage_train = [
            r for r in records
            if r.split == "train" and any(fnmatch.fnmatch(r.source, p) for p in stage["sources"])
        ]
        if not stage_train:
            print(f"== Etapa '{stage['name']}': sin imágenes para {stage['sources']}, se salta")
            continue
        counts = np.bincount([labels.index(r.label) for r in stage_train], minlength=len(labels))
        # pesos por clase para compensar el desbalance (normalmente hay muchos más 'ok')
        weights = torch.tensor(
            [len(stage_train) / (len(labels) * c) if c else 0.0 for c in counts], dtype=torch.float32
        )
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=stage["lr"], weight_decay=tc["weight_decay"])
        epochs = epochs_override or stage["epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loader = _loader(stage_train, data_root, labels, train_transform(size), tc, shuffle=True)
        print(f"== Etapa '{stage['name']}': {len(stage_train)} imágenes "
              f"{dict(zip(labels, counts.tolist(), strict=True))} · {epochs} épocas · {device}")

        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum, seen = 0.0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(y)
                seen += len(y)
            scheduler.step()
            entry = {"stage": stage["name"], "epoch": epoch, "train_loss": loss_sum / seen}
            line = f"   época {epoch:>2}/{epochs} · loss {entry['train_loss']:.4f}"
            if val:
                m = evaluate(model, val_loader, device, labels)
                entry["val"], score = m, monitor_value(m, monitor)
                line += f" · val {_fmt(m, labels)}"
                if score >= best_score:  # en empate gana la época más reciente (más entrenada)
                    best_score, entry["best"] = score, True
                    torch.save(model.state_dict(), best_path)
                    line += "  *"
            history.append(entry)
            print(line)

    if not history:
        raise RuntimeError("No hay imágenes de entrenamiento: corre `visionqc catalogo` primero.")
    if not best_path.exists():  # sin val: se guarda el último modelo
        torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=device))

    results: dict = {"monitor": monitor, "best_val_score": best_score if val else None, "history": history}
    if test:
        test_loader = _loader(test, data_root, labels, eval_transform(size), tc, shuffle=False)
        results["test"] = evaluate(model, test_loader, device, labels)
        print(f"\nTEST ({len(test)} imágenes): {_fmt(results['test'], labels)}")
    else:
        print("\nSin split de test: aún no hay suficientes panetones propios.")

    model = model.cpu()
    export_onnx(model, size, run_dir / "model.onnx")
    sample_records = (val or test or records)[:4]
    sample = np.stack([
        preprocess(Image.open(data_root / r.image_path), size) for r in sample_records
    ]).astype(np.float32)
    results["onnx_max_abs_diff"] = verify_onnx(model, run_dir / "model.onnx", sample)
    if results["onnx_max_abs_diff"] is not None:
        print(f"ONNX verificado: diferencia máxima vs PyTorch = {results['onnx_max_abs_diff']:.2e}")

    meta = {
        "labels": labels,
        "arch": cfg["model"]["arch"],
        "image_size": size,
        "resize_ratio": RESIZE_RATIO,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }
    (run_dir / "labels.json").write_text(json.dumps(meta, indent=2), "utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), "utf-8")
    return run_dir
