"""Split train/val/test por panetón.

Reglas:
  1. Todas las fotos de un mismo panetón quedan en el mismo split (evita que el modelo
     "memorice" una unidad y las métricas salgan infladas).
  2. Solo las fotos propias se reparten; las fuentes externas van siempre a train.
     Así val y test miden el rendimiento sobre panetones reales.
  3. Se estratifica por tipo de defecto del panetón para que val/test tengan cada clase.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from visionqc.catalog import OWN_SOURCE, Record

SPLITS = ("train", "val", "test")


def _group_key(labels: set[str], defect_labels: list[str]) -> tuple[str, ...]:
    """Panetones con el mismo conjunto de defectos se reparten juntos."""
    return tuple(sorted(labels & set(defect_labels)))


def assign_splits(
    records: list[Record],
    val_frac: float,
    test_frac: float,
    seed: int,
    defect_labels: list[str],
) -> list[Record]:
    labels_by_id: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.source == OWN_SOURCE:
            labels_by_id[r.panetone_id].add(r.label)

    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for pid in sorted(labels_by_id):
        groups[_group_key(labels_by_id[pid], defect_labels)].append(pid)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for key in sorted(groups):
        ids = groups[key][:]
        rng.shuffle(ids)
        n = len(ids)
        n_test, n_val = round(n * test_frac), round(n * val_frac)
        if n >= 3:  # con 3 o más panetones garantizamos al menos uno en val y en test
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        for i, pid in enumerate(ids):
            assignment[pid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    for r in records:
        r.split = assignment[r.panetone_id] if r.source == OWN_SOURCE else "train"
    return records


def summarize(records: list[Record]) -> dict[str, Counter]:
    summary: dict[str, Counter] = {s: Counter() for s in SPLITS}
    for r in records:
        summary[r.split][r.label] += 1
    return summary


def check_split(records: list[Record], labels: list[str]) -> tuple[list[str], list[str]]:
    """Devuelve (errores, avisos). Un error significa que el split no es confiable."""
    errors: list[str] = []
    warnings: list[str] = []

    bad_external = [r.image_path for r in records if r.source != OWN_SOURCE and r.split != "train"]
    if bad_external:
        errors.append(f"{len(bad_external)} imagen(es) externas fuera de train (p. ej. {bad_external[0]})")

    splits_by_id: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.source == OWN_SOURCE:
            splits_by_id[r.panetone_id].add(r.split)
    leaked = sorted(pid for pid, s in splits_by_id.items() if len(s) > 1)
    if leaked:
        errors.append(f"panetones repartidos en más de un split: {', '.join(leaked)}")

    own_ids = defaultdict(set)
    for r in records:
        if r.source == OWN_SOURCE:
            own_ids[r.label].add(r.panetone_id)
    for label in labels:
        if len(own_ids[label]) < 3:
            warnings.append(
                f"clase '{label}': solo {len(own_ids[label])} panetón(es) propio(s); "
                "con menos de 3 no se puede validar ni probar esa clase"
            )

    summary = summarize(records)
    for split in ("val", "test"):
        missing = [lbl for lbl in labels if summary[split][lbl] == 0]
        if missing:
            warnings.append(f"{split} no tiene fotos de: {', '.join(missing)}")
    return errors, warnings
