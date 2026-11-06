"""Línea de comandos: `visionqc externos | catalogo | entrenar`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from visionqc import catalog as cat
from visionqc import roboflow as rf
from visionqc.split import assign_splits, check_split, summarize


def _load_train_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cmd_externos(args: argparse.Namespace) -> int:
    cfg = _load_train_config(args.config)
    data_root = Path(cfg["data_root"])
    priority = cfg["defect_labels"] + [lbl for lbl in cfg["labels"] if lbl not in cfg["defect_labels"]]
    for ds in rf.load_config(Path(args.datasets)):
        if args.solo and ds.name not in args.solo:
            continue
        raw_dir = data_root / "external" / "_descargas" / ds.name
        out_dir = data_root / "external" / ds.name
        print(f"\n== {ds.name} ({ds.license})")
        if not args.solo_adaptar:
            rf.download(ds, raw_dir)
        result = rf.adapt_yolo(raw_dir, out_dir, ds, priority=priority)
        print(f"   imágenes generadas por clase: {dict(result.counts)}")
        if result.skipped_small:
            print(f"   recortes descartados por pequeños: {result.skipped_small}")
        if result.unmapped:
            print(f"   AVISO clases sin mapear (agrégalas a class_map si sirven): {dict(result.unmapped)}")
    return 0


def cmd_catalogo(args: argparse.Namespace) -> int:
    cfg = _load_train_config(args.config)
    data_root, labels = Path(cfg["data_root"]), cfg["labels"]
    records = cat.scan_own_photos(data_root, labels) + cat.scan_external(data_root, labels)
    if not records:
        print(f"No hay imágenes en {data_root}/raw/propia ni en {data_root}/external.")
        return 1
    s = cfg["split"]
    assign_splits(records, s["val_frac"], s["test_frac"], s["seed"], cfg["defect_labels"])
    errors, warnings = check_split(records, labels)
    cat.write_catalog(records, Path(cfg["catalog"]))

    own = {r.panetone_id for r in records if r.source == cat.OWN_SOURCE}
    sources = sorted({r.source for r in records})
    print(f"Catálogo: {cfg['catalog']}  ·  {len(records)} imágenes  ·  {len(own)} panetones propios")
    print(f"Fuentes: {', '.join(sources)}\n")
    summary = summarize(records)
    print(f"{'split':<7}" + "".join(f"{lbl:>10}" for lbl in labels))
    for split, counts in summary.items():
        print(f"{split:<7}" + "".join(f"{counts[lbl]:>10}" for lbl in labels))
    for w in warnings:
        print(f"AVISO: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    return 1 if errors else 0


def cmd_entrenar(args: argparse.Namespace) -> int:
    try:
        from visionqc.train import train
    except ImportError as exc:
        print(f'Falta instalar el extra de entrenamiento: pip install -e ".[train]"  ({exc})')
        return 1
    cfg = _load_train_config(args.config)
    run_dir = train(cfg, epochs_override=args.epocas, pretrained=not args.sin_pesos)
    print(f"\nListo: {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # La consola de Windows usa cp1252: sin esto los acentos salen mal y algunas librerías fallan
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="visionqc", description="Herramientas de datos y entrenamiento")
    parser.add_argument("--config", default="configs/train.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("externos", help="Descarga y adapta datasets públicos de Roboflow")
    p.add_argument("--datasets", default="configs/external_datasets.yaml")
    p.add_argument("--solo", nargs="*", help="nombres de dataset a procesar (por defecto todos)")
    p.add_argument("--solo-adaptar", action="store_true", help="no descarga; solo re-adapta lo ya descargado")
    p.set_defaults(func=cmd_externos)

    p = sub.add_parser("catalogo", help="Genera data/catalog.csv con el split por panetón")
    p.set_defaults(func=cmd_catalogo)

    p = sub.add_parser("entrenar", help="Entrena en dos etapas y exporta a ONNX")
    p.add_argument("--epocas", type=int, help="fuerza N épocas por etapa (útil para pruebas rápidas)")
    p.add_argument("--sin-pesos", action="store_true", help="no descarga pesos de ImageNet (pruebas)")
    p.set_defaults(func=cmd_entrenar)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
