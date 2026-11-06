from collections import defaultdict

from visionqc.catalog import Record
from visionqc.split import assign_splits, check_split

DEFECTS = ["moho", "deforme"]


def own(pid, label, n=3):
    return [Record(f"raw/propia/{label}/{pid}/{i}.jpg", pid, label, "propia", "propia", "") for i in range(n)]


def external(label, n=5):
    return [
        Record(f"external/rf/{label}/{i}.jpg", "", label, "roboflow-x", "CC BY 4.0", "") for i in range(n)
    ]


def dataset():
    records = []
    for i in range(1, 7):
        records += own(f"P{i:02}", "ok")
    for i in range(7, 12):  # panetones de moho: fotos 'ok' al inicio y 'moho' después
        records += own(f"P{i:02}", "ok", n=2) + own(f"P{i:02}", "moho")
    for i in range(12, 16):
        records += own(f"P{i:02}", "deforme")
    return records + external("moho") + external("ok")


def test_each_panetone_stays_in_one_split():
    records = assign_splits(dataset(), 0.15, 0.15, seed=1, defect_labels=DEFECTS)
    splits = defaultdict(set)
    for r in records:
        if r.source == "propia":
            splits[r.panetone_id].add(r.split)
    assert all(len(s) == 1 for s in splits.values())


def test_external_always_in_train():
    records = assign_splits(dataset(), 0.3, 0.3, seed=1, defect_labels=DEFECTS)
    assert {r.split for r in records if r.source != "propia"} == {"train"}


def test_val_and_test_get_every_class_and_no_errors(labels):
    records = assign_splits(dataset(), 0.15, 0.15, seed=7, defect_labels=DEFECTS)
    errors, warnings = check_split(records, labels)
    assert errors == []
    assert warnings == []
    for split in ("val", "test"):
        assert {r.label for r in records if r.split == split} == {"ok", "moho", "deforme"}


def test_split_is_deterministic():
    a = [r.split for r in assign_splits(dataset(), 0.15, 0.15, seed=3, defect_labels=DEFECTS)]
    b = [r.split for r in assign_splits(dataset(), 0.15, 0.15, seed=3, defect_labels=DEFECTS)]
    assert a == b


def test_check_split_detects_leakage_and_few_panetones(labels):
    records = own("P01", "ok") + own("P02", "moho")
    for r in records:
        r.split = "train"
    records[0].split = "val"
    errors, warnings = check_split(records, labels)
    assert any("P01" in e for e in errors)
    assert any("deforme" in w for w in warnings)
