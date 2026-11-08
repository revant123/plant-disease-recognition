"""Leakage audit for the New Plant Diseases (augmented PlantVillage) dataset.

Every augmented file keeps the source PlantVillage file name and adds a suffix
such as _flipLR, _90deg or _new30degFlipLR. Stripping the suffix recovers the
source leaf ID, so we can check whether the same physical leaf appears in both
the training and the validation split.

Usage: python leakage_audit.py <dataset_root_or_zip> [--out report.json]
The dataset root must contain train/ and valid/ (or val/) class folders.
"""
import argparse, collections, json, os, re, zipfile

AUG = re.compile(r"_(flipLR|flipTB|90deg|180deg|270deg|new30degFlipLR|newGRR)$", re.I)


def source_id(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = AUG.sub("", stem)
    return stem.lower().strip()


def list_files(root):
    if root.endswith(".zip"):
        return [n for n in zipfile.ZipFile(root).namelist() if not n.endswith("/")]
    out = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            out.append(os.path.relpath(os.path.join(dp, f), root))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="leakage_report.json")
    a = ap.parse_args()
    files = list_files(a.root)
    split_of = lambda p: p.replace("\\", "/").split("/")[0]
    groups = {"train": collections.defaultdict(set), "val": collections.defaultdict(set)}
    n_files = collections.Counter()
    for p in files:
        s = split_of(p)
        s = "val" if s in ("valid", "val", "validation") else s
        if s not in groups:
            continue
        cls = p.replace("\\", "/").split("/")[-2]
        groups[s][cls].add(source_id(p))
        n_files[s] += 1
    per_class, tot_val_ids, tot_shared = {}, 0, 0
    val_files_leaky = 0
    shared_ids_all = set()
    for cls in sorted(groups["val"]):
        v, t = groups["val"][cls], groups["train"].get(cls, set())
        shared = v & t
        shared_ids_all |= {(cls, x) for x in shared}
        per_class[cls] = {"val_sources": len(v), "shared_with_train": len(shared),
                          "pct": round(100 * len(shared) / max(1, len(v)), 2)}
        tot_val_ids += len(v); tot_shared += len(shared)
    for p in files:
        s = split_of(p)
        if s in ("valid", "val", "validation"):
            cls = p.replace("\\", "/").split("/")[-2]
            if (cls, source_id(p)) in shared_ids_all:
                val_files_leaky += 1
    rep = {
        "train_files": n_files["train"], "val_files": n_files["val"],
        "train_unique_sources": sum(len(x) for x in groups["train"].values()),
        "val_unique_sources": tot_val_ids,
        "val_sources_also_in_train": tot_shared,
        "pct_val_sources_leaked": round(100 * tot_shared / max(1, tot_val_ids), 2),
        "val_files_whose_source_leaf_is_in_train": val_files_leaky,
        "pct_val_files_leaked": round(100 * val_files_leaky / max(1, n_files["val"]), 2),
        "per_class": per_class,
    }
    json.dump(rep, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
