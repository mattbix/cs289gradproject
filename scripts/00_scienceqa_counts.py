from datasets import load_dataset

DATASETS = {
    "IMG": "lmms-lab/ScienceQA-IMG",
    "ALL": "derek-thomas/ScienceQA",
}

def count_split(path, split):
    try:
        ds = load_dataset(path, split=split)
    except Exception as e:
        print(f"  {path} / {split}: LOAD FAILED ({type(e).__name__}: {e})")
        return None
    n = len(ds)
    cols = ds.column_names
    img_col = next((c for c in cols if c in ("image", "image_1")), None)
    if img_col is None:
        print(f"  {path} / {split}: no image column (cols={cols})")
        return {"total": n, "img": 0, "txt": n, "cols": cols}
    n_img = sum(1 for ex in ds if ex[img_col] is not None)
    return {"total": n, "img": n_img, "txt": n - n_img, "cols": cols}

for label, path in DATASETS.items():
    print(f"\n=== {label}: {path} ===")
    for split in ["train", "validation", "test"]:
        r = count_split(path, split)
        if r:
            print(f"  {split:10s}  total={r['total']:6d}  image={r['img']:6d}  text_only={r['txt']:6d}  cols={r['cols']}")
