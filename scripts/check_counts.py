from collections import Counter
from datasets import load_dataset, Image

ALL, IMG = "derek-thomas/ScienceQA", "lmms-lab/ScienceQA-IMG"

def key(ex):
    return (ex["question"], tuple(ex["choices"]), ex["answer"], ex["hint"])

for split in ["train", "validation", "test"]:
    a = load_dataset(ALL, split=split).cast_column("image", Image(decode=False))
    b = load_dataset(IMG, split=split).cast_column("image", Image(decode=False))
    a_img = [key(ex) for ex in a if ex["image"] is not None]
    b_all = [key(ex) for ex in b]
    print(split, "total", len(a), "image", len(a_img), "text", len(a) - len(a_img),
          "| IMG total", len(b_all))
    print("  same order:", a_img == b_all, "| same rows:", Counter(a_img) == Counter(b_all))
    print("  IMG columns:", b.column_names)
    print("  hint nonempty:", sum(bool(ex["hint"]) for ex in b), "/", len(b))
