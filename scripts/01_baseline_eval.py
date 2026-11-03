import argparse, json, os, random, re
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
LETTERS = "ABCDEFGH"

def build_prompt(ex):
    choices = list(ex["choices"])
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"Question: {ex['question']}\n{opts}\n"
        f"Answer with a single letter ({', '.join(LETTERS[:len(choices)])})."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/baseline_zeroshot.jsonl")
    args = ap.parse_args()

    ds = load_dataset("lmms-lab/ScienceQA-IMG", split="test")
    random.seed(args.seed)
    idx = sorted(random.sample(range(len(ds)), min(args.n, len(ds))))

    done = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done[r["idx"]] = r
    correct = sum(1 for r in done.values() if r["ok"])
    total = len(done)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    ).eval()

    for i in idx:
        if i in done:
            continue
        ex = ds[i]
        prompt = build_prompt(ex)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=text, images=ex["image"].convert("RGB"), return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)

        gen = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        n = len(list(ex["choices"]))
        matches = re.findall(rf'\b([{LETTERS[:n]}])\b', gen.upper())
        pred = matches[-1] if matches else None
        gold = LETTERS[ex["answer"]]
        ok = pred == gold
        correct += int(ok)
        total += 1

        rec = {"idx": i, "pred": pred, "gold": gold, "ok": ok, "raw": gen}
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if total % 25 == 0 or total == len(idx):
            print(f"[{total:4d}/{len(idx)}] acc={correct/total:.4f}  raw={gen!r} pred={pred} gold={gold}")

    print(f"\nFINAL zero-shot greedy accuracy: {correct/total:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()
