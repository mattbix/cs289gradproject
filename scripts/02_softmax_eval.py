import argparse, json, os, random
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
        f"Answer with a single letter ({', '.join(LETTERS[:len(choices)])}).\nAnswer:"
    )

def answer_token_ids(processor, n_choices):
    ids, labels = [], []
    for L in LETTERS[:n_choices]:
        toks = processor.tokenizer.encode(" " + L, add_special_tokens=False)
        if len(toks) != 1:
            raise RuntimeError(f"' {L}' tokenizes to {toks}, not single token")
        ids.append(toks[0]); labels.append(L)
    return ids, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/baseline_softmax.jsonl")
    args = ap.parse_args()

    ds = load_dataset("lmms-lab/ScienceQA-IMG", split="test")
    random.seed(args.seed)
    idx = sorted(random.sample(range(len(ds)), min(args.n, len(ds))))

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                done.add(json.loads(line)["idx"])
    correct, total = 0, len(done)
    if done:
        with open(args.out) as f:
            for line in f:
                correct += int(json.loads(line)["ok"])

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
            out = model(**inputs)
        last_logits = out.logits[0, -1, :]

        n = len(list(ex["choices"]))
        ids, labels = answer_token_ids(processor, n)
        sub = last_logits[ids]
        probs = torch.softmax(sub.float(), dim=-1).cpu().tolist()

        pred = labels[int(torch.argmax(sub).item())]
        gold = LETTERS[ex["answer"]]
        ok = pred == gold
        correct += int(ok); total += 1

        rec = {"idx": i, "pred": pred, "gold": gold, "ok": ok,
               "probs": {L: round(p, 4) for L, p in zip(labels, probs)}}
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if total % 25 == 0 or total == len(idx):
            print(f"[{total:4d}/{len(idx)}] acc={correct/total:.4f}  probs={rec['probs']}  pred={pred} gold={gold}")

    print(f"\nFINAL zero-shot softmax accuracy: {correct/total:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()
