import json, random, torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
LETTERS = "ABCDEFGH"
N = 50

def build_prompt(ex):
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(ex["choices"]))
    n = len(ex["choices"])
    return (f"Question: {ex['question']}\n{opts}\n"
            f"Answer with a single letter ({', '.join(LETTERS[:n])}).")

ds = load_dataset("lmms-lab/ScienceQA-IMG", split="test")
random.seed(0)
idx = sorted(random.sample(range(len(ds)), 500))[:N]
greedy = {r["idx"]: r for r in map(json.loads, open("results/baseline_zeroshot.jsonl"))}
matt = {r["idx"]: r for r in map(json.loads, open("results/baseline_softmax.jsonl"))}

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32).eval()

def read_position(ex, text):
    inputs = processor(text=text, images=ex["image"].convert("RGB"), return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
    p = torch.softmax(logits, dim=-1)
    n = len(ex["choices"])
    ids = [processor.tokenizer.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS[:n]]
    letter = LETTERS[int(torch.argmax(p[ids]))]
    conf = float(torch.softmax(logits[ids], dim=-1).max())
    top = processor.tokenizer.decode([int(torch.argmax(p))])
    return letter, float(p[ids].sum()), top, conf

def template(prompt):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)

rows = []
for i in idx:
    ex = ds[i]
    text = template(build_prompt(ex))
    a = read_position(ex, text)
    b = read_position(ex, text + " Answer:")
    c = read_position(ex, template(build_prompt(ex) + "\nAnswer:"))  # exact 02_softmax_eval.py prompt
    rows.append((i, a, b, c))
    print(i, "first:", a, "| after Answer:", b, "| 02 prompt:", c, "| greedy:", greedy[i]["pred"],
          "matt softmax:", matt[i]["pred"], "gold:", greedy[i]["gold"], flush=True)

for name, k in [("first", 1), ("after Answer:", 2), ("02 prompt", 3)]:
    mass = sum(r[k][1] for r in rows) / N
    conf = sum(r[k][3] for r in rows) / N
    agree = sum(r[k][0] == greedy[r[0]]["pred"] for r in rows) / N
    acc = sum(r[k][0] == greedy[r[0]]["gold"] for r in rows) / N
    print(f"{name}: option mass={mass:.3f} letter conf={conf:.3f} agree_with_greedy={agree:.3f} acc={acc:.3f}")
print("02 prompt matches matt's pred:", sum(r[3][0] == matt[r[0]]["pred"] for r in rows), "/", N)
print("max |conf - matt conf|:", max(abs(r[3][3] - max(matt[r[0]]["probs"].values())) for r in rows))
