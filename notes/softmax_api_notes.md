# SmolVLM softmax API notes

Shiqi Yang, Nov 3, 2026. Independent check of `scripts/00_scienceqa_counts.py`
and `scripts/02_softmax_eval.py`.

## Summary

- The ScienceQA counts reproduce exactly, and the row-for-row claim holds.
- `02_softmax_eval.py` reproduces (same prediction on 49/50 questions), but on
  those questions it reads a position where the model is not yet answering.
  The option letters hold about 7% of the probability there on average, and the
  median is about 1%.
- Reading the probabilities after an assistant `Answer:` prefill puts about 100%
  of the probability on the options. Agreement with greedy rises from 76% to 96%.
- So most of the 25% greedy-vs-softmax disagreement looks like a position and
  prompt effect, not two complementary signals. This is based on 50 questions and
  should be rerun on the 500-question subset before we rely on it.

## How to read option probabilities

1. Build the chat text with `processor.apply_chat_template(..., add_generation_prompt=True)`.
2. Run one forward pass: `logits = model(**inputs).logits[0, -1]`. This is the
   distribution over the next token, across the whole vocabulary.
3. Get the option token ids. The letters follow a space, so use `" A"`, `" B"`, and
   so on, with `add_special_tokens=False`. Each is a single token.
4. Compute both of these:
   - `p = softmax(logits)` over the full vocabulary, then `option_mass = p[ids].sum()`
   - `softmax(logits[ids])`, the probabilities renormalized over the options
5. Save `option_mass` with every prediction. Low mass means the renormalized
   numbers are read from a position where the model is not answering.

## Counts (`scripts/check_counts.py`)

| Split | `derek-thomas` total | Image | Text-only | `lmms-lab` IMG total |
|---|---:|---:|---:|---:|
| train | 12,726 | 6,218 | 6,508 | 6,218 |
| validation | 4,241 | 2,097 | 2,144 | 2,097 |
| test | 4,241 | 2,017 | 2,224 | 2,017 |

All numbers match the progress report. The pushed `00_scienceqa_counts.py` only
counts rows, so this script also compares `question`, `choices`, `answer`, and
`hint`. The image rows of `derek-thomas` match `lmms-lab` exactly, in the same
order, in all three splits.

`hint` is non-empty for 63% of image rows (train 3,944/6,218, validation
1,331/2,097, test 1,257/2,017). The current prompts do not use it.

## Softmax position check (`scripts/check_softmax.py`)

Setup: the first 50 questions of the seed-0 subset, CPU, `float32`, torch 2.15.0,
transformers 5.22.1. Output is in `results/softmax_check_shiqi.txt`.

| Condition | Option mass (mean) | Letter confidence (mean) | Agree with greedy | Accuracy |
|---|---:|---:|---:|---:|
| First position, greedy prompt | 0.209 | 0.895 | 74% | 35/50 |
| First position, exact `02` prompt | 0.070 | 0.890 | 76% | 33/50 |
| After assistant `Answer:` prefill | 1.000 | 0.889 | 96% | 31/50 |

- **Reproduction:** the exact `02` prompt matched `baseline_softmax.jsonl` on 49/50
  predictions. Confidence differed by at most 0.063, which may come from the
  `float32` CPU run versus the original `bfloat16` run.
- **Top token:** under the `02` prompt, the most likely next token was `" Answer"`
  on 48/50 questions. The median option mass was 0.013. This fits the greedy
  output, where 390/500 responses start with `Answer:`.
- **Prompt:** adding `\nAnswer:` to the user turn does not make the model answer
  directly. It lowers option mass from 0.209 to 0.070.

## What this does not show

- It does not show that the first-position numbers are useless. On these 50
  questions they were not less accurate (33/50 vs 31/50), and the sample is too
  small to rank the methods.
- It does not settle whether softmax beats greedy. The reported +2.2 points was not
  significant even before this check (McNemar on 59 vs 48, p about 0.3).
- It says nothing about k-sampling, which has not been run yet.

## Proposed next steps

1. Rerun greedy and softmax on the same 500 questions with one shared prompt, using
   the prefill position and recording `option_mass`.
2. Decide on that basis what the softmax variant means, and keep the first-position
   version only as an explicit comparison.
3. Use the same prompt and answer position for k-sampling.
4. Decide whether to include `hint` in the prompt. It is a separate question from
   the position issue, so test it on its own.
