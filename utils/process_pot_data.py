import json
from data import split_step, find_max_step

OPENMATH_PATH = "data/openmathinstruct-2-train.json"
TRAIN_OUT   = "data/train_math.json"
META_OUT    = "data/meta_math.json"

# load the full OpenMathInstruct-2 dump
with open(OPENMATH_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

train_list = []
meta_list  = []

for item in raw:
    prob       = item["problem"]
    solution   = item["generated_solution"]
    expected   = str(item["expected_answer"]).strip()
    domain     = item.get("problem_source", "openmath")
    # accuracy = 1.0 if the model's solution actually contains the expected answer
    acc = 1.0 if expected in solution else 0.0

    # --- for train.json ---
    train_list.append({
        "prompt":        prob,
        "full_solution": solution,
        "accuracy":      acc,
        "domain":        domain
    })

    # --- for meta.json ---
    max_step = find_max_step(solution)
    if max_step == 0:
        # fallback: split on double‐newline if no explicit "Step N"
        paras = [p.strip() for p in solution.split("\n\n") if p.strip()]
        max_step = len(paras)

    # assume all steps correct in this synthetic data
    labels = [1.0] * max_step

    meta_list.append({
        "full_solution": solution,
        "step_labels":   labels
    })


# write out
with open(TRAIN_OUT, "w", encoding="utf-8") as f:
    json.dump(train_list, f, indent=2, ensure_ascii=False)

with open(META_OUT, "w", encoding="utf-8") as f:
    json.dump(meta_list, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(train_list)} examples to {TRAIN_OUT}")
print(f"Wrote {len(meta_list)} examples to {META_OUT}")
