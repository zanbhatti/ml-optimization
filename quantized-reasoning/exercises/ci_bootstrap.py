import json
import random

PATH = "results/thinking/raw/qwen3-1.7b/proofwriter_bf16_t0.6_p0.95_k20_s1.jsonl"

# 1. Load the rows, skipping the first line (the header)
rows = []
with open(PATH) as f:
    lines = f.readlines()
    for line in lines[1:]:
        rows.append(json.loads(line))
    print(len(rows))
    print(rows[0]['depth'], rows[0]['correct'])
# TODO: turn each line after the first into a dict with json.loads and append it to rows

# 2. Keep `correct` (as 1 or 0) for rows at depth 4 or 5
data = []  # TODO
for row in rows:
    if (row['depth'] == 4 or row['depth'] == 5):
        if row['correct'] == True:
            data.append(1)
        else:
            data.append(0)

print("items:", len(data), "correct:", sum(data), "accuracy:", sum(data) / len(data))

# 3. Bootstrap: resample with replacement 10,000 times and record the accuracy each time
random.seed(0)
accs = []
for _ in range(10_000):
    sample = random.choices(data, k=len(data))
    # TODO: append this sample's accuracy to accs
    accuracy = sum(sample) / len(sample)
    accs.append(accuracy)

# 4. Sort and take the 2.5th and 97.5th percentiles
accs.sort()
print(f'2.5% percentile: {accs[250]} | 97.5% percentile: {accs[9750]}')
# TODO: print the interval