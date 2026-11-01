"""Re-check rate: how often does the model reopen its own reasoning?

Counts phrases like "but wait" per 1,000 words, comparing 4-bit and BF16 outputs
on the same puzzles (the ones where 1.7B 4-bit got stuck).

Run from quantized-reasoning/:
    python src/recheck_rate.py
"""
import json

FOURBIT = "results/thinking/raw/qwen3-1.7b/proofwriter_4bit.jsonl"
BF16 = "results/thinking/raw/qwen3-1.7b/proofwriter_bf16.jsonl"

PHRASES = ["but wait", "let me check again", "let me think again"]


def load(path):
    """Return a dict: puzzle key -> row."""
    rows = [json.loads(line) for line in open(path)][1:]  # skip the header
    return {r["key"]: r for r in rows}


def recheck_rate(text):
    """Re-check phrases per 1,000 words."""
    lower = text.lower()
    words = len(text.split())
    if words == 0:
        return 0.0
    
    count = {}
    for phrase in PHRASES:
        count[phrase] = lower.count(phrase)
    

    
    hits = sum(count.values())
    return (hits / words) * 1000



if __name__ == "__main__":
    four = load(FOURBIT)
    bf16 = load(BF16)
    stuck_keys = [k for k, r in four.items() if not r["closed_thinking"]]
    print(len(stuck_keys), "puzzles where 4-bit got stuck")

    
    four_bit_list = []
    bf_sixten_list = []
    for key in stuck_keys:
        four_bit_list.append(recheck_rate(four[key]['output']))
        bf_sixten_list.append(recheck_rate(bf16[key]['output']))


    
    average_four = sum(four_bit_list) / len(four_bit_list)
    average_bf = sum(bf_sixten_list) / len(bf_sixten_list)
    ratio = average_four / average_bf
    print(f'4-bit: {average_four} | BF16: {average_bf} | ratio: {ratio}')