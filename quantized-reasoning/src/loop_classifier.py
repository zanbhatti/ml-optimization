"""Classify truncated thinking outputs as looping vs. not looping.

Idea: in a loop the model copies itself, so long word sequences (n-grams)
reappear. In honest text, long n-grams almost never repeat by chance.

Run from quantized-reasoning/:
    python src/loop_classifier.py
"""
import json

PATH = "results/thinking/raw/qwen3-1.7b/proofwriter_4bit.jsonl"
TAIL_CHARS = 2000  # only look at the end of the output, where a loop would be
N = 20        # chosen 11/01: only n that gets all 9 hand labels right
CUTOFF = 0.8  # score >= CUTOFF means loop

# Hand labels from reading the first 8 stuck 0.6B 4-bit outputs (2026-10-31).
# True = loop, False = not a loop. Key 4f198c... was "circling" (ambiguous); labelled False.
HAND_LABELS = {
    "4f198ceee685788b": False,  # depth 0, circling on "kind things are red"
    "42b7987d2dab1c6c": True,  # relabeled 11/01: wanders, then ends in a verbatim loop
    "d030862b30737f2b": False,  # depth 3, contradicts itself
    "aff6bbc4ad504e81": True,   # depth 3, "both furry and green" x8
    "5faf2bf7e4b90d83": True,  # relabeled 11/01: wanders, then ends in a verbatim loop
    "4b33c517f09ad993": True,   # depth 4, 3-sentence cycle
    "82ae5b30e81c34eb": False,  # depth 4, wandering
    "d3f15647eee673da": True,   # depth 5, paragraph repeated
    "7a05b6b035a83a16": False,  # circling: verbatim repeats with insertions
}


def repeat_fraction(text, n):
    """Fraction of n-word chunks in `text` that appear more than once.

    Example: words a b a b a b with n=2 gives chunks ab, ba, ab, ba, ab.
    Every chunk appears more than once, so the result is 1.0.
    """
    words = text.split()
    if len(words) < n:
        return 0.0
    
    chunks = []
    for i in range(len(words) - n + 1):
        chunks.append(tuple(words[i:i+n]))

    
    count = {}
    for chunk in chunks:
        count[chunk] = count.get(chunk, 0) + 1
    
    chunk_count = 0
    for chunk in chunks:
        if count[chunk] >= 2:
            chunk_count += 1
        else:
            continue
    return chunk_count / len(chunks)


def load_stuck(path):
    rows = [json.loads(line) for line in open(path)][1:]  # skip the header
    return [r for r in rows if not r["closed_thinking"]]


if __name__ == "__main__":
    stuck = load_stuck(PATH)
    print(len(stuck), "stuck outputs")

    # Step A: for each n, how well does the score separate your hand labels?
    for n in (3, 5, 8, 10, 12, 20):
        scores = {r["key"]: repeat_fraction(r["output"][-TAIL_CHARS:], n)
                  for r in stuck if r["key"] in HAND_LABELS}
        print(f"\nn={n}")
        for key, is_loop in HAND_LABELS.items():
            if key not in scores: continue # line added to prevent KeyErrors from scores that doesnt exist in current path
            print(f"  {'LOOP' if is_loop else '    '}  {scores[key]:.2f}  {key}")

    # Step B (after you pick n and a threshold): count loops among all stuck outputs
    # and compare with the zlib method's 38 of 62.
    
    loops = 0
    for r in stuck:
        if repeat_fraction(r["output"][-TAIL_CHARS:], N) >= CUTOFF:
            loops += 1
    print(f"{loops} of {len(stuck)} stuck outputs are loops (n={N}, cutoff={CUTOFF})")