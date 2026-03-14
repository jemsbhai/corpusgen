#!/usr/bin/env python3
"""Example 02: Select sentences for maximal phoneme coverage.

Demonstrates how to use the greedy selection algorithm to pick a minimal
subset of sentences that maximizes phoneme coverage.

Requirements:
    pip install corpusgen
    espeak-ng must be installed and on PATH
"""

from corpusgen import select_sentences

# --- Candidate sentences ---
candidates = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
    "How much wood would a woodchuck chuck?",
    "The five boxing wizards jump quickly.",
    "A journey of a thousand miles begins with a single step.",
    "Knowledge is power and enthusiasm pulls the switch.",
    "The thick fog delayed flights at the airport.",
    "Bright vixens jump and lazy fowl quack.",
    "Pack my box with five dozen liquor jugs.",
]

# --- Select sentences using greedy algorithm ---
result = select_sentences(
    candidates=candidates,
    language="en-us",
    algorithm="greedy",
    target_coverage=1.0,  # aim for full coverage of discovered phonemes
)

print(f"Algorithm : {result.algorithm}")
print(f"Coverage  : {result.coverage:.1%}")
print(f"Selected  : {result.num_selected} / {len(candidates)} sentences")
print(f"Covered   : {len(result.covered_units)} units")
print(f"Missing   : {len(result.missing_units)} units")
print(f"Time      : {result.elapsed_seconds:.3f}s")
print()

# --- Show selected sentences ---
print("Selected sentences (in selection order):")
for i, (idx, sentence) in enumerate(
    zip(result.selected_indices, result.selected_sentences), 1
):
    print(f"  {i}. [{idx}] {sentence}")
print()

# --- Show missing phonemes, if any ---
if result.missing_units:
    print(f"Missing units: {', '.join(sorted(result.missing_units))}")
else:
    print("All target phonemes covered!")
print()

# --- Try diphone-level coverage ---
diphone_result = select_sentences(
    candidates=candidates,
    language="en-us",
    algorithm="greedy",
    unit="diphone",
    target_coverage=0.8,  # diphone space is much larger; small candidate sets won't reach this
)

print(f"Diphone coverage: {diphone_result.coverage:.1%} "
      f"({diphone_result.num_selected} sentences)")
