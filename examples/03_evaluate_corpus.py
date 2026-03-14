#!/usr/bin/env python3
"""Example 03: Evaluate a corpus for phoneme coverage.

Demonstrates how to evaluate an existing corpus of sentences against
a target phoneme inventory and inspect the coverage report.

Requirements:
    pip install corpusgen
    espeak-ng must be installed and on PATH
"""

from corpusgen import evaluate

# --- A small example corpus ---
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "A journey of a thousand miles begins with a single step.",
    "Pack my box with five dozen liquor jugs.",
]

# --- Evaluate against auto-discovered phonemes ---
report = evaluate(sentences=corpus, language="en-us")

print("=== Coverage Report (auto-discovered inventory) ===")
print(f"Language  : {report.language}")
print(f"Unit      : {report.unit}")
print(f"Sentences : {report.total_sentences}")
print(f"Coverage  : {report.coverage:.1%}")
print(f"Covered   : {len(report.covered_phonemes)} / {len(report.target_phonemes)} phonemes")
print()

# --- Evaluate against PHOIBLE inventory ---
phoible_report = evaluate(
    sentences=corpus,
    language="en-us",
    target_phonemes="phoible",
)

print("=== Coverage Report (PHOIBLE inventory) ===")
print(f"Coverage  : {phoible_report.coverage:.1%}")
print(f"Covered   : {len(phoible_report.covered_phonemes)} / "
      f"{len(phoible_report.target_phonemes)} phonemes")
print()

if phoible_report.missing_phonemes:
    print(f"Missing phonemes ({len(phoible_report.missing_phonemes)}):")
    print(f"  {', '.join(sorted(phoible_report.missing_phonemes))}")
    print()

# --- Distribution metrics ---
dist = phoible_report.distribution
print("=== Distribution Metrics ===")
print(f"Shannon entropy      : {dist.entropy:.3f} bits")
print(f"Normalized entropy   : {dist.normalized_entropy:.3f} (1.0 = perfectly uniform)")
print(f"JSD vs uniform       : {dist.jsd_uniform:.3f} (0.0 = perfectly uniform)")
print(f"Coeff. of variation  : {dist.coefficient_of_variation:.3f}")
print(f"PCD (uniform)        : {dist.pcd_uniform:.3f}")
print()

# --- Text quality metrics ---
tq = phoible_report.text_quality
print("=== Text Quality Metrics ===")
print(f"Mean sentence length : {tq.sentence_length_phonemes_mean:.1f} phonemes")
print(f"Type-token ratio     : {tq.type_token_ratio:.3f}")
print()

# --- Per-sentence details ---
print("=== Per-Sentence Breakdown ===")
for detail in phoible_report.sentence_details:
    new = len(detail.new_phonemes)
    print(f"  [{detail.index}] {detail.phoneme_count} phonemes, "
          f"{new} new → {detail.text[:50]}...")
