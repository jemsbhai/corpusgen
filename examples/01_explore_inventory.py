#!/usr/bin/env python3
"""Example 01: Explore a phoneme inventory.

Demonstrates how to load a PHOIBLE phoneme inventory for a language,
inspect its segments, and query by distinctive features.

Requirements:
    pip install corpusgen
"""

from corpusgen import get_inventory

# --- Load inventory for American English ---
inv = get_inventory("en-us")

print(f"Language : {inv.language_name} [{inv.iso639_3}]")
print(f"Source   : {inv.source} (PHOIBLE inventory #{inv.inventory_id})")
print(f"Total    : {inv.size} segments")
print(f"Consonants: {inv.consonant_count}, Vowels: {inv.vowel_count}, Tones: {inv.tone_count}")
print()

# --- List all phonemes ---
print("Consonants:", " ".join(inv.consonants))
print("Vowels    :", " ".join(inv.vowels))
print()

# --- Query by distinctive features ---
nasals = inv.segments_with_feature("nasal", "+")
print(f"Nasal segments ({len(nasals)}):")
for seg in nasals:
    print(f"  {seg.phoneme:4s}  class={seg.segment_class}, marginal={seg.marginal}")
print()

# --- Multi-feature query: voiceless fricatives ---
voiceless_fricatives = inv.segments_with_features({
    "continuant": "+",
    "consonantal": "+",
    "periodicGlottalSource": "-",
})
print(f"Voiceless fricatives ({len(voiceless_fricatives)}):")
for seg in voiceless_fricatives:
    print(f"  {seg.phoneme}")
print()

# --- Compare inventory sizes across languages ---
print("Inventory sizes across languages:")
for lang in ["en-us", "fr-fr", "de", "ar", "zh"]:
    try:
        other = get_inventory(lang)
        print(f"  {lang:6s} → {other.language_name:20s} {other.size} segments")
    except KeyError:
        print(f"  {lang:6s} → (not found in PHOIBLE)")
