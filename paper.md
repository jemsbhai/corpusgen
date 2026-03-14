---
title: 'corpusgen: Language-Agnostic Speech Corpus Generation with Maximal Phoneme Coverage'
tags:
  - Python
  - speech corpus
  - phoneme coverage
  - text-to-speech
  - automatic speech recognition
authors:
  - name: Muntaser Mansur Syed
    orcid: 0000-0002-4777-6469
    affiliation: 1
affiliations:
 - name: Florida Institute of Technology
   index: 1
date: 13 March 2026
bibliography: paper.bib
---

# Summary

Building high-quality datasets for Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) systems requires meticulous engineering to ensure that all phonetic units (e.g., phonemes, diphones) of a target language are adequately represented. Traditionally, researchers have had to manually curate scripts or rely on brute-force statistical sampling, which often leaves rare linguistic sounds underrepresented or balloons the cost of voice-actor studio time. 

`corpusgen` is a comprehensive, language-agnostic Python framework designed to automatically evaluate, select, and generate phonetically balanced speech corpora. By bridging grapheme-to-phoneme (G2P) engines like `espeak-ng` with the massive PHOIBLE phonological database, the framework natively supports phonetic evaluations across more than 2,100 languages. 

To solve the corpus optimization problem, `corpusgen` provides a suite of advanced Set-Cover selection algorithms—ranging from exact Integer Linear Programming (ILP) to fast greedy approximations—minimizing the amount of text required to achieve maximal phonetic coverage. Furthermore, it introduces novel Constrained Text Generation (CTG) capabilities, allowing researchers to steer Large Language Models (LLMs) via Proximal Policy Optimization (Phon-RL) or inference-time logit steering (Phon-DATG) to dynamically generate text containing rare, missing, or specifically targeted phonetic sounds.

# Statement of need

The development of speech technologies fundamentally relies on phonetically balanced text corpora. A well-designed corpus ensures that downstream models are trained on or evaluated against the full phonetic distribution of a target language. Historically, researchers have been forced to build custom, ad-hoc scripts to map text to phonemes and statistically select sentences for each new language they study. This fragmentation limits reproducibility and creates a steep barrier to entry for low-resource language research.

While excellent standalone tools exist for grapheme-to-phoneme (G2P) conversion such as `espeak-ng` [@espeak_ng] and open-access phonological inventories such as PHOIBLE [@phoible], there has been a lack of unified, language-agnostic infrastructure connecting these resources to modern optimization algorithms and Large Language Model (LLM) text generation frameworks.

`corpusgen` addresses this gap by providing a comprehensive, modular Python library for speech corpus engineering. A critical bottleneck in global speech technology is the sheer difficulty of building resources for non-Western or low-resource languages. To solve this, `corpusgen` features a robust multilingual architecture that natively bridges grapheme-to-phoneme engines directly with the PHOIBLE phonological database. The framework includes automatic ISO 639-3 macrolanguage resolution and has been structurally validated across 40 distinct languages spanning 12 language families.

Designed for researchers in speech processing, corpus linguistics, and Natural Language Processing (NLP), the software standardizes the pipeline into distinct, extensible modules:

1. **Evaluation:** Standardized linguistic distribution metrics (e.g., normalized entropy, Jensen-Shannon divergence) and coverage trajectory tracking [@jsd_lin].
2. **Selection:** A suite of pluggable Set-Cover approximation algorithms [@set_cover_chvatal] and exact Integer Linear Programming (ILP) solvers.
3. **Generation:** An orchestration framework supporting local HuggingFace transformers and remote LLM APIs, enabling research into Constrained Text Generation (CTG) for phonetic targeting via custom Proximal Policy Optimization loops [@ppo_schulman].

By abstracting away the heavy engineering requirements of G2P mapping and distributed LLM generation, `corpusgen` allows researchers to focus directly on algorithmic design and linguistic analysis, accelerating the creation of equitable, high-quality speech resources across thousands of languages.

# State of the field

Existing tools address individual components of the speech corpus engineering pipeline but none provide an integrated, language-agnostic framework. Festival/FestVox [@black_lenzo_festvox] offers greedy bigram-based prompt selection for TTS voice building, but is primarily English-centric and tightly coupled to its synthesis architecture. Phonemizer [@phonemizer] provides robust grapheme-to-phoneme conversion—which `corpusgen` uses as a backend—but includes no selection, generation, or evaluation capabilities. Phonological CorpusTools [@pct_hall] supports phonological analysis of existing corpora (e.g., functional load, phonotactic probability) but does not address corpus construction or optimization. Beyond these tools, researchers typically rely on ad-hoc, language-specific scripts implementing greedy selection heuristics [@bozkurt_greedy], which are rarely packaged or reproducible.

`corpusgen` fills this gap by providing a unified pipeline from G2P conversion and PHOIBLE-backed inventory lookup through six pluggable selection algorithms to LLM-driven constrained text generation—capabilities that, to our knowledge, no existing package combines.

# Software design

`corpusgen` follows a modular pipeline architecture with four decoupled stages: inventory lookup, G2P conversion, selection/generation, and evaluation. Each stage communicates through plain Python data structures (lists of phoneme strings, dataclasses), allowing components to be used independently or composed into full workflows. The selection module implements a strategy pattern where all six algorithms share a common `SelectorBase` interface, making it straightforward to benchmark algorithms against each other or add new ones. The generation framework separates backend concerns (where text comes from) from orchestration logic (how coverage targets are tracked and updated), enabling the same `GenerationLoop` to drive repository selection, LLM API calls, or local model inference. Optional heavyweight dependencies (torch, pymoo, litellm) are isolated behind lazy imports and pip extras, keeping the core installation lightweight. The framework is distributed on PyPI and uses Poetry for reproducible dependency management, with a CI pipeline testing across Python 3.10, 3.12, and 3.13.

# Research impact statement

`corpusgen` was developed as part of doctoral research at the Florida Institute of Technology investigating phonetically-controlled text generation for speech corpus construction. The framework is being used to prepare experimental benchmarks for an upcoming submission to INTERSPEECH 2026. By providing a reproducible, open-source toolkit for a task that has historically required custom scripting, `corpusgen` aims to lower the barrier to entry for speech corpus research, particularly for under-resourced languages where phonetic coverage engineering is most critical.

# AI usage disclosure

Development of `corpusgen` was assisted by Claude (Anthropic), used as an interactive pair-programming and drafting tool throughout the project. Specifically, Claude assisted with code generation, test scaffolding, documentation writing, and drafting of this paper. All architectural decisions—including the choice of algorithms, module boundaries, API design, and evaluation methodology—were made by the human author. All code was reviewed, tested, and validated by the author before integration, following a strict test-driven development workflow with over 2,200 automated tests. The author takes full responsibility for the correctness and originality of the submitted work.

# Acknowledgements

This framework was developed in support of doctoral research conducted at the Florida Institute of Technology. The author thanks Marius Silaghi for guidance and feedback during the development of this work. The author also acknowledges the developers of espeak-ng and the PHOIBLE project, whose open resources make language-agnostic phonetic analysis possible.

# References