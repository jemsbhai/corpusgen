"""corpusgen generate — generate sentences for maximal phoneme coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from corpusgen import get_inventory
from corpusgen.generate.backends.llm_api import LLMBackend
from corpusgen.generate.backends.local import LocalBackend
from corpusgen.generate.backends.repository import RepositoryBackend
from corpusgen.generate.phon_ctg.loop import GenerationLoop, StoppingCriteria
from corpusgen.generate.phon_ctg.scorer import PhoneticScorer
from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory
from corpusgen.generate.phon_datg import DATGStrategy
from corpusgen.generate.phon_rl.policy import PhonRLStrategy
from corpusgen.generate.scorers.fluency import PerplexityFluencyScorer


def _parse_weights(value: str) -> dict[str, float]:
    """Parse weights from an inline string or JSON file path.

    If the value is a path to an existing file, load it as JSON.
    Otherwise, parse as comma-separated key:value pairs.

    Args:
        value: Inline "p:2.0,b:1.5" or path to a JSON file.

    Returns:
        Dict mapping phoneme strings to float weights.

    Raises:
        click.BadParameter: If the format is invalid.
    """
    path = Path(value)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise click.BadParameter(
                    f"Weights JSON file must contain a dict, got {type(data).__name__}."
                )
            return {str(k): float(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError) as exc:
            raise click.BadParameter(f"Invalid weights JSON file: {exc}")

    # Parse inline format: "p:2.0,b:1.5,ʃ:3.0"
    weights: dict[str, float] = {}
    for pair in value.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise click.BadParameter(
                f"Invalid weight pair {pair!r}. Expected 'phoneme:weight' format."
            )
        key, val = pair.rsplit(":", 1)
        key = key.strip()
        try:
            weights[key] = float(val.strip())
        except ValueError:
            raise click.BadParameter(
                f"Invalid weight value for {key!r}: {val!r}. Must be a number."
            )
    return weights


def _resolve_prompt_template(value: str | None) -> str | None:
    """Resolve a prompt template from an inline string or file path.

    If the value is a path to an existing file, load its contents.
    Otherwise, return the string as-is.

    Args:
        value: Inline template string, file path, or None.

    Returns:
        Resolved template string, or None.
    """
    if value is None:
        return None

    path = Path(value)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()

    return value


# --- Backend-specific flag sets for validation ---

_REPOSITORY_ONLY = {"file"}
_LLM_API_ONLY = {"api_key", "llm_temperature", "llm_max_tokens"}
_LOCAL_ONLY = {"device", "quantization", "local_temperature", "local_max_tokens"}

_BACKEND_FLAGS = {
    "repository": _REPOSITORY_ONLY,
    "llm_api": _LLM_API_ONLY,
    "local": _LOCAL_ONLY,
}

# Flags that are invalid for each backend (flags belonging to OTHER backends)
_INVALID_FOR = {
    "repository": _LLM_API_ONLY | _LOCAL_ONLY | {"model"},
    "llm_api": _REPOSITORY_ONLY | _LOCAL_ONLY,
    "local": _REPOSITORY_ONLY | _LLM_API_ONLY,
}

# Map CLI flag names to display names for error messages
_FLAG_DISPLAY = {
    "file": "--file",
    "model": "--model",
    "api_key": "--api-key",
    "llm_temperature": "--llm-temperature",
    "llm_max_tokens": "--llm-max-tokens",
    "device": "--device",
    "quantization": "--quantization",
    "local_temperature": "--local-temperature",
    "local_max_tokens": "--local-max-tokens",
}


@click.command()
@click.option(
    "--backend", "-b",
    required=True,
    type=click.Choice(["repository", "llm_api", "local"]),
    help="Generation backend: repository (sentence pool), llm_api (LLM via litellm), local (HuggingFace).",
)
@click.option(
    "--language", "-l",
    required=True,
    help="Language code for G2P and PHOIBLE lookup (e.g., en-us, fr-fr).",
)
@click.option(
    "--file", "-f",
    "input_file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Input file with one sentence per line (repository backend).",
)
@click.option(
    "--model", "-m",
    default=None,
    help="Model string: litellm model (llm_api) or HuggingFace model ID (local).",
)
@click.option(
    "--api-key",
    "api_key",
    default=None,
    help="API key for LLM provider (llm_api backend). Falls back to environment variables.",
)
@click.option(
    "--target", "-t",
    "target_source",
    default="phoible",
    help='Target phoneme inventory source. Default: "phoible".',
)
@click.option(
    "--phonemes",
    default=None,
    help="Additional phonemes to add to the target inventory (comma-separated IPA).",
)
@click.option(
    "--unit", "-u",
    type=click.Choice(["phoneme", "diphone", "triphone"]),
    default="phoneme",
    help="Coverage unit type. Default: phoneme.",
)
@click.option(
    "--weights", "-w",
    default=None,
    help='Priority weights: inline "p:2.0,b:1.5" or path to JSON file.',
)
@click.option(
    "--target-coverage",
    type=float,
    default=1.0,
    help="Stop when this coverage fraction is reached. Default: 1.0.",
)
@click.option(
    "--max-sentences", "-n",
    type=int,
    default=None,
    help="Maximum number of sentences to generate.",
)
@click.option(
    "--max-iterations",
    type=int,
    default=None,
    help="Maximum generation loop iterations.",
)
@click.option(
    "--timeout",
    type=float,
    default=None,
    help="Wall-clock time limit in seconds.",
)
@click.option(
    "--candidates", "-k",
    type=int,
    default=5,
    help="Candidates per iteration. Default: 5.",
)
@click.option(
    "--llm-temperature",
    type=float,
    default=0.8,
    help="Sampling temperature for llm_api backend. Default: 0.8.",
)
@click.option(
    "--llm-max-tokens",
    type=int,
    default=1024,
    help="Maximum tokens for llm_api backend. Default: 1024.",
)
@click.option(
    "--local-temperature",
    type=float,
    default=0.8,
    help="Sampling temperature for local backend. Default: 0.8.",
)
@click.option(
    "--local-max-tokens",
    type=int,
    default=256,
    help="Maximum new tokens for local backend. Default: 256.",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "auto"]),
    default=None,
    help="Device for local backend. Default: auto-detect.",
)
@click.option(
    "--quantization",
    type=click.Choice(["none", "4bit", "8bit"]),
    default=None,
    help="Quantization for local backend. Default: none.",
)
@click.option(
    "--dataset",
    default=None,
    help="HuggingFace dataset name for repository backend (e.g., 'wikitext'). Mutually exclusive with --file.",
)
@click.option(
    "--text-column",
    default="text",
    help="Column name containing text in HuggingFace dataset. Default: text.",
)
@click.option(
    "--split",
    "dataset_split",
    default=None,
    help="HuggingFace dataset split (e.g., 'train', 'test').",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum samples to load from HuggingFace dataset.",
)
@click.option(
    "--prompt-template",
    default=None,
    help='Custom prompt template (inline string or file path). Must contain {target_units}. For llm_api/local backends.',
)
@click.option(
    "--guidance",
    type=click.Choice(["none", "datg", "rl"]),
    default="none",
    help="Guidance strategy for local backend: none, datg (logit steering), rl (RL-tuned). Default: none.",
)
@click.option(
    "--guidance-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="JSON config file for guidance strategy. Overrides flat flags.",
)
@click.option(
    "--datg-boost",
    type=float,
    default=5.0,
    help="DATG: additive boost for attribute token logits. Default: 5.0.",
)
@click.option(
    "--datg-penalty",
    type=float,
    default=-5.0,
    help="DATG: additive penalty for anti-attribute token logits. Default: -5.0.",
)
@click.option(
    "--datg-anti-mode",
    type=click.Choice(["covered", "frequency"]),
    default="covered",
    help="DATG: anti-attribute mode. Default: covered.",
)
@click.option(
    "--datg-freq-threshold",
    type=int,
    default=10,
    help="DATG: frequency threshold for frequency anti-attribute mode. Default: 10.",
)
@click.option(
    "--datg-batch-size",
    type=int,
    default=512,
    help="DATG: batch size for vocabulary phonemization. Default: 512.",
)
@click.option(
    "--rl-adapter-path",
    default=None,
    help="RL: path to PEFT/LoRA adapter checkpoint.",
)
@click.option(
    "--coverage-weight",
    type=float,
    default=1.0,
    help="Weight for coverage component in composite score. Default: 1.0.",
)
@click.option(
    "--phonotactic-weight",
    type=float,
    default=0.0,
    help="Weight for phonotactic component. Requires --phonotactic-scorer. Default: 0.0.",
)
@click.option(
    "--phonotactic-scorer",
    type=click.Choice(["none", "ngram"]),
    default="none",
    help="Phonotactic scorer: none or ngram (n-gram transition model). Default: none.",
)
@click.option(
    "--phonotactic-corpus",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Reference corpus for corpus-trained phonotactic model (one sentence per line).",
)
@click.option(
    "--phonotactic-n",
    type=int,
    default=2,
    help="N-gram order for phonotactic scorer. Default: 2 (bigram).",
)
@click.option(
    "--fluency-weight",
    type=float,
    default=0.0,
    help="Weight for fluency component. Requires --fluency-scorer. Default: 0.0.",
)
@click.option(
    "--fluency-scorer",
    type=click.Choice(["none", "perplexity"]),
    default="none",
    help="Fluency scorer: none or perplexity (LM perplexity). Default: none.",
)
@click.option(
    "--fluency-model",
    default=None,
    help="Model for perplexity fluency scorer. Defaults to --model for local backend.",
)
@click.option(
    "--fluency-device",
    type=click.Choice(["cuda", "cpu", "auto"]),
    default=None,
    help="Device for fluency model. Default: auto-detect.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format. Default: text.",
)
@click.option(
    "--output", "-o",
    "output_file",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write generated sentences to this file (one per line).",
)
def generate_cmd(
    backend: str,
    language: str,
    input_file: str | None,
    model: str | None,
    api_key: str | None,
    target_source: str,
    phonemes: str | None,
    unit: str,
    weights: str | None,
    target_coverage: float,
    max_sentences: int | None,
    max_iterations: int | None,
    timeout: float | None,
    candidates: int,
    llm_temperature: float,
    llm_max_tokens: int,
    local_temperature: float,
    local_max_tokens: int,
    device: str | None,
    quantization: str | None,
    dataset: str | None,
    text_column: str,
    dataset_split: str | None,
    max_samples: int | None,
    prompt_template: str | None,
    guidance: str,
    guidance_config: str | None,
    datg_boost: float,
    datg_penalty: float,
    datg_anti_mode: str,
    datg_freq_threshold: int,
    datg_batch_size: int,
    rl_adapter_path: str | None,
    coverage_weight: float,
    phonotactic_weight: float,
    phonotactic_scorer: str,
    phonotactic_corpus: str | None,
    phonotactic_n: int,
    fluency_weight: float,
    fluency_scorer: str,
    fluency_model: str | None,
    fluency_device: str | None,
    output_format: str,
    output_file: str | None,
) -> None:
    """Generate sentences targeting maximal phoneme coverage.

    \b
    Examples:
        corpusgen generate -b repository -l en-us -f pool.txt
        corpusgen generate -b llm_api -l en-us -m openai/gpt-4o-mini
        corpusgen generate -b local -l en-us -m gpt2 --device cuda
        corpusgen generate -b repository -l en-us -f pool.txt --phonemes "ʃ,ʒ,θ"
        corpusgen generate -b repository -l en-us -f pool.txt --weights "p:2.0,b:1.5"
    """
    # ---------------------------------------------------------------
    # 1. Validate backend-specific flags
    # ---------------------------------------------------------------
    _validate_backend_flags(
        backend=backend,
        input_file=input_file,
        model=model,
        api_key=api_key,
        device=device,
        quantization=quantization,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        local_temperature=local_temperature,
        local_max_tokens=local_max_tokens,
    )

    # Validate dataset/file flags
    if backend == "repository" and input_file is None and dataset is None:
        click.echo(
            "Error: repository backend requires --file or --dataset.",
            err=True,
        )
        sys.exit(1)

    if dataset is not None and backend != "repository":
        click.echo(
            f"Error: --dataset is only valid for the repository backend, "
            f"not {backend}.",
            err=True,
        )
        sys.exit(1)

    if dataset is not None and input_file is not None:
        click.echo(
            "Error: --dataset and --file are mutually exclusive. "
            "Use one or the other.",
            err=True,
        )
        sys.exit(1)

    # Validate prompt template
    resolved_prompt_template = _resolve_prompt_template(prompt_template)
    if resolved_prompt_template is not None:
        if backend == "repository":
            click.echo(
                "Error: --prompt-template is not valid for the repository backend.",
                err=True,
            )
            sys.exit(1)
        if "{target_units}" not in resolved_prompt_template:
            click.echo(
                "Error: --prompt-template must contain {target_units} placeholder.",
                err=True,
            )
            sys.exit(1)

    # Validate guidance flags
    if guidance != "none" and backend != "local":
        click.echo(
            f"Error: --guidance is only valid for the local backend, "
            f"not {backend}.",
            err=True,
        )
        sys.exit(1)

    # Validate scorer flags
    _validate_scorer_flags(
        backend=backend,
        model=model,
        phonotactic_weight=phonotactic_weight,
        phonotactic_scorer=phonotactic_scorer,
        fluency_weight=fluency_weight,
        fluency_scorer=fluency_scorer,
        fluency_model=fluency_model,
    )

    # ---------------------------------------------------------------
    # 2. Safety warning for unbounded generation
    # ---------------------------------------------------------------
    if max_sentences is None and max_iterations is None and timeout is None:
        click.echo(
            "Warning: No safety stopping limit set (--max-sentences, "
            "--max-iterations, or --timeout). Generation will run until "
            "target coverage is reached or the backend is exhausted.",
            err=True,
        )

    # ---------------------------------------------------------------
    # 3. Build target phoneme list
    # ---------------------------------------------------------------
    try:
        inv = get_inventory(language)
        target_phoneme_list = list(inv.phonemes)
    except KeyError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Additive phonemes
    if phonemes is not None:
        extra = [p.strip() for p in phonemes.split(",") if p.strip()]
        # Union: add only new phonemes
        existing = set(target_phoneme_list)
        for p in extra:
            if p not in existing:
                target_phoneme_list.append(p)
                existing.add(p)

    # Parse weights
    parsed_weights: dict[str, float] | None = None
    if weights is not None:
        try:
            parsed_weights = _parse_weights(weights)
        except click.BadParameter as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)

    # ---------------------------------------------------------------
    # 4. Build components
    # ---------------------------------------------------------------

    # Target inventory
    targets = PhoneticTargetInventory(
        target_phonemes=target_phoneme_list,
        unit=unit,
        weights=parsed_weights,
    )

    # Stopping criteria
    stopping = StoppingCriteria(
        target_coverage=target_coverage,
        max_sentences=max_sentences,
        max_iterations=max_iterations,
        timeout_seconds=timeout,
    )

    # Guidance strategy (local backend only)
    guidance_strategy = _build_guidance_strategy(
        guidance=guidance,
        guidance_config=guidance_config,
        targets=targets,
        language=language,
        datg_boost=datg_boost,
        datg_penalty=datg_penalty,
        datg_anti_mode=datg_anti_mode,
        datg_freq_threshold=datg_freq_threshold,
        datg_batch_size=datg_batch_size,
        rl_adapter_path=rl_adapter_path,
    )

    # Backend (built before scorer to enable model sharing)
    try:
        backend_obj = _build_backend(
            backend=backend,
            language=language,
            input_file=input_file,
            model=model,
            api_key=api_key,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            local_temperature=local_temperature,
            local_max_tokens=local_max_tokens,
            device=device,
            quantization=quantization,
            guidance_strategy=guidance_strategy,
            dataset=dataset,
            text_column=text_column,
            dataset_split=dataset_split,
            max_samples=max_samples,
            prompt_template=resolved_prompt_template,
        )
    except (ValueError, ImportError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Scorer (with optional built-in phonotactic/fluency hooks)
    phonotactic_hook = _build_phonotactic_scorer(
        phonotactic_scorer=phonotactic_scorer,
        target_phoneme_list=target_phoneme_list,
        phonotactic_n=phonotactic_n,
        phonotactic_corpus=phonotactic_corpus,
        language=language,
    )
    fluency_hook = _build_fluency_scorer(
        fluency_scorer=fluency_scorer,
        fluency_model=fluency_model,
        fluency_device=fluency_device,
        backend=backend,
        model=model,
        backend_obj=backend_obj,
    )
    scorer = PhoneticScorer(
        targets=targets,
        phonotactic_scorer=phonotactic_hook,
        fluency_scorer=fluency_hook,
        coverage_weight=coverage_weight,
        phonotactic_weight=phonotactic_weight,
        fluency_weight=fluency_weight,
    )

    # ---------------------------------------------------------------
    # 5. Run generation loop
    # ---------------------------------------------------------------
    loop = GenerationLoop(
        backend=backend_obj,
        targets=targets,
        scorer=scorer,
        stopping_criteria=stopping,
        candidates_per_iteration=candidates,
    )

    try:
        gen_result = loop.run()
    except Exception as exc:
        click.echo(f"Error during generation: {exc}", err=True)
        sys.exit(1)

    # ---------------------------------------------------------------
    # 6. Output
    # ---------------------------------------------------------------

    # Write output file if requested
    if output_file:
        Path(output_file).write_text(
            "\n".join(gen_result.generated_sentences) + "\n",
            encoding="utf-8",
        )
        click.echo(f"Wrote {gen_result.num_generated} sentences to {output_file}")

    # Display results
    if output_format == "json":
        data = {
            "generated_sentences": gen_result.generated_sentences,
            "coverage": gen_result.coverage,
            "covered_units": sorted(gen_result.covered_units),
            "missing_units": sorted(gen_result.missing_units),
            "unit": gen_result.unit,
            "backend": gen_result.backend,
            "elapsed_seconds": gen_result.elapsed_seconds,
            "iterations": gen_result.iterations,
            "stop_reason": gen_result.stop_reason,
            "num_generated": gen_result.num_generated,
        }
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        click.echo(f"Generated {gen_result.num_generated} sentences")
        click.echo(f"Coverage: {gen_result.coverage * 100:.1f}%")
        click.echo(f"Backend: {gen_result.backend}")
        click.echo(f"Stop reason: {gen_result.stop_reason}")
        click.echo(f"Time: {gen_result.elapsed_seconds:.2f}s")
        if gen_result.missing_units:
            missing = sorted(gen_result.missing_units)
            if len(missing) > 30:
                display = ", ".join(missing[:30]) + f" (+{len(missing) - 30} more)"
            else:
                display = ", ".join(missing)
            click.echo(f"Missing: {display}")


def _validate_backend_flags(
    backend: str,
    input_file: str | None,
    model: str | None,
    api_key: str | None,
    device: str | None,
    quantization: str | None,
    llm_temperature: float,
    llm_max_tokens: int,
    local_temperature: float,
    local_max_tokens: int,
) -> None:
    """Validate that flags are appropriate for the chosen backend."""
    # Required flags per backend
    # (repository needs --file or --dataset; checked later for mutual exclusivity)

    if backend in ("llm_api", "local") and model is None:
        click.echo(f"Error: --model is required for the {backend} backend.", err=True)
        sys.exit(1)

    # Reject flags that belong to other backends
    # We check for non-default values to detect explicit usage.
    invalid_checks: list[tuple[str, bool]] = []

    if backend != "llm_api":
        if api_key is not None:
            invalid_checks.append(("--api-key", True))
        if llm_temperature != 0.8:
            invalid_checks.append(("--llm-temperature", True))
        if llm_max_tokens != 1024:
            invalid_checks.append(("--llm-max-tokens", True))

    if backend != "local":
        if device is not None:
            invalid_checks.append(("--device", True))
        if quantization is not None:
            invalid_checks.append(("--quantization", True))
        if local_temperature != 0.8:
            invalid_checks.append(("--local-temperature", True))
        if local_max_tokens != 256:
            invalid_checks.append(("--local-max-tokens", True))

    if backend != "repository":
        if input_file is not None:
            invalid_checks.append(("--file", True))

    for flag_name, is_set in invalid_checks:
        if is_set:
            click.echo(
                f"Error: {flag_name} is not valid for the {backend} backend.",
                err=True,
            )
            sys.exit(1)


def _build_backend(
    backend: str,
    language: str,
    input_file: str | None,
    model: str | None,
    api_key: str | None,
    llm_temperature: float,
    llm_max_tokens: int,
    local_temperature: float,
    local_max_tokens: int,
    device: str | None,
    quantization: str | None,
    guidance_strategy: object | None = None,
    dataset: str | None = None,
    text_column: str = "text",
    dataset_split: str | None = None,
    max_samples: int | None = None,
    prompt_template: str | None = None,
):
    """Construct the appropriate GenerationBackend."""
    if backend == "repository":
        if dataset is not None:
            return RepositoryBackend.from_huggingface(
                dataset_name=dataset,
                text_column=text_column,
                split=dataset_split,
                language=language,
                max_samples=max_samples,
            )
        text = Path(input_file).read_text(encoding="utf-8")  # type: ignore[arg-type]
        sentences = [line.strip() for line in text.splitlines() if line.strip()]
        if not sentences:
            raise ValueError("Input file contains no sentences.")
        return RepositoryBackend.from_texts(sentences, language=language)

    if backend == "llm_api":
        kwargs: dict = {
            "model": model,
            "language": language,
            "api_key": api_key,
            "temperature": llm_temperature,
            "max_tokens": llm_max_tokens,
        }
        if prompt_template is not None:
            kwargs["prompt_template"] = prompt_template
        return LLMBackend(**kwargs)

    if backend == "local":
        # Convert "none" string to None for quantization
        quant = None if quantization is None or quantization == "none" else quantization
        kwargs_local: dict = {
            "model_name": model,
            "language": language,
            "device": device,
            "quantization": quant,
            "temperature": local_temperature,
            "max_new_tokens": local_max_tokens,
            "guidance_strategy": guidance_strategy,
        }
        if prompt_template is not None:
            kwargs_local["prompt_template"] = prompt_template
        return LocalBackend(**kwargs_local)

    raise ValueError(f"Unknown backend: {backend!r}")


def _validate_scorer_flags(
    backend: str,
    model: str | None,
    phonotactic_weight: float,
    phonotactic_scorer: str,
    fluency_weight: float,
    fluency_scorer: str,
    fluency_model: str | None,
) -> None:
    """Validate scorer-related flag combinations."""
    if phonotactic_weight > 0.0 and phonotactic_scorer == "none":
        click.echo(
            "Error: --phonotactic-weight > 0 requires --phonotactic-scorer "
            "to be set (e.g., --phonotactic-scorer ngram).",
            err=True,
        )
        sys.exit(1)

    if fluency_weight > 0.0 and fluency_scorer == "none":
        click.echo(
            "Error: --fluency-weight > 0 requires --fluency-scorer "
            "to be set (e.g., --fluency-scorer perplexity).",
            err=True,
        )
        sys.exit(1)

    if fluency_scorer == "perplexity" and fluency_model is None:
        # Can fall back to --model for local backend
        if backend == "local" and model is not None:
            pass  # will use --model
        else:
            click.echo(
                "Error: --fluency-scorer perplexity requires --fluency-model "
                "(or --model with local backend).",
                err=True,
            )
            sys.exit(1)


def _phonemize_corpus(
    corpus_path: str,
    language: str,
) -> list[list[str]]:
    """Read a text file and phonemize each line via G2P.

    Args:
        corpus_path: Path to text file (one sentence per line).
        language: Language code for G2P.

    Returns:
        List of phoneme sequences.
    """
    from corpusgen.g2p.manager import G2PManager

    text = Path(corpus_path).read_text(encoding="utf-8")
    sentences = [line.strip() for line in text.splitlines() if line.strip()]

    if not sentences:
        return []

    g2p = G2PManager()
    results = g2p.phonemize_batch(sentences, language=language)
    return [r.phonemes for r in results if r.phonemes]


def _build_phonotactic_scorer(
    phonotactic_scorer: str,
    target_phoneme_list: list[str],
    phonotactic_n: int,
    phonotactic_corpus: str | None,
    language: str,
):
    """Build the phonotactic scorer callable, or None."""
    if phonotactic_scorer == "none":
        return None

    from corpusgen.generate.scorers.phonotactic import NgramPhonotacticScorer

    if phonotactic_corpus is not None:
        # Corpus-trained mode
        sequences = _phonemize_corpus(phonotactic_corpus, language)
        return NgramPhonotacticScorer.from_corpus(sequences, n=phonotactic_n)

    # Inventory-derived mode (PHOIBLE baseline)
    return NgramPhonotacticScorer(
        phonemes=target_phoneme_list,
        n=phonotactic_n,
    )


def _build_fluency_scorer(
    fluency_scorer: str,
    fluency_model: str | None,
    fluency_device: str | None,
    backend: str,
    model: str | None,
    backend_obj: object | None = None,
):
    """Build the fluency scorer callable, or None.

    When the local backend is used and the fluency model matches the
    backend model (or defaults to it), we share the model to avoid
    loading it twice — important for VRAM efficiency.
    """
    if fluency_scorer == "none":
        return None

    # Resolve model name: explicit --fluency-model, or fallback to --model for local
    resolved_model = fluency_model
    if resolved_model is None and backend == "local":
        resolved_model = model

    # Model sharing: if local backend and same model, share via from_model
    if (
        backend == "local"
        and backend_obj is not None
        and (fluency_model is None or fluency_model == model)
        and hasattr(backend_obj, "_ensure_loaded")
    ):
        backend_obj._ensure_loaded()
        return PerplexityFluencyScorer.from_model(
            backend_obj._model,
            backend_obj._tokenizer,
        )

    return PerplexityFluencyScorer(
        model_name=resolved_model or "gpt2",
        device=fluency_device,
    )


def _build_guidance_strategy(
    guidance: str,
    guidance_config: str | None,
    targets: PhoneticTargetInventory,
    language: str,
    datg_boost: float,
    datg_penalty: float,
    datg_anti_mode: str,
    datg_freq_threshold: int,
    datg_batch_size: int,
    rl_adapter_path: str | None,
):
    """Build the guidance strategy, or None."""
    if guidance == "none":
        return None

    # Load config file if provided (overrides flat flags)
    config: dict = {}
    if guidance_config is not None:
        config = json.loads(
            Path(guidance_config).read_text(encoding="utf-8")
        )

    if guidance == "datg":
        kwargs = {
            "targets": targets,
            "language": language,
            "boost_strength": config.get("boost_strength", datg_boost),
            "penalty_strength": config.get("penalty_strength", datg_penalty),
            "anti_attribute_mode": config.get("anti_attribute_mode", datg_anti_mode),
            "frequency_threshold": config.get("frequency_threshold", datg_freq_threshold),
            "batch_size": config.get("batch_size", datg_batch_size),
        }
        return DATGStrategy(**kwargs)

    if guidance == "rl":
        adapter = config.get("adapter_path", rl_adapter_path)
        return PhonRLStrategy(adapter_path=adapter)

    raise ValueError(f"Unknown guidance strategy: {guidance!r}")
