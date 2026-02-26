"""corpusgen select — select optimal sentences from a candidate file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from corpusgen.select import select_sentences


@click.command()
@click.option(
    "--file", "-f",
    "input_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Input file with one candidate sentence per line.",
)
@click.option(
    "--language", "-l",
    required=True,
    help="Language code for G2P (e.g., en-us, fr-fr).",
)
@click.option(
    "--target", "-t",
    "target_phonemes",
    default=None,
    help='Target phoneme inventory. Use "phoible" for automatic PHOIBLE lookup, '
    "or omit to derive from the candidates.",
)
@click.option(
    "--unit", "-u",
    type=click.Choice(["phoneme", "diphone", "triphone"]),
    default="phoneme",
    help="Coverage unit type. Default: phoneme.",
)
@click.option(
    "--algorithm", "-a",
    default="greedy",
    help="Selection algorithm: greedy, celf, stochastic, ilp, distribution, nsga2. Default: greedy.",
)
@click.option(
    "--max-sentences", "-n",
    type=int,
    default=None,
    help="Maximum number of sentences to select.",
)
@click.option(
    "--target-coverage",
    type=float,
    default=1.0,
    help="Stop when this coverage fraction is reached. Default: 1.0.",
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
    help="Write selected sentences to this file (one per line).",
)
def select_cmd(
    input_file: str,
    language: str,
    target_phonemes: str | None,
    unit: str,
    algorithm: str,
    max_sentences: int | None,
    target_coverage: float,
    output_format: str,
    output_file: str | None,
) -> None:
    """Select optimal sentences from a candidate file for maximal phoneme coverage.

    \b
    Examples:
        corpusgen select --file candidates.txt --language en-us
        corpusgen select -f pool.txt -l en-us --algorithm celf --max-sentences 50
        corpusgen select -f pool.txt -l fr-fr --target phoible --format json
        corpusgen select -f pool.txt -l en-us --output selected.txt
    """
    # Read candidates
    text = Path(input_file).read_text(encoding="utf-8")
    candidates = [line.strip() for line in text.splitlines() if line.strip()]

    if not candidates:
        click.echo("Error: Input file contains no sentences.", err=True)
        sys.exit(1)

    # Run selection
    try:
        result = select_sentences(
            candidates,
            language=language,
            target_phonemes=target_phonemes,
            unit=unit,
            algorithm=algorithm,
            max_sentences=max_sentences,
            target_coverage=target_coverage,
        )
    except (ValueError, ImportError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Write output file if requested
    if output_file:
        Path(output_file).write_text(
            "\n".join(result.selected_sentences) + "\n",
            encoding="utf-8",
        )
        click.echo(f"Wrote {result.num_selected} sentences to {output_file}")

    # Display results
    if output_format == "json":
        data = {
            "selected_indices": result.selected_indices,
            "selected_sentences": result.selected_sentences,
            "coverage": result.coverage,
            "covered_units": sorted(result.covered_units),
            "missing_units": sorted(result.missing_units),
            "unit": result.unit,
            "algorithm": result.algorithm,
            "elapsed_seconds": result.elapsed_seconds,
            "num_selected": result.num_selected,
            "num_candidates": len(candidates),
        }
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        click.echo(
            f"Selected {result.num_selected} of {len(candidates)} sentences"
        )
        click.echo(f"Coverage: {result.coverage * 100:.1f}%")
        click.echo(f"Algorithm: {result.algorithm}")
        click.echo(f"Time: {result.elapsed_seconds:.2f}s")
        if result.missing_units:
            missing = sorted(result.missing_units)
            if len(missing) > 30:
                display = ", ".join(missing[:30]) + f" (+{len(missing) - 30} more)"
            else:
                display = ", ".join(missing)
            click.echo(f"Missing: {display}")
