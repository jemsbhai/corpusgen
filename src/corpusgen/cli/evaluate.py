"""corpusgen evaluate — evaluate text for phoneme coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from corpusgen.evaluate import evaluate
from corpusgen.evaluate.report import Verbosity


@click.command()
@click.argument("sentences", nargs=-1)
@click.option(
    "--file", "-f",
    "input_file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Input file with one sentence per line.",
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
    "or omit to derive from the corpus.",
)
@click.option(
    "--unit", "-u",
    type=click.Choice(["phoneme", "diphone", "triphone"]),
    default="phoneme",
    help="Coverage unit type. Default: phoneme.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "jsonld"]),
    default="text",
    help="Output format. Default: text.",
)
@click.option(
    "--verbosity", "-v",
    type=click.Choice(["minimal", "normal", "verbose"]),
    default="normal",
    help="Verbosity level for text output. Default: normal.",
)
def evaluate_cmd(
    sentences: tuple[str, ...],
    input_file: str | None,
    language: str,
    target_phonemes: str | None,
    unit: str,
    output_format: str,
    verbosity: str,
) -> None:
    """Evaluate text for phoneme coverage.

    Accepts inline sentences as arguments, or a file via --file
    (one sentence per line).

    \b
    Examples:
        corpusgen evaluate "The cat sat." --language en-us
        corpusgen evaluate --file corpus.txt --language en-us --target phoible
        corpusgen evaluate --file corpus.txt -l fr-fr --unit diphone --format json
    """
    # Resolve input sentences
    sentence_list: list[str] = []
    if input_file:
        text = Path(input_file).read_text(encoding="utf-8")
        sentence_list = [line.strip() for line in text.splitlines() if line.strip()]
    elif sentences:
        sentence_list = list(sentences)

    if not sentence_list:
        click.echo("Error: Provide sentences as arguments or via --file.", err=True)
        sys.exit(1)

    # Run evaluation
    report = evaluate(
        sentence_list,
        language=language,
        target_phonemes=target_phonemes,
        unit=unit,
    )

    # Output
    if output_format == "json":
        click.echo(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    elif output_format == "jsonld":
        click.echo(json.dumps(report.to_jsonld_ex(), ensure_ascii=False, indent=2))
    else:
        verbosity_level = Verbosity(verbosity)
        click.echo(report.render(verbosity=verbosity_level))
