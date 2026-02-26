"""corpusgen inventory — show PHOIBLE phoneme inventory for a language."""

from __future__ import annotations

import json
import sys

import click

from corpusgen import get_inventory


@click.command()
@click.option(
    "--language", "-l",
    required=True,
    help="Language code (espeak-ng voice, ISO 639-3, or Glottocode). Examples: en-us, fr-fr, eng.",
)
@click.option(
    "--source", "-s",
    default=None,
    help="PHOIBLE source filter (e.g., spa, upsid, ph). Default: best available.",
)
@click.option(
    "--format", "-f",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format. Default: text.",
)
def inventory(language: str, source: str | None, output_format: str) -> None:
    """Show the PHOIBLE phoneme inventory for a language."""
    try:
        inv = get_inventory(language, source=source)
    except KeyError as exc:
        click.echo(f"Error: Language not found: {exc}", err=True)
        sys.exit(1)

    if output_format == "json":
        data = {
            "language_name": inv.language_name,
            "iso639_3": inv.iso639_3,
            "glottocode": inv.glottocode,
            "source": inv.source,
            "inventory_id": inv.inventory_id,
            "phonemes": inv.phonemes,
            "consonants": inv.consonants,
            "vowels": inv.vowels,
            "total": len(inv.phonemes),
        }
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        click.echo(f"{inv.language_name} ({inv.iso639_3}, {inv.glottocode})")
        click.echo(f"Source: {inv.source} (inventory {inv.inventory_id})")
        click.echo()
        click.echo(f"Consonants ({len(inv.consonants)}): {' '.join(inv.consonants)}")
        click.echo(f"Vowels ({len(inv.vowels)}): {' '.join(inv.vowels)}")
        click.echo(f"Total: {len(inv.phonemes)} phonemes")
