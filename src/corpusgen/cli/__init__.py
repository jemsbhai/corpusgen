"""Command-line interface for corpusgen."""

import click


@click.group()
@click.version_option()
def main() -> None:
    """corpusgen: Generate and evaluate speech corpora with maximal phoneme coverage."""


# Subcommands will be added in later phases:
# - corpusgen evaluate <text_or_file> --language <lang>
# - corpusgen generate --language <lang> --strategy <strategy>
# - corpusgen inventory --language <lang>
