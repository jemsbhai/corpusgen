"""Command-line interface for corpusgen."""

import click

from corpusgen.cli.evaluate import evaluate_cmd
from corpusgen.cli.generate import generate_cmd
from corpusgen.cli.inventory import inventory
from corpusgen.cli.select import select_cmd


@click.group()
@click.version_option()
def main() -> None:
    """corpusgen: Generate and evaluate speech corpora with maximal phoneme coverage."""


main.add_command(inventory)
main.add_command(evaluate_cmd, name="evaluate")
main.add_command(select_cmd, name="select")
main.add_command(generate_cmd, name="generate")
