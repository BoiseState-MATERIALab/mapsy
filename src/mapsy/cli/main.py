import click

import mapsy


@click.group()
def main() -> None:
    """MapSy CLI - Command line interface for MapSy"""
    pass


@click.command()
def version() -> None:
    click.echo(f"MapSy version: {mapsy.__version__}")


@click.command(name="test", options_metavar="[options]", short_help="testing")
@click.argument("filename", metavar="filename", nargs=1)
@click.option("-o", "--output", metavar="", default="dataset", show_default=True)
def test(filename: str, output: str) -> None:
    """MapSy Test Function"""
    click.echo(f"Reading from {filename}")
    click.echo(f"Writing results into {output}")


main.add_command(test)
main.add_command(version)

if __name__ == "__main__":
    main()
