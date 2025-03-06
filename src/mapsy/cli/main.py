import click, mapsy, os

@click.group()
def main():
    """MapSy CLI - Command line interface for MapSy"""
    pass

@click.command()
def version():
    click.echo(f'MapSy version: {mapsy.__version__}')

@click.command(name='test', options_metavar='[options]', short_help='testing')
@click.argument('filename', metavar='filename', nargs=1)
@click.option('-o', '--output', metavar='', default='dataset', show_default=True)
def test(filename,output):
    """MapSy Test Function"""
    print('Reading from '+filename)
    print('Writing results into '+output)

main.add_command(test)
main.add_command(version)

if __name__ == '__main__':
    main()