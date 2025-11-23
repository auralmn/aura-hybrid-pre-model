import click

@click.command()
@click.option('--new', default=False, help='Create a new brain system')
@click.option('--name', default='AURA',prompt='The entity name',
              help='The name of the neuromorphic brain system')
def start(new, name):
    """Start the brain system"""
    if new:
        create_new_brain(name)
    else:
        load_brain(name)

if __name__ == '__main__':
    start()