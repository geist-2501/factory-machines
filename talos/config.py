import typer
import configparser
from rich import print

app = typer.Typer()


@app.callback()
def doc():
    """Manage configurations for Talos."""
    pass


@app.command()
def create(
        path: str = typer.Option(
            "talos_settings.ini",
            "--path",
            "-p",
            prompt="Path to create the new config file?"
        )
):
    """Create a new config file populated with sensible defaults."""
    config = configparser.ConfigParser()
    config.read_string("""
    [DEFAULT]
    timesteps_per_epoch = 1
    batch_size = 32
    total_steps = 1 * 10**4
    decay_steps = 5 * 10**3
    learning_rate = 1e-4    
    
    init_epsilon = 1
    final_epsilon = 0.1
    
    loss_freq = 20
    refresh_target_network_freq = 100
    eval_freq = 1000
    
    [dqn]
    total_steps = 4 * 10**4
    decay_steps = 1 * 10**4
    """)

    with open(path, 'w') as configfile:
        config.write(configfile)

    typer.echo(f"Wrote config file '{path}'")


def load_config(config_path: str) -> configparser.ConfigParser | None:
    config = configparser.ConfigParser()
    result = config.read(config_path)
    if not result:
        print(f"Could not load config {config}.")
        return None
    else:
        print(f"Loaded config {config_path}, contains settings for {config.sections()}.")
        return config


