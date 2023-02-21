from typing import Optional

import gym
import torch
import typer
import configparser
from rich import print
from talos import __version__, __app_name__
from talos.agents import resolve_agent
from talos.output import print_err
import talos.config

app = typer.Typer()
app.add_typer(talos.config.app, name="config")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    return


@app.command()
def train(
        opt_agent: str = typer.Option(
            "DQN",
            "--agent",
            "-a",
            prompt="Agent to train with?"
        ),
        opt_config: str = typer.Option(
            "talos_settings.ini",
            "--config",
            "-c",
            prompt="Configuration to use?"
        ),
        opt_env: str = typer.Option(
            "CartPole-v1",
            "--env",
            "-e",
            prompt="Environment to train in?"
        )
) -> None:
    """Train an agent on a given environment."""

    # Load config.
    config = configparser.ConfigParser()
    config.read(opt_config)
    print(f"Loaded config {opt_config}, contains settings for {config.sections()}")

    # Get device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device [bold green]{device}[/bold green]")

    # Load environment.
    def env_factory(seed: int):
        return gym.make(opt_env).unwrapped

    training_wrapper = resolve_agent(opt_agent)

    if training_wrapper is None:
        print_err(f"Couldn't resolve agent {opt_agent}")

    training_wrapper(env_factory, config)
