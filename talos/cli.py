from typing import Optional, List

import gym
import torch
import typer
from rich import print
from talos.registration import get_agent, registry
from talos.config import app as config_app, load_config
from talos.error import *
from talos.agent import play_agent

app = typer.Typer()
app.add_typer(config_app, name="config")

__app_name__ = "talos"
__version__ = "0.1.0"


def talos_app():
    app(prog_name=__app_name__)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(r"""
 _____  _    _     ___  ____  
|_   _|/ \  | |   / _ \/ ___| 
  | | / _ \ | |  | | | \___ \ 
  | |/ ___ \| |__| |_| |___) |
  |_/_/   \_\_____\___/|____/
  RL agent training assistant""")
        print(f"[bold green]  v{__version__}[/]")
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


@app.command("list")
def list_agents():
    """List all registered agents."""
    print("[bold]Currently registered agents[/]:")
    for agent in registry.keys():
        print(agent)


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
        ),
        opt_weights: str = typer.Option(
            None,
            "--weights",
            "-w"
        )
) -> None:
    """Train an agent on a given environment."""

    # Load config.
    config = load_config(opt_config)
    if not config:
        raise typer.Abort()

    env_factory = _create_env_factory(opt_env)
    agent, training_wrapper = _create_agent(env_factory, opt_agent, opt_weights)

    agent_config = config[opt_agent]
    print(f"\nProceeding to train a {opt_agent} on {opt_env} with config values:")
    print(dict(agent_config))

    if typer.confirm("Ready to proceed?", default=True) is False:
        return

    try:
        training_wrapper(env_factory, agent, agent_config)
    except KeyboardInterrupt:
        print("[bold red]Training interrupted[/bold red]")

    if typer.confirm("Save agent to disk?"):
        try:
            path = typer.prompt("Enter a path to save to")
            print(f"Saving agent to disk ([italic]{path}[/]) ...")
            agent.save(path)
        except OSError as ex:
            print("[bold red]Saving failed![/] " + ex.strerror)


@app.command()
def compare(
        opt_agents: List[str] = typer.Option(
            ["DQN.tal"],
            "--agent",
            "-a",
            prompt="Enter the .tal files for the agents you'd like to compare"
        ),
        opt_env: str = typer.Option(
            "CartPole-v1",
            "--env",
            "-e",
            prompt="Environment to evaluate in?"
        ),
):
    for agent in opt_agents:
        print(f" > {agent} [green]loaded![/]")


@app.command()
def play(
        opt_agent: str = typer.Option(
            "DQN",
            "--agent",
            "-a",
            prompt="Agent you want to play?"
        ),
        opt_env: str = typer.Option(
            "CartPole-v1",
            "--env",
            "-e",
            prompt="Environment to play in?"
        ),
        opt_weights: str = typer.Option(
            None,
            "--weights",
            "-w",
            prompt="Path to serialised agent weights?"
        )
):
    env_factory = _create_env_factory(opt_env)
    agent, training_wrapper = _create_agent(env_factory, opt_agent, opt_weights)

    try:
        play_agent(agent, env_factory(0))
    except KeyboardInterrupt:
        raise typer.Abort()


def _get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device [bold green]{device}[/bold green]")
    return device


def _load_weights(agent, opt_weights):
    if opt_weights is not None:
        try:
            print("\nLoading weights...")
            agent.load(opt_weights)
            print("...[bold green]success![/bold green]")
        except OSError as ex:
            print("...[bold red]failed![/bold red] " + ex.strerror)
            typer.confirm("Do you want to continue?", abort=True)


def _create_agent(env_factory, opt_agent, opt_weights):
    try:
        agent_factory, training_wrapper = get_agent(opt_agent)
    except AgentNotFound:
        print(f"[bold red]Couldn't find agent {opt_agent}[/]")
        raise typer.Abort()

    device = _get_device()
    env = env_factory(0)
    agent = agent_factory(
        env.observation_space.shape,
        env.action_space.n,
        device
    )
    _load_weights(agent, opt_weights)
    return agent, training_wrapper


def _create_env_factory(env_name=None):
    def env_factory(seed: int):
        env = gym.make(env_name, render_mode='human').unwrapped
        if seed is not None:
            env.reset(seed=seed)
        return env

    return env_factory
