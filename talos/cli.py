from typing import Optional, List

import gym
import torch
import typer
from rich import print
from gym.utils.play import play as gym_play
from talos.registration import get_agent, agent_registry, get_wrapper, wrapper_registry
from talos.config import app as config_app, load_config
from talos.error import *
from talos.agent import play_agent
from talos.file import TalFile, read_talfile

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
def list_all():
    """List all registered agents and wrappers."""
    print("[bold]Currently registered agents[/]:")
    for agent in agent_registry.keys():
        print(" " + agent)
    print("\n[bold]Currently registered wrappers[/]:")
    for wrapper in wrapper_registry.keys():
        print(" " + wrapper)
    print("\n[bold]Currently registered environments[/]:")
    for env in gym.envs.registry.keys():
        print(" " + env)


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
        opt_wrapper: str = typer.Option(
            None,
            "--wrapper",
            "-r"
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

    env_factory = _create_env_factory(opt_env, opt_wrapper)
    agent, training_wrapper = _create_agent(env_factory, opt_agent, opt_weights)

    agent_config = config[opt_agent]
    print(f"\nProceeding to train a {opt_agent} on {opt_env} with config values:")
    print(dict(agent_config))

    if typer.confirm("Ready to proceed?", default=True) is False:
        return

    try:
        training_wrapper(env_factory, agent, agent_config)
    except KeyboardInterrupt:
        print("[bold red]Training interrupted[/bold red].")

    if typer.confirm("Save agent to disk?"):
        try:
            path = typer.prompt("Enter a path to save to")
            print(f"Saving agent to disk ([italic]{path}[/]) ...")
            data = agent.save()
            talfile = TalFile(opt_agent, 0, 0, [], data)
            talfile.write(path)
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
        opt_agent_talfile: str = typer.Option(
            "DQN.tal",
            "--talfile",
            "-t",
            prompt="Location of the agent's talfile?"
        ),
        opt_wrapper: str = typer.Option(
            None,
            "--wrapper",
            "-w"
        ),
        opt_env: str = typer.Option(
            "CartPole-v1",
            "--env",
            "-e",
            prompt="Environment to play in?"
        ),
        opt_seed: int = typer.Option(
            None,
            "--seed",
            "-s"
        )
):
    try:
        talfile = read_talfile(opt_agent_talfile)
    except TalfileLoadError as ex:
        print(f"Couldn't load talfile {opt_agent_talfile}")
        raise typer.Abort()

    if opt_agent_talfile == "me":
        env_factory = _create_env_factory(opt_env, opt_wrapper, render_mode='rgb_array')
        env = env_factory(opt_seed)
        gym_play(env)
    else:
        env_factory = _create_env_factory(opt_env, opt_wrapper, render_mode='human')
        agent, _ = _create_agent(env_factory, talfile.id, talfile.agent_data)
        try:
            play_agent(agent, env_factory(opt_seed))
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
    state, _ = env.reset()
    agent = agent_factory(
        len(state),
        env.action_space.n,
        device
    )
    _load_weights(agent, opt_weights)
    return agent, training_wrapper


def _create_env_factory(env_name, wrapper_name=None, render_mode=None):
    def env_factory(seed: int = None):
        env = gym.make(env_name, render_mode=render_mode).unwrapped

        if seed is not None:
            env.reset(seed=seed)

        if wrapper_name is not None:
            wrapper_factory = get_wrapper(wrapper_name)
            env = wrapper_factory(env)

        return env

    return env_factory
