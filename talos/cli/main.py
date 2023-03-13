from typing import Optional, List

import typer
from gym.utils.play import play as gym_play
from rich import print

from talos.cli.cli_utils import _convert_to_key_value_list
from talos.cli.config import app as config_app
from talos.cli.list import app as list_app
from talos.cli.talfile import talfile_app
from talos.core import evaluate_agents, load_config, create_env_factory, get_device, create_agent
from talos.error import *
from talos.file import TalFile, read_talfile

app = typer.Typer()
app.add_typer(config_app, name="config")
app.add_typer(list_app, name="list")
app.add_typer(talfile_app, name="talfile")

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
            "-w"
        ),
        opt_env: str = typer.Option(
            "CartPole-v1",
            "--env",
            "-e",
            prompt="Environment to train in?"
        ),
        opt_env_args: List[str] = typer.Option(
            [],
            "--env-arg",
        ),
        opt_autosave: Optional[str] = typer.Option(
            None,
            "--autosave-path"
        )
) -> None:
    """Train an agent on a given environment."""
    opt_env_args = _convert_to_key_value_list(opt_env_args)

    device = get_device()
    print(f"Using device [bold white]{device}.[/]")

    # Load config.
    try:
        print(f"Loading config `{opt_config}`... ", end="")
        config = load_config(opt_config)
        print("[bold green]success![/]")
    except ConfigNotFound:
        print("[bold green]failure![/]")
        raise typer.Abort()

    env_factory = create_env_factory(opt_env, opt_wrapper, env_args=opt_env_args)
    agent, training_wrapper = create_agent(env_factory, opt_agent, device=device)

    agent_config = config[opt_agent] if opt_agent in config.sections() else config['DEFAULT']
    print(f"\nProceeding to train a {opt_agent} on {opt_env} with config values:")
    print(dict(agent_config))

    if typer.confirm("Ready to proceed?", default=True) is False:
        return

    training_artifacts = {}
    try:
        training_wrapper(env_factory, agent, agent_config, training_artifacts)
    except KeyboardInterrupt:
        print("[bold red]Training interrupted[/bold red].")

    if opt_autosave or typer.confirm("Save agent to disk?"):
        try:
            if opt_autosave:
                path = opt_autosave
            else:
                path = typer.prompt("Enter a path to save to")
            print(f"Saving agent to disk ([italic]{path}[/]) ...")
            data = agent.save()
            talfile = TalFile(
                id=opt_agent,
                env_name=opt_env,
                agent_data=data,
                training_artifacts=training_artifacts,
                used_wrappers=opt_wrapper,
                config=dict(agent_config)
            )
            talfile.write(path)
        except OSError as ex:
            print("[bold red]Saving failed![/] " + ex.strerror)


@app.command()
def play(
        arg_env: str = typer.Argument(
            "CartPole-v1",
            help="The environment to play in"
        ),
        opt_wrapper: str = typer.Option(
            None,
            "--wrapper",
            "-w"
        ),
        opt_seed: int = typer.Option(
            None,
            "--seed",
            "-s"
        ),
        opt_env_args: List[str] = typer.Option(
            [],
            "--env-arg",
        )
):
    """Play the environment as a human. (Not for procrastination!)"""
    opt_env_args = _convert_to_key_value_list(opt_env_args)

    env_factory = create_env_factory(arg_env, opt_wrapper, render_mode='rgb_array', env_args=opt_env_args)
    env = env_factory(opt_seed)
    gym_play(env)
