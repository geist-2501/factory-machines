from typing import Optional, List, Dict

import typer
from rich import print
from gym.utils.play import play as gym_play
from gym.wrappers import RecordVideo
from talos.cli.config import app as config_app
from talos.cli.list import app as list_app
from talos.error import *
from talos.core import play_agent, evaluate_agents, load_config, create_env_factory, get_device, create_agent
from talos.file import TalFile, read_talfile

app = typer.Typer()
app.add_typer(config_app, name="config")
app.add_typer(list_app, name="list")

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


def _convert_to_key_value_list(args: List[str]) -> Dict[str, str]:

    key_values = {}
    for arg in args:
        parts = arg.split('=')
        assert len(parts) == 2
        key, value = parts
        key_values[key] = value

    return key_values


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
                used_wrappers=opt_wrapper
            )
            talfile.write(path)
        except OSError as ex:
            print("[bold red]Saving failed![/] " + ex.strerror)


@app.command()
def compare(
        opt_agent_talfiles: List[str] = typer.Option(
            ["DQN.tal"],
            "--talfile",
            "-t",
            prompt="Enter the .tal files for the agents you'd like to compare"
        ),
        opt_env: str = typer.Option(
            "CartPole-v1",
            "--env",
            "-e",
            prompt="Environment to evaluate in?"
        ),
        opt_wrapper: str = typer.Option(
            None,
            "--wrapper",
            "-w"
        ),
):
    env_factory = create_env_factory(opt_env, opt_wrapper)

    agents = []
    for agent_talfile in opt_agent_talfiles:
        print(f" > {agent_talfile}... ", end="")
        try:
            talfile = read_talfile(agent_talfile)
            agent, _ = create_agent(env_factory, talfile.id)
            agent.load(talfile.agent_data)
            agents.append(agent)
            print("[bold green]success![/]")
        except TalfileLoadError:
            print("[bold red]failed![/] Couldn't load .tal file.")
        except AgentNotFound:
            print("[bold red]failed![/] Couldn't find agent definition. Make sure it's been registered.")

    if len(agents) == len(opt_agent_talfiles):
        should_continue = typer.confirm("All agents loaded, ready to proceed?", default=True)
    else:
        should_continue = typer.confirm("Only some agents loaded, ready to proceed?", default=False)

    if should_continue:
        evaluate_agents(agents, opt_agent_talfiles, env_factory)


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
            None,
            "--env",
            "-e",
        ),
        opt_seed: int = typer.Option(
            None,
            "--seed",
            "-s"
        ),
        opt_env_args: List[str] = typer.Option(
            [],
            "--env-arg",
        ),
        opt_save_to: Optional[str] = typer.Option(
            None,
            "--save-to",
            "-s"
        ),
        opt_num_eps: int = typer.Option(
            1,
            "--num-eps",
            "-n"
        ),
        opt_render: str = typer.Option(
            "human",
            "--render-as"
        )
):
    opt_env_args = _convert_to_key_value_list(opt_env_args)

    if opt_agent_talfile == "me":
        if opt_env is None:
            opt_env = typer.prompt("Environment to play in?", default="CartPole-v1")
        env_factory = create_env_factory(opt_env, opt_wrapper, render_mode='rgb_array', env_args=opt_env_args)
        env = env_factory(opt_seed)
        gym_play(env)
    else:
        try:
            talfile = read_talfile(opt_agent_talfile)
        except TalfileLoadError as ex:
            print(f"Couldn't load talfile {opt_agent_talfile}, " + str(ex))
            raise typer.Abort()

        env_factory = create_env_factory(opt_env, opt_wrapper, render_mode=opt_render, env_args=opt_env_args)
        agent, _ = create_agent(env_factory, talfile.id)
        agent.load(talfile.agent_data)

        env = env_factory(opt_seed)

        if opt_save_to:
            file_prefix = f"{agent.name}-{opt_env}"
            print(f"Recording to {opt_save_to}/{file_prefix}")
            env = RecordVideo(
                env=env,
                video_folder=opt_save_to,
                episode_trigger=lambda ep_num: True,
                name_prefix=file_prefix
            )

        try:
            for _ in range(opt_num_eps):
                play_agent(agent, env, wait_time=0.5)
        except KeyboardInterrupt:
            env.close()
            raise typer.Abort()

        env.close()
