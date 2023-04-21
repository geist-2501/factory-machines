from typing import List, Optional

import typer
from gym.wrappers import RecordVideo
from rich import print

from talos.cli.cli_utils import _convert_to_key_value_list
from talos.core import create_env_factory, create_agent, play_agent, evaluate_agents, graph_agent, graph_env_results
from talos.error import TalfileLoadError, AgentNotFound
from talos.file import read_talfile

talfile_app = typer.Typer()


@talfile_app.callback()
def doc():
    """Perform operations on talfiles, like replaying and viewing."""
    pass


@talfile_app.command()
def view(path: str):
    """View the metadata of a talfile"""
    try:
        talfile = read_talfile(path)
    except TalfileLoadError as ex:
        print(f"Couldn't load talfile {path}, " + str(ex))
        raise typer.Abort()

    print(f"[bold white]Agent name[/]:       {talfile.id}")
    print(f"[bold white]Environment used[/]: {talfile.env_name}")
    print(f"[bold white]Wrapper used[/]:     {talfile.used_wrappers}")
    print(f"[bold white]Env args[/]:         {talfile.env_args}")
    print("[bold white]Configuration used[/]:")
    print(talfile.config)


@talfile_app.command()
def replay(
        path: str,
        opt_env: str = typer.Option(
            None,
            "--env",
            "-e"
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
        ),
        opt_target_fps: int = typer.Option(
            30,
            "--fps"
        ),
        opt_wait_for_keypress: bool = typer.Option(
            False,
            "--wait-for-keypress",
            "-k"
        )
):
    """Play the agent in the environment it was trained in."""
    opt_env_args = _convert_to_key_value_list(opt_env_args)

    try:
        talfile = read_talfile(path)
    except TalfileLoadError as ex:
        print(f"Couldn't load talfile {path}, " + str(ex))
        raise typer.Abort()

    if opt_env is None:
        opt_env = talfile.env_name
    else:
        print(f"Overwriting environment specified in talfile ({talfile.env_name}) with {opt_env}")

    if opt_wrapper is None:
        opt_wrapper = talfile.used_wrappers
    else:
        print(f"Overwriting wrapper specified in talfile ({talfile.used_wrappers}) with {opt_wrapper}")

    opt_env_args = {**opt_env_args, **talfile.env_args}

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
            reward_history = play_agent(agent, env, wait_time=1/opt_target_fps, wait_for_keypress=opt_wait_for_keypress)
            print(f"Final score: {sum(reward_history):.2f}")
    except KeyboardInterrupt:
        env.close()
        raise typer.Abort()

    env.close()


@talfile_app.command()
def graph(path: str):
    """Produce graphs of the agent gather during training."""
    try:
        talfile = read_talfile(path)
    except TalfileLoadError as ex:
        print(f"Couldn't load talfile {path}, " + str(ex))
        raise typer.Abort()

    graph_agent(talfile.id, talfile.training_artifacts)


@talfile_app.command()
def compare(
        paths: List[str] = typer.Argument(
            None,
            help="Talfiles of agents to compare against each other."
        ),
        opt_env_args: List[str] = typer.Option(
            [],
            "--env-arg",
        ),
        opt_n_episodes: int = typer.Option(
            3,
            "--num-episodes",
            "-n"
        )
):
    """
    Compare several agents against each other in an environment.
    Agents must have a common environment, but can have different wrappers.
    """

    opt_env_args = _convert_to_key_value_list(opt_env_args)

    loaded_agents = []
    for agent_talfile in paths:
        print(f" > {agent_talfile}... ", end="")
        try:
            # Load talfile.
            talfile = read_talfile(agent_talfile)

            # Recreate the env factory and wrapper.
            env_args = {**talfile.env_args, **opt_env_args}
            agent_env_factory = create_env_factory(talfile.env_name, talfile.used_wrappers, env_args=env_args)
            agent, _ = create_agent(agent_env_factory, talfile.id)
            agent.load(talfile.agent_data)
            loaded_agents.append({
                "agent_name": f"{agent_talfile} ({talfile.id})",
                "agent_id": talfile.id,
                "agent": agent,
                "env_name": talfile.env_name,
                "env_factory": agent_env_factory
            })
            extra_info = f"Uses {talfile.env_name}"
            extra_info += f" with wrapper {talfile.used_wrappers}." if talfile.used_wrappers else "."
            print(f"[bold green]success![/] {extra_info}")
        except RuntimeError as ex:
            print("[bold red]failed![/] Couldn't load .tal file. " + str(ex))
        except AgentNotFound:
            print("[bold red]failed![/] Couldn't find agent definition. Make sure it's been registered.")

    # Check all the environments are the same.
    common_env_id = loaded_agents[0]["env_name"]
    if not all(agent["env_name"] == common_env_id for agent in loaded_agents):
        print("[bold red]Failure![/] Agents don't all use the same environment. Cannot proceed.")
        raise typer.Abort()

    if len(loaded_agents) == len(paths):
        should_continue = typer.confirm("All agents loaded, ready to proceed?", default=True)
    else:
        should_continue = typer.confirm("Only some agents loaded, ready to proceed?", default=False)

    if should_continue:
        rewards, infos = evaluate_agents(loaded_agents, n_episodes=opt_n_episodes)
        graph_env_results(common_env_id, env_args, loaded_agents, rewards, infos)
