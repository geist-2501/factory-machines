import os.path
from typing import Optional

import typer

from talos.core import get_device, create_env_factory, create_agent, create_save_callback
from talos.file import TalFile
from talos.profile import read_profile, Profile

profile_app = typer.Typer()


@profile_app.command()
def batch(
        profile_path: str,
        opt_target_profile: Optional[str] = typer.Option(
            None,
            "--target",
            "-t",
            help="Choose a specific profile from the list of profiles."
        ),
        opt_out_dir: str = typer.Option(
            ".",
            "--out",
            "-o",
            help="Directory to place trained agents."
        )
):
    # Load config.
    try:
        print(f"Loading profiles `{profile_path}`... ", end="")
        profiles = read_profile(profile_path)
        print("[bold green]success![/]")
    except RuntimeError:
        print("[bold green]failure![/]")
        raise typer.Abort()

    if opt_target_profile is not None:
        if opt_target_profile not in profiles:
            print(f"Profile {opt_target_profile} doesn't exist in {profile_path}! Choices are;")
            print(profiles.keys())
            raise typer.Abort()

        target_profile = profiles[opt_target_profile]

        _train_with_profile(target_profile, halt=True, out_dir=opt_out_dir)
    else:
        # Train all profiles.
        for _, target_profile in profiles.items():
            _train_with_profile(target_profile, halt=False, out_dir=opt_out_dir)


def _train_with_profile(target_profile: Profile, halt: bool = False, out_dir: str = "."):
    device = get_device()
    print(f"Using device [bold white]{device}.[/]")

    env_factory = create_env_factory(
        target_profile.env_id,
        target_profile.env_wrapper,
        env_args=target_profile.env_args
    )
    agent, training_wrapper = create_agent(env_factory, target_profile.agent_id, device=device)

    print(f"\nProceeding to train a {target_profile.agent_id} on {target_profile.env_id} with config values:")
    print(target_profile.config.to_dict())

    if halt:
        if typer.confirm("Ready to proceed?", default=True) is False:
            return

    training_artifacts = {}
    try:
        save_callback = create_save_callback(
            target_profile.agent_id,
            target_profile.config.to_dict(),
            target_profile.env_wrapper,
            target_profile.env_id,
            target_profile.env_args
        )

        training_wrapper(env_factory, agent, target_profile.config, training_artifacts, save_callback)
    except KeyboardInterrupt:
        print("[bold red]Training interrupted[/bold red].")

    if halt:
        if typer.confirm("Save agent to disk?") is False:
            return

    try:
        path = f"{target_profile.name}.tal"
        path = os.path.join(out_dir, path)
        print(f"Saving agent to disk ([italic]{path}[/]) ...")
        data = agent.save()
        talfile = TalFile(
            id=target_profile.agent_id,
            env_name=target_profile.env_id,
            agent_data=data,
            training_artifacts=training_artifacts,
            used_wrappers=target_profile.env_wrapper,
            config=target_profile.config.to_dict(),
            env_args=target_profile.env_args
        )
        talfile.write(path)
    except OSError as ex:
        print("[bold red]Saving failed![/] " + ex.strerror)