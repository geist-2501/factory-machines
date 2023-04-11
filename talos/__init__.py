try:
    import tkinter
except ModuleNotFoundError:
    import matplotlib as mpl
    print("Using headless backend!")
    mpl.use("agg")

from talos.registration import register_agent, register_wrapper
from talos.cli.main import talos_app
from talos.agent import Agent, ExtraState
from talos.types import SaveCallback, EnvFactory
