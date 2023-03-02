try:
    import tkinter
except ModuleNotFoundError:
    import matplotlib as mpl
    print("Using headless backend!")
    mpl.use("agg")

from talos.registration import register_agent, register_wrapper
from talos.cli import talos_app
from talos.core import Agent
