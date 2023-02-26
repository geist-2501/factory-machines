try:
    import tkinter as tk
except ModuleNotFoundError:
    import matplotlib as mpl
    mpl.use("Agg")

from talos.registration import register_agent, register_wrapper
from talos.cli import talos_app
from talos.core import Agent
