import csv
import sys
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


def _parse_score(raw: str) -> Tuple[float, float]:
    if raw == "N/A":
        return 0, 0
    score, error = raw.split(" ")
    score = float(score)
    error = float(error[1:-1])
    return score, error


if __name__ == '__main__':
    csvs = sys.argv[1:]
    maps = []
    for csv_name in csvs:
        map_name = csv_name.split('.')[0]
        with open(csv_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            agents = []
            for i, row in enumerate(reader):
                if i == 0:
                    # Skip header.
                    continue
                reward, reward_err = _parse_score(row[1])
                timesteps, timesteps_err = _parse_score(row[2])
                distance, distance_err = _parse_score(row[3])
                orders, orders_err = _parse_score(row[4])
                agents.append({
                    "id": row[0],
                    "reward": reward,
                    "reward_err": reward_err,
                    "distance": distance,
                    "distance_err": distance_err,
                    "timesteps": timesteps,
                    "timesteps_err": timesteps_err,
                    "orders": orders,
                    "orders_err": orders_err
                })
            maps.append({
                "name": map_name,
                "agents": agents
            })

    map_names = [m["name"] for m in maps]
    agent_names = [a["id"] for a in maps[0]["agents"]]
    y_pos = np.arange(len(map_names))
    cmap = plt.get_cmap("tab10")
    colours = [cmap(i) for i in range(len(agent_names))]
    agent_styles = {
        "FM-AisledNN": {"height": 0.9, "color": colours[0]},
        "FM-Highest": {"height": 0.2, "color": None, "edgecolor": colours[1], "linestyle": '--'},
        "DQN": {"height": 0.4, "color": colours[1]},
        "FM-HDQN": {"height": 0.4, "color": colours[3]},
    }

    fig, ax = plt.subplots()

    for agent_idx, agent_name in enumerate(agent_names):
        style = agent_styles[agent_name]
        data = [m["agents"][agent_idx]["reward"] for m in maps]
        ax.barh(y_pos, data, **style)

    ax.set_yticks(y_pos, labels=map_names)
    ax.invert_yaxis()  # labels read top-to-bottom

    plt.show()
