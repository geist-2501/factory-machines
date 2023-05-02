import csv
import sys

import numpy as np
from matplotlib import pyplot as plt


def _format_map_name(raw: str) -> str:
    raw = raw.replace("_", " ")
    return raw.capitalize()


def _graph_barh(ax, maps, property_name):
    map_names = [m["name"] for m in maps]
    agent_names = [a["id"] for a in maps[0]["agents"]]
    y_pos = np.arange(len(map_names))
    cmap = plt.get_cmap("tab10")
    colours = [cmap(i) for i in range(len(agent_names))]

    height = 0.15
    multiplier = 0

    for agent_idx, agent_name in enumerate(agent_names):
        data = [m["agents"][agent_idx][property_name] for m in maps]
        offset = height * multiplier
        ax.barh(y_pos + offset, data, height=height, color=colours[agent_idx], label=agent_name)
        multiplier += 1

    ax.set_yticks(y_pos + height, labels=map_names)
    ax.invert_yaxis()  # labels read top-to-bottom


def _parse_csv(row, name: str) -> float:
    name_mapping = [
        "agentId",
        "reward",
        "rewardErr",
        "timesteps",
        "timestepsErr",
        "distance",
        "distanceErr",
        "ordersPerMinute",
        "ordersPerMinuteErr"
    ]

    value = row[name_mapping.index(name)]
    if value == "N/A":
        return 0
    else:
        return float(value)

def main():
    property_name = sys.argv[1]
    csvs = sys.argv[2:]
    maps = []
    for csv_name in csvs:
        map_name = _format_map_name(csv_name.split('.')[-2])
        with open(csv_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            agents = []
            for i, row in enumerate(reader):
                if i == 0:
                    # Skip header.
                    continue
                reward = _parse_csv(row, "reward")
                distance = _parse_csv(row, "distance")
                timesteps = _parse_csv(row, "timesteps")
                orders = _parse_csv(row, "ordersPerMinute")
                agents.append({
                    "id": row[0],
                    "reward": reward,
                    "distance": distance,
                    "timesteps": timesteps,
                    "orders": orders,
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

    fig, (ax_reward, ax_orders) = plt.subplots(2, 1, layout='constrained')

    _graph_barh(ax_reward, maps, "reward")
    ax_reward.set_xlim(0, 280)

    ax_reward.legend(loc="lower right")
    ax_reward.set_title("Episode reward")

    _graph_barh(ax_orders, maps, "orders")

    ax_orders.legend(loc="lower right")
    ax_orders.set_title("Orders per minute")
    ax_orders.set_xlabel("Orders/m")
    ax_orders.set_xlim(0, 10)

    plt.show()


if __name__ == '__main__':
    main()
