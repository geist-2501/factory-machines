import csv
import sys

import numpy as np
from matplotlib import pyplot as plt


def _format_map_name(raw: str) -> str:
    raw = raw.replace("_", " ")
    return raw.capitalize()


def _graph_barh(ax, maps, property_name):
    map_names = [m["name"] for m in maps]
    show_labels = len(map_names) > 1
    agent_names = [a["id"] for a in maps[0]["agents"]]
    y_pos = np.arange(len(map_names))
    cmap = plt.get_cmap("tab10")
    colours = [cmap(i) for i in range(len(agent_names))]

    height = 0.16
    multiplier = 0

    for agent_idx, agent_name in enumerate(agent_names):
        data = [m["agents"][agent_idx][property_name] for m in maps]
        offset = height * multiplier
        ax.barh(y_pos + offset, data, height=height, color=colours[agent_idx], label=agent_name)
        multiplier += 1

    if show_labels:
        ax.set_yticks(y_pos + height, labels=map_names)
        ax.legend(loc="lower right")

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
    csvs = sys.argv[1:]
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
                agents.append({
                    "id": row[0],
                    "reward": _parse_csv(row, "reward"),
                    "rewardErr": _parse_csv(row, "rewardErr"),
                    "distance": _parse_csv(row, "distance"),
                    "distanceErr": _parse_csv(row, "distanceErr"),
                    "timesteps": _parse_csv(row, "timesteps"),
                    "timestepsErr": _parse_csv(row, "timestepsErr"),
                    "ordersPerMinute": _parse_csv(row, "ordersPerMinute"),
                    "ordersPerMinuteErr": _parse_csv(row, "ordersPerMinuteErr"),
                })
            maps.append({
                "name": map_name,
                "agents": agents
            })

    fig, (ax_reward, ax_orders) = plt.subplots(2, 1)

    _graph_barh(ax_reward, maps, "reward")
    ax_reward.set_xlim(0, 300)

    ax_reward.set_title("Average episode reward")

    _graph_barh(ax_orders, maps, "ordersPerMinute")

    ax_orders.set_title("Average orders completed per minute")
    ax_orders.set_xlabel("Orders/minute")
    ax_orders.set_xlim(0, 15)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
