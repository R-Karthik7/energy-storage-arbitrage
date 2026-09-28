import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# FINAL FOUR-AGENT RESULTS
# ============================================================

agents = [
    "Random",
    "Price-Based",
    "Dueling DQN",
    "Double DQN"
]

average_rewards = [
    2.19,
    3.32,
    2.78,
    2.83
]

std_rewards = [
    0.82,
    0.67,
    0.64,
    0.60
]

final_soc = [
    90.00,
    46.04,
    13.24,
    38.48
]

charge_actions = [
    1000,
    616,
    57,
    248
]

discharge_actions = [
    600,
    616,
    2151,
    2061
]

idle_actions = [
    800,
    1168,
    192,
    91
]


# ============================================================
# 1. REWARD COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    agents,
    average_rewards,
    yerr=std_rewards,
    capsize=5
)

plt.ylabel("Average Reward (€)")
plt.xlabel("Agent")
plt.title("Four-Agent Comparison on Real Test Data")

plt.tight_layout()

plt.savefig(
    "final_four_agent_reward_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 2. FINAL SOC COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    agents,
    final_soc
)

plt.ylabel("Average Final SOC (%)")
plt.xlabel("Agent")
plt.title("Average Final Battery SOC")

plt.tight_layout()

plt.savefig(
    "final_four_agent_soc_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. ACTION DISTRIBUTION
# ============================================================

x = np.arange(
    len(agents)
)

width = 0.25

plt.figure(figsize=(11, 6))

plt.bar(
    x - width,
    charge_actions,
    width,
    label="Charge"
)

plt.bar(
    x,
    discharge_actions,
    width,
    label="Discharge"
)

plt.bar(
    x + width,
    idle_actions,
    width,
    label="Idle"
)

plt.xticks(
    x,
    agents
)

plt.ylabel("Number of Actions")
plt.xlabel("Agent")
plt.title("Action Distribution on Real Test Data")

plt.legend()

plt.tight_layout()

plt.savefig(
    "final_four_agent_action_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 4. SAVE SUMMARY
# ============================================================

with open(
    "final_four_agent_summary.txt",
    "w"
) as file:

    file.write(
        "FINAL FOUR-AGENT REAL-DATA COMPARISON\n"
    )

    file.write(
        "=====================================\n\n"
    )

    file.write(
        "Test Dataset: real_test_prices.csv\n"
    )

    file.write(
        "Episodes: 100\n"
    )

    file.write(
        "Episode Length: 24 hours\n\n"
    )

    for i, agent in enumerate(agents):

        file.write(
            f"{agent}\n"
        )

        file.write(
            f"Average Reward: "
            f"€{average_rewards[i]:.2f}\n"
        )

        file.write(
            f"Standard Deviation: "
            f"€{std_rewards[i]:.2f}\n"
        )

        file.write(
            f"Average Final SOC: "
            f"{final_soc[i]:.2f}%\n"
        )

        file.write(
            f"Charge Actions: "
            f"{charge_actions[i]}\n"
        )

        file.write(
            f"Discharge Actions: "
            f"{discharge_actions[i]}\n"
        )

        file.write(
            f"Idle Actions: "
            f"{idle_actions[i]}\n\n"
        )


print("=" * 65)
print("FINAL FOUR-AGENT GRAPHS CREATED")
print("=" * 65)

print("\nGenerated files:")

print(
    "1. final_four_agent_reward_comparison.png"
)

print(
    "2. final_four_agent_soc_comparison.png"
)

print(
    "3. final_four_agent_action_distribution.png"
)

print(
    "4. final_four_agent_summary.txt"
)