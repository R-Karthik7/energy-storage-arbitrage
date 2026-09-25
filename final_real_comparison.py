import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# RESULTS FROM REAL-DATA COMPARISON
# ============================================================

agents = [
    "Random",
    "Price-Based",
    "Dueling DQN"
]

average_rewards = [
    2.22,
    3.32,
    2.83
]

std_rewards = [
    0.71,
    0.67,
    0.59
]

final_soc = [
    44.92,
    46.04,
    35.32
]

charge_actions = [
    817,
    616,
    198
]

discharge_actions = [
    809,
    616,
    822
]

idle_actions = [
    774,
    1168,
    1380
]


# ============================================================
# 1. REWARD COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    agents,
    average_rewards,
    yerr=std_rewards,
    capsize=5
)

plt.ylabel("Average Reward (€)")
plt.xlabel("Agent")
plt.title("Real-Data Agent Performance")

plt.tight_layout()

plt.savefig(
    "final_real_reward_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 2. FINAL SOC COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    agents,
    final_soc
)

plt.ylabel("Average Final SOC (%)")
plt.xlabel("Agent")
plt.title("Average Final Battery SOC")

plt.tight_layout()

plt.savefig(
    "final_real_soc_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. ACTION DISTRIBUTION
# ============================================================

x = np.arange(len(agents))

width = 0.25

plt.figure(figsize=(10, 6))

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
    "final_real_action_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

with open(
    "final_real_summary.txt",
    "w"
) as file:

    file.write(
        "ENERGY STORAGE ARBITRAGE\n"
    )

    file.write(
        "FINAL REAL-DATA RESULTS\n"
    )

    file.write(
        "========================\n\n"
    )

    file.write(
        "Test dataset: real_test_prices.csv\n"
    )

    file.write(
        "Episodes: 100\n"
    )

    file.write(
        "Episode length: 24 hours\n\n"
    )

    for i, agent in enumerate(agents):

        file.write(
            f"{agent}\n"
        )

        file.write(
            f"Average Reward: €{average_rewards[i]:.2f}\n"
        )

        file.write(
            f"Standard Deviation: €{std_rewards[i]:.2f}\n"
        )

        file.write(
            f"Average Final SOC: {final_soc[i]:.2f}%\n"
        )

        file.write(
            f"Charge Actions: {charge_actions[i]}\n"
        )

        file.write(
            f"Discharge Actions: {discharge_actions[i]}\n"
        )

        file.write(
            f"Idle Actions: {idle_actions[i]}\n\n"
        )


print("=" * 60)
print("FINAL REAL-DATA GRAPHS CREATED")
print("=" * 60)

print("\nGenerated files:")
print("1. final_real_reward_comparison.png")
print("2. final_real_soc_comparison.png")
print("3. final_real_action_distribution.png")
print("4. final_real_summary.txt")