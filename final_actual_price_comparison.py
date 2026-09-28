import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# ACTUAL-PRICE SCENARIO RESULTS
# ============================================================

agents = [
    "Random",
    "Price-Based",
    "Dueling DQN",
    "Double DQN"
]

average_rewards = [
    2.45,
    3.68,
    3.01,
    3.03
]

std_rewards = [
    0.68,
    0.64,
    0.52,
    0.59
]

final_soc = [
    44.92,
    49.82,
    11.69,
    27.24
]

charge_actions = [
    817,
    601,
    20,
    131
]

discharge_actions = [
    809,
    601,
    2248,
    2211
]

idle_actions = [
    774,
    1198,
    132,
    58
]


# ============================================================
# 1. ACTUAL-PRICE REWARD COMPARISON
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
plt.title("Actual-Price Scenario: Agent Performance")

plt.tight_layout()

plt.savefig(
    "actual_price_reward_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 2. ACTUAL-PRICE SOC COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    agents,
    final_soc
)

plt.ylabel("Average Final SOC (%)")
plt.xlabel("Agent")
plt.title("Actual-Price Scenario: Average Final SOC")

plt.tight_layout()

plt.savefig(
    "actual_price_soc_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. ACTUAL-PRICE ACTION DISTRIBUTION
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
plt.title("Actual-Price Scenario: Action Distribution")

plt.legend()

plt.tight_layout()

plt.savefig(
    "actual_price_action_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 4. SAVE SUMMARY
# ============================================================

with open(
    "actual_price_summary.txt",
    "w"
) as file:

    file.write(
        "ACTUAL-PRICE SCENARIO RESULTS\n"
    )

    file.write(
        "=============================\n\n"
    )

    file.write(
        "Dataset: real_test_actual_prices.csv\n"
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
print("ACTUAL-PRICE GRAPHS CREATED")
print("=" * 65)

print("\nGenerated files:")

print(
    "1. actual_price_reward_comparison.png"
)

print(
    "2. actual_price_soc_comparison.png"
)

print(
    "3. actual_price_action_distribution.png"
)

print(
    "4. actual_price_summary.txt"
)