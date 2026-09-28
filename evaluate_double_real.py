import random
import numpy as np
import pandas as pd
import torch

from energy_env import EnergyStorageEnv
from double_dqn import DoubleDQN


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = "real_test_prices.csv"
MODEL_FILE = "double_dqn_model.pth"

NUM_TEST_EPISODES = 100
EPISODE_LENGTH = 24

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 65)
print("DOUBLE DQN - REAL DATA EVALUATION")
print("=" * 65)

df = pd.read_csv(TEST_FILE)

prices = df[
    "price day ahead"
].values.astype(np.float32)

print(
    f"\nTest records: {len(prices)}"
)

print(
    f"Minimum price: €{prices.min():.2f}/MWh"
)

print(
    f"Maximum price: €{prices.max():.2f}/MWh"
)

print(
    f"Average price: €{prices.mean():.2f}/MWh"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = DoubleDQN(
    state_size=3,
    action_size=3
)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location="cpu"
    )
)

model.eval()


# ============================================================
# EVALUATION STORAGE
# ============================================================

rewards = []
final_socs = []

charge_count = 0
discharge_count = 0
idle_count = 0


# ============================================================
# SELECT FIXED TEST DAYS
# ============================================================

max_start = (
    len(prices)
    - EPISODE_LENGTH
)

test_starts = np.linspace(
    0,
    max_start,
    NUM_TEST_EPISODES,
    dtype=int
)


# ============================================================
# START EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("STARTING DOUBLE DQN EVALUATION")
print("=" * 65)


for episode, start_index in enumerate(
    test_starts,
    start=1
):

    end_index = (
        start_index
        + EPISODE_LENGTH
    )

    daily_prices = prices[
        start_index:end_index
    ]

    env = EnergyStorageEnv(
        daily_prices
    )

    state, info = env.reset()

    total_reward = 0.0

    for step in range(
        EPISODE_LENGTH
    ):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = model(
                state_tensor
            )

            action = torch.argmax(
                q_values,
                dim=1
            ).item()

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        if action == 0:
            idle_count += 1

        elif action == 1:
            charge_count += 1

        elif action == 2:
            discharge_count += 1

        state = next_state

        if terminated or truncated:
            break

    rewards.append(
        total_reward
    )

    final_socs.append(
        env.battery.get_soc()
    )

    if episode % 20 == 0:

        print(
            f"Episode {episode:3d} | "
            f"Reward: €{total_reward:7.3f}"
        )


# ============================================================
# CALCULATE RESULTS
# ============================================================

average_reward = np.mean(
    rewards
)

std_reward = np.std(
    rewards
)

minimum_reward = np.min(
    rewards
)

maximum_reward = np.max(
    rewards
)

average_final_soc = np.mean(
    final_socs
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("DOUBLE DQN TEST RESULTS")
print("=" * 65)

print(
    f"\nAverage Reward       : "
    f"€{average_reward:.2f}"
)

print(
    f"Standard Deviation   : "
    f"€{std_reward:.2f}"
)

print(
    f"Minimum Reward       : "
    f"€{minimum_reward:.2f}"
)

print(
    f"Maximum Reward       : "
    f"€{maximum_reward:.2f}"
)

print(
    f"Average Final SOC    : "
    f"{average_final_soc * 100:.2f}%"
)

print("\nACTION DISTRIBUTION")

print(
    f"CHARGE     : {charge_count}"
)

print(
    f"DISCHARGE  : {discharge_count}"
)

print(
    f"IDLE       : {idle_count}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "double_dqn_real_results.txt",
    "w"
) as file:

    file.write(
        "DOUBLE DQN REAL DATA EVALUATION\n"
    )

    file.write(
        "================================\n\n"
    )

    file.write(
        f"Test records: {len(prices)}\n"
    )

    file.write(
        f"Test episodes: {NUM_TEST_EPISODES}\n"
    )

    file.write(
        "Episode length: 24 hours\n\n"
    )

    file.write(
        f"Average Reward: €{average_reward:.2f}\n"
    )

    file.write(
        f"Standard Deviation: €{std_reward:.2f}\n"
    )

    file.write(
        f"Minimum Reward: €{minimum_reward:.2f}\n"
    )

    file.write(
        f"Maximum Reward: €{maximum_reward:.2f}\n"
    )

    file.write(
        f"Average Final SOC: "
        f"{average_final_soc * 100:.2f}%\n\n"
    )

    file.write(
        f"Charge Actions: {charge_count}\n"
    )

    file.write(
        f"Discharge Actions: {discharge_count}\n"
    )

    file.write(
        f"Idle Actions: {idle_count}\n"
    )


print("\nResults saved as:")
print("double_dqn_real_results.txt")

print("\n" + "=" * 65)
print("DOUBLE DQN EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 65)