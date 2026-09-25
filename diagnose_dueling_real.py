import pandas as pd
import torch

from energy_env import EnergyStorageEnv
from dueling_dqn import DuelingDQN


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = "real_test_prices.csv"
MODEL_FILE = "dueling_dqn_model.pth"

EPISODES_TO_CHECK = 5
EPISODE_LENGTH = 24


# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(TEST_FILE)

prices = df["price day ahead"].values


# ============================================================
# LOAD TRAINED DUELING DQN
# ============================================================

model = DuelingDQN(
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


action_names = [
    "IDLE",
    "CHARGE",
    "DISCHARGE"
]


# ============================================================
# DIAGNOSIS
# ============================================================

print("=" * 70)
print("DUELING DQN REAL-DATA POLICY DIAGNOSIS")
print("=" * 70)


for episode in range(EPISODES_TO_CHECK):

    # Use consecutive test days
    start = episode * EPISODE_LENGTH
    end = start + EPISODE_LENGTH

    daily_prices = prices[start:end]

    env = EnergyStorageEnv(
        daily_prices
    )

    state, info = env.reset()

    total_reward = 0.0

    charge_count = 0
    discharge_count = 0
    idle_count = 0

    print("\n" + "=" * 70)
    print(f"TEST DAY {episode + 1}")
    print("=" * 70)

    for hour in range(EPISODE_LENGTH):

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

        print(
            f"Hour {hour:2d} | "
            f"Price: €{info['price']:6.2f} | "
            f"Action: {action_names[action]:9s} | "
            f"Reward: €{reward:7.3f} | "
            f"SOC: {info['soc'] * 100:5.1f}% | "
            f"Q: {q_values[0].numpy()}"
        )

        state = next_state

        if terminated or truncated:
            break

    print("\nDAY SUMMARY")

    print(
        f"Total reward: €{total_reward:.3f}"
    )

    print(
        f"Charge: {charge_count}"
    )

    print(
        f"Discharge: {discharge_count}"
    )

    print(
        f"Idle: {idle_count}"
    )

    print(
        f"Final SOC: {info['soc'] * 100:.2f}%"
    )


print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETED")
print("=" * 70)