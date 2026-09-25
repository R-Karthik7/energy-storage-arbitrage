import random
import numpy as np
import pandas as pd
import torch

from energy_env import EnergyStorageEnv
from dueling_dqn import DuelingDQN


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = "real_test_prices.csv"
MODEL_FILE = "dueling_dqn_model.pth"

NUM_EPISODES = 100
EPISODE_LENGTH = 24

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# LOAD REAL TEST DATA
# ============================================================

print("=" * 70)
print("REAL-DATA AGENT COMPARISON")
print("=" * 70)

df = pd.read_csv(TEST_FILE)

prices = df["price day ahead"].values.astype(np.float32)

print(f"\nTest records : {len(prices)}")
print(f"Minimum price: €{prices.min():.2f}/MWh")
print(f"Maximum price: €{prices.max():.2f}/MWh")
print(f"Average price: €{prices.mean():.2f}/MWh")


# ============================================================
# LOAD DUELING DQN
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


# ============================================================
# SELECT TEST DAYS
# ============================================================

max_start = len(prices) - EPISODE_LENGTH

test_starts = np.linspace(
    0,
    max_start,
    NUM_EPISODES,
    dtype=int
)


# ============================================================
# RUN ONE EPISODE
# ============================================================

def run_episode(prices_for_day, agent_type):

    env = EnergyStorageEnv(
        prices_for_day
    )

    state, info = env.reset()

    total_reward = 0.0

    charge_count = 0
    discharge_count = 0
    idle_count = 0

    # Daily price thresholds
    low_price = np.percentile(
        prices_for_day,
        25
    )

    high_price = np.percentile(
        prices_for_day,
        75
    )

    for hour in range(EPISODE_LENGTH):

        # ----------------------------------------------------
        # RANDOM AGENT
        # ----------------------------------------------------

        if agent_type == "random":

            action = random.randint(
                0,
                2
            )

        # ----------------------------------------------------
        # PRICE-BASED AGENT
        # ----------------------------------------------------

        elif agent_type == "price":

            current_price = prices_for_day[hour]

            if current_price <= low_price:

                action = 1       # CHARGE

            elif current_price >= high_price:

                action = 2       # DISCHARGE

            else:

                action = 0       # IDLE

        # ----------------------------------------------------
        # DUELING DQN
        # ----------------------------------------------------

        elif agent_type == "dueling":

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

        else:

            raise ValueError(
                "Unknown agent type"
            )

        # ----------------------------------------------------
        # ENVIRONMENT STEP
        # ----------------------------------------------------

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        # ----------------------------------------------------
        # ACTION COUNTS
        # ----------------------------------------------------

        if action == 0:

            idle_count += 1

        elif action == 1:

            charge_count += 1

        elif action == 2:

            discharge_count += 1

        state = next_state

        if terminated or truncated:

            break

    final_soc = env.battery.get_soc()

    return (
        total_reward,
        final_soc,
        charge_count,
        discharge_count,
        idle_count
    )


# ============================================================
# RUN ALL AGENTS
# ============================================================

agents = [
    "random",
    "price",
    "dueling"
]

results = {}


for agent_type in agents:

    rewards = []
    final_socs = []

    total_charge = 0
    total_discharge = 0
    total_idle = 0

    print(
        f"\nRunning {agent_type.upper()} agent..."
    )

    for episode_number, start_index in enumerate(
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

        (
            reward,
            final_soc,
            charge,
            discharge,
            idle
        ) = run_episode(
            daily_prices,
            agent_type
        )

        rewards.append(
            reward
        )

        final_socs.append(
            final_soc
        )

        total_charge += charge
        total_discharge += discharge
        total_idle += idle

    results[agent_type] = {

        "average_reward":
            np.mean(rewards),

        "std_reward":
            np.std(rewards),

        "minimum_reward":
            np.min(rewards),

        "maximum_reward":
            np.max(rewards),

        "average_final_soc":
            np.mean(final_socs),

        "charge":
            total_charge,

        "discharge":
            total_discharge,

        "idle":
            total_idle
    }


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL REAL-DATA COMPARISON")
print("=" * 70)


for agent_type in agents:

    result = results[agent_type]

    if agent_type == "random":

        name = "Random Agent"

    elif agent_type == "price":

        name = "Price-Based Agent"

    else:

        name = "Dueling DQN"

    print(f"\n{name}")
    print("-" * 70)

    print(
        f"Average Reward     : "
        f"€{result['average_reward']:.2f}"
    )

    print(
        f"Standard Deviation : "
        f"€{result['std_reward']:.2f}"
    )

    print(
        f"Minimum Reward     : "
        f"€{result['minimum_reward']:.2f}"
    )

    print(
        f"Maximum Reward     : "
        f"€{result['maximum_reward']:.2f}"
    )

    print(
        f"Average Final SOC  : "
        f"{result['average_final_soc'] * 100:.2f}%"
    )

    print(
        f"Charge Actions     : "
        f"{result['charge']}"
    )

    print(
        f"Discharge Actions  : "
        f"{result['discharge']}"
    )

    print(
        f"Idle Actions       : "
        f"{result['idle']}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "real_agent_comparison.txt",
    "w"
) as file:

    file.write(
        "REAL-DATA AGENT COMPARISON\n"
    )

    file.write(
        "==========================\n\n"
    )

    file.write(
        f"Test records: {len(prices)}\n"
    )

    file.write(
        f"Test episodes: {NUM_EPISODES}\n"
    )

    file.write(
        "Episode length: 24 hours\n\n"
    )

    for agent_type in agents:

        result = results[agent_type]

        if agent_type == "random":

            name = "Random Agent"

        elif agent_type == "price":

            name = "Price-Based Agent"

        else:

            name = "Dueling DQN"

        file.write(
            f"{name}\n"
        )

        file.write(
            f"Average Reward: "
            f"€{result['average_reward']:.2f}\n"
        )

        file.write(
            f"Standard Deviation: "
            f"€{result['std_reward']:.2f}\n"
        )

        file.write(
            f"Minimum Reward: "
            f"€{result['minimum_reward']:.2f}\n"
        )

        file.write(
            f"Maximum Reward: "
            f"€{result['maximum_reward']:.2f}\n"
        )

        file.write(
            f"Average Final SOC: "
            f"{result['average_final_soc'] * 100:.2f}%\n"
        )

        file.write(
            f"Charge Actions: "
            f"{result['charge']}\n"
        )

        file.write(
            f"Discharge Actions: "
            f"{result['discharge']}\n"
        )

        file.write(
            f"Idle Actions: "
            f"{result['idle']}\n\n"
        )


print("\nResults saved as:")
print("real_agent_comparison.txt")

print("\n" + "=" * 70)
print("COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 70)