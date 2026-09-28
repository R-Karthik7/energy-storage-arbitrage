import random
import numpy as np
import pandas as pd
import torch

from energy_env import EnergyStorageEnv
from dueling_dqn import DuelingDQN
from double_dqn import DoubleDQN


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = "real_test_prices.csv"

DUELING_MODEL = "dueling_dqn_model.pth"
DOUBLE_MODEL = "double_dqn_model.pth"

NUM_EPISODES = 100
EPISODE_LENGTH = 24

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(TEST_FILE)

prices = df[
    "price day ahead"
].values.astype(np.float32)

print("=" * 70)
print("FINAL REAL-DATA RL AGENT COMPARISON")
print("=" * 70)

print(
    f"\nTest records : {len(prices)}"
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
# LOAD DUELING DQN
# ============================================================

dueling_model = DuelingDQN(
    state_size=3,
    action_size=3
)

dueling_model.load_state_dict(
    torch.load(
        DUELING_MODEL,
        map_location="cpu"
    )
)

dueling_model.eval()


# ============================================================
# LOAD DOUBLE DQN
# ============================================================

double_model = DoubleDQN(
    state_size=3,
    action_size=3
)

double_model.load_state_dict(
    torch.load(
        DOUBLE_MODEL,
        map_location="cpu"
    )
)

double_model.eval()


# ============================================================
# FIXED TEST DAYS
# ============================================================

max_start = (
    len(prices)
    - EPISODE_LENGTH
)

test_starts = np.linspace(
    0,
    max_start,
    NUM_EPISODES,
    dtype=int
)


# ============================================================
# RUN ONE EPISODE
# ============================================================

def run_episode(
    daily_prices,
    agent_type
):

    env = EnergyStorageEnv(
        daily_prices
    )

    state, info = env.reset()

    total_reward = 0.0

    charge_count = 0
    discharge_count = 0
    idle_count = 0

    # Daily price thresholds
    low_price = np.percentile(
        daily_prices,
        25
    )

    high_price = np.percentile(
        daily_prices,
        75
    )

    for hour in range(
        EPISODE_LENGTH
    ):

        # ====================================================
        # RANDOM
        # ====================================================

        if agent_type == "random":

            action = random.Random(
                SEED + hour
            ).randint(0, 2)

        # ====================================================
        # PRICE-BASED
        # ====================================================

        elif agent_type == "price":

            current_price = (
                daily_prices[hour]
            )

            if current_price <= low_price:

                action = 1

            elif current_price >= high_price:

                action = 2

            else:

                action = 0

        # ====================================================
        # DUELING DQN
        # ====================================================

        elif agent_type == "dueling":

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():

                q_values = dueling_model(
                    state_tensor
                )

                action = torch.argmax(
                    q_values,
                    dim=1
                ).item()

        # ====================================================
        # DOUBLE DQN
        # ====================================================

        elif agent_type == "double":

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():

                q_values = double_model(
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

        # ====================================================
        # STEP ENVIRONMENT
        # ====================================================

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

    return (
        total_reward,
        env.battery.get_soc(),
        charge_count,
        discharge_count,
        idle_count
    )


# ============================================================
# AGENTS
# ============================================================

agents = [
    "random",
    "price",
    "dueling",
    "double"
]


# ============================================================
# RUN COMPARISON
# ============================================================

results = {}


for agent_type in agents:

    print(
        f"\nRunning {agent_type.upper()}..."
    )

    rewards = []
    final_socs = []

    total_charge = 0
    total_discharge = 0
    total_idle = 0

    # Separate reproducible random stream
    if agent_type == "random":

        random.seed(SEED)

    for start_index in test_starts:

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
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL 4-AGENT COMPARISON")
print("=" * 70)


display_names = {
    "random": "Random Agent",
    "price": "Price-Based Agent",
    "dueling": "Dueling DQN",
    "double": "Double DQN"
}


for agent_type in agents:

    result = results[
        agent_type
    ]

    print(
        f"\n{display_names[agent_type]}"
    )

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
    "final_four_agent_comparison.txt",
    "w"
) as file:

    file.write(
        "FINAL FOUR-AGENT REAL-DATA COMPARISON\n"
    )

    file.write(
        "======================================\n\n"
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

        result = results[
            agent_type
        ]

        file.write(
            f"{display_names[agent_type]}\n"
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
print(
    "final_four_agent_comparison.txt"
)

print("\n" + "=" * 70)
print("FINAL COMPARISON COMPLETED")
print("=" * 70)