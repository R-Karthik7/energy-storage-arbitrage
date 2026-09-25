import random
import numpy as np
import pandas as pd
import torch

from energy_env import EnergyStorageEnv
from dueling_dqn_agent import DuelingDQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "real_train_prices.csv"

NUM_EPISODES = 4000

EPISODE_LENGTH = 24

TARGET_UPDATE_FREQUENCY = 20

MODEL_FILE = "dueling_dqn_model.pth"

REWARD_LOG_FILE = "dueling_dqn_training_rewards.txt"


# ============================================================
# LOAD REAL TRAINING DATA
# ============================================================

print("=" * 65)
print("DUELING DQN - REAL DATA TRAINING")
print("=" * 65)

df = pd.read_csv(TRAIN_FILE)

prices = df["price day ahead"].values.astype(np.float32)

print("\nReal training records:", len(prices))

print(
    "Minimum price: €"
    f"{prices.min():.2f}"
)

print(
    "Maximum price: €"
    f"{prices.max():.2f}"
)

print(
    "Average price: €"
    f"{prices.mean():.2f}"
)


# ============================================================
# CREATE AGENT
# ============================================================

agent = DuelingDQNAgent()


# ============================================================
# TRAINING STORAGE
# ============================================================

episode_rewards = []

episode_losses = []


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 65)
print("STARTING TRAINING")
print("=" * 65)

for episode in range(1, NUM_EPISODES + 1):

    # --------------------------------------------------------
    # Select a random 24-hour period
    # --------------------------------------------------------

    max_start = len(prices) - EPISODE_LENGTH

    start_index = random.randint(
        0,
        max_start
    )

    end_index = start_index + EPISODE_LENGTH

    daily_prices = prices[
        start_index:end_index
    ]

    # --------------------------------------------------------
    # Create environment for this 24-hour period
    # --------------------------------------------------------

    env = EnergyStorageEnv(
        daily_prices
    )

    state, info = env.reset()

    total_reward = 0.0

    losses = []

    # --------------------------------------------------------
    # Run one episode
    # --------------------------------------------------------

    for step in range(EPISODE_LENGTH):

        # Choose action
        action = agent.choose_action(state)

        # Environment step
        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated

        # Store experience
        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Learn
        loss = agent.learn()

        if loss is not None:
            losses.append(loss)

        total_reward += reward

        state = next_state

        if done:
            break

    # --------------------------------------------------------
    # Decay exploration
    # --------------------------------------------------------

    agent.decay_epsilon()

    # --------------------------------------------------------
    # Update target network
    # --------------------------------------------------------

    if episode % TARGET_UPDATE_FREQUENCY == 0:

        agent.update_target_network()

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    episode_rewards.append(
        total_reward
    )

    if losses:

        episode_losses.append(
            np.mean(losses)
        )

    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if episode % 100 == 0:

        recent_rewards = episode_rewards[-100:]

        average_reward = np.mean(
            recent_rewards
        )

        if episode_losses:

            average_loss = np.mean(
                episode_losses[-100:]
            )

        else:

            average_loss = 0.0

        print(
            f"Episode {episode:4d} | "
            f"Average Reward: €{average_reward:8.2f} | "
            f"Loss: {average_loss:10.4f} | "
            f"Epsilon: {agent.epsilon:.4f}"
        )


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    agent.q_network.state_dict(),
    MODEL_FILE
)


# ============================================================
# SAVE TRAINING REWARDS
# ============================================================

with open(
    REWARD_LOG_FILE,
    "w"
) as file:

    file.write(
        "Episode,Reward\n"
    )

    for episode_number, reward in enumerate(
        episode_rewards,
        start=1
    ):

        file.write(
            f"{episode_number},{reward}\n"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

final_rewards = episode_rewards[-100:]

final_average = np.mean(
    final_rewards
)

final_std = np.std(
    final_rewards
)


print("\n" + "=" * 65)
print("DUELING DQN TRAINING COMPLETED")
print("=" * 65)

print(
    f"\nFinal 100-episode average reward: "
    f"€{final_average:.2f}"
)

print(
    f"Final 100-episode standard deviation: "
    f"€{final_std:.2f}"
)

print(
    f"Final epsilon: "
    f"{agent.epsilon:.4f}"
)

print(
    f"\nModel saved as:"
    f"\n{MODEL_FILE}"
)

print(
    f"\nTraining rewards saved as:"
    f"\n{REWARD_LOG_FILE}"
)

print("\nTraining completed successfully!")