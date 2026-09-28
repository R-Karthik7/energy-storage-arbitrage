import pandas as pd

from energy_env import EnergyStorageEnv
from double_dqn_agent import DoubleDQNAgent


# ==========================================
# LOAD REAL TRAINING DATA
# ==========================================

df = pd.read_csv(
    "real_train_prices.csv"
)

prices = df[
    "price day ahead"
].values


# ==========================================
# CREATE ENVIRONMENT
# ==========================================

env = EnergyStorageEnv(
    prices[:24]
)


# ==========================================
# CREATE AGENT
# ==========================================

agent = DoubleDQNAgent()


# ==========================================
# RESET
# ==========================================

state, info = env.reset()


print("=" * 60)
print("DOUBLE DQN AGENT TEST")
print("=" * 60)

print(
    "\nTraining prices:",
    len(prices)
)

print(
    "Initial state:",
    state
)

print(
    "Initial epsilon:",
    agent.epsilon
)


# ==========================================
# RUN 100 STEPS
# ==========================================

total_reward = 0.0

losses = []


for step in range(24):

    action = agent.choose_action(
        state
    )

    next_state, reward, terminated, truncated, info = env.step(
        action
    )

    agent.remember(
        state,
        action,
        reward,
        next_state,
        terminated or truncated
    )

    loss = agent.learn()

    if loss is not None:

        losses.append(loss)

    total_reward += reward

    state = next_state

    if terminated or truncated:

        break


# ==========================================
# RESULTS
# ==========================================

print("\n" + "=" * 60)
print("DOUBLE DQN AGENT TEST RESULTS")
print("=" * 60)

print(
    f"\nTotal reward: €{total_reward:.4f}"
)

print(
    "Replay buffer size:",
    len(agent.memory)
)

print(
    "Learning updates:",
    len(losses)
)

if losses:

    print(
        f"Last loss: {losses[-1]:.6f}"
    )

print(
    f"Final epsilon: {agent.epsilon:.4f}"
)

print("\n" + "=" * 60)
print("DOUBLE DQN AGENT TEST COMPLETED")
print("=" * 60)