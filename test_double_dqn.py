import torch

from double_dqn import DoubleDQN


# ==========================================
# CONFIGURATION
# ==========================================

state_size = 3
action_size = 3


# ==========================================
# CREATE MODEL
# ==========================================

model = DoubleDQN(
    state_size,
    action_size
)


# ==========================================
# TEST STATE
# ==========================================

test_state = torch.tensor(
    [[
        0.50,
        0.30,
        0.50
    ]],
    dtype=torch.float32
)


# ==========================================
# FORWARD PASS
# ==========================================

q_values = model(test_state)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("=" * 60)
print("DOUBLE DQN ARCHITECTURE TEST")
print("=" * 60)

print("\nInput state:")
print(test_state)

print("\nQ-values:")

print(
    f"IDLE      : {q_values[0][0].item():.4f}"
)

print(
    f"CHARGE    : {q_values[0][1].item():.4f}"
)

print(
    f"DISCHARGE : {q_values[0][2].item():.4f}"
)


best_action = torch.argmax(
    q_values,
    dim=1
).item()


action_names = [
    "IDLE",
    "CHARGE",
    "DISCHARGE"
]

print("\nBest action:")
print(action_names[best_action])


total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nTotal trainable parameters:")
print(total_parameters)

print("\n" + "=" * 60)
print("DOUBLE DQN TEST COMPLETED SUCCESSFULLY")
print("=" * 60)