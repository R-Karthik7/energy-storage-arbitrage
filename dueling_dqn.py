import torch
import torch.nn as nn


class DuelingDQN(nn.Module):

    def __init__(self, state_size, action_size):

        super(DuelingDQN, self).__init__()

        # ==========================================
        # SHARED FEATURE NETWORK
        # ==========================================

        self.feature_layer = nn.Sequential(

            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU()
        )

        # ==========================================
        # VALUE STREAM
        # ==========================================

        # Estimates the value of the current state V(s)

        self.value_stream = nn.Sequential(

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

        # ==========================================
        # ADVANTAGE STREAM
        # ==========================================

        # Estimates the advantage of each action A(s,a)

        self.advantage_stream = nn.Sequential(

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)
        )

    def forward(self, state):

        # ==========================================
        # SHARED FEATURES
        # ==========================================

        features = self.feature_layer(state)

        # ==========================================
        # VALUE
        # ==========================================

        value = self.value_stream(features)

        # ==========================================
        # ADVANTAGE
        # ==========================================

        advantage = self.advantage_stream(features)

        # ==========================================
        # COMBINE VALUE + ADVANTAGE
        # ==========================================

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

        q_values = (
            value
            + advantage
            - advantage.mean(dim=1, keepdim=True)
        )

        return q_values