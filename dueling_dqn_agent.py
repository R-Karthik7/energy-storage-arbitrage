import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from replay_buffer import ReplayBuffer
from dueling_dqn import DuelingDQN


class DuelingDQNAgent:

    def __init__(self):

        # ==========================================
        # ENVIRONMENT CONFIGURATION
        # ==========================================

        self.state_size = 3
        self.action_size = 3

        # ==========================================
        # RL PARAMETERS
        # ==========================================

        self.gamma = 0.99

        self.batch_size = 64

        # ==========================================
        # REPLAY BUFFER
        # ==========================================

        self.memory = ReplayBuffer(
            capacity=50000
        )

        # ==========================================
        # DUELING DQN NETWORK
        # ==========================================

        self.q_network = DuelingDQN(
            self.state_size,
            self.action_size
        )

        # ==========================================
        # TARGET NETWORK
        # ==========================================

        self.target_network = DuelingDQN(
            self.state_size,
            self.action_size
        )

        # Copy initial weights
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        # Target network is only used for prediction
        self.target_network.eval()

        # ==========================================
        # OPTIMIZER
        # ==========================================

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=0.0003
        )

        # ==========================================
        # EPSILON-GREEDY
        # ==========================================

        self.epsilon = 1.0

        self.epsilon_min = 0.05

        self.epsilon_decay = 0.997

    # ==============================================
    # CHOOSE ACTION
    # ==============================================

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:

            return random.randint(
                0,
                self.action_size - 1
            )

        # Exploitation

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.q_network(
                state_tensor
            )

        action = torch.argmax(
            q_values,
            dim=1
        ).item()

        return action

    # ==============================================
    # STORE EXPERIENCE
    # ==============================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.add(
            state,
            action,
            reward,
            next_state,
            done
        )

    # ==============================================
    # LEARN FROM REPLAY BUFFER
    # ==============================================

    def learn(self):

        # Wait until enough experiences exist

        if len(self.memory) < self.batch_size:

            return None

        # ==========================================
        # SAMPLE BATCH
        # ==========================================

        batch = self.memory.sample(
            self.batch_size
        )

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for experience in batch:

            state, action, reward, next_state, done = experience

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # ==========================================
        # CONVERT TO TENSORS
        # ==========================================

        states = torch.tensor(
            states,
            dtype=torch.float32
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        )

        # ==========================================
        # CURRENT Q VALUES
        # ==========================================

        current_q_values = self.q_network(
            states
        )

        current_q_values = current_q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # ==========================================
        # TARGET Q VALUES
        # ==========================================

        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            )

            max_next_q_values = next_q_values.max(
                dim=1
            )[0]

            target_q_values = (
                rewards
                + self.gamma
                * max_next_q_values
                * (1 - dones)
            )

        # ==========================================
        # LOSS
        # ==========================================

        loss = F.mse_loss(
            current_q_values,
            target_q_values
        )

        # ==========================================
        # BACKPROPAGATION
        # ==========================================

        self.optimizer.zero_grad()

        loss.backward()

        # Prevent very large gradients
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        return loss.item()

    # ==============================================
    # UPDATE TARGET NETWORK
    # ==============================================

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

    # ==============================================
    # EPSILON DECAY
    # ==============================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )