import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from double_dqn import DoubleDQN


class DoubleDQNAgent:

    def __init__(self):

        # ==========================================
        # ENVIRONMENT
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
        # ONLINE NETWORK
        # ==========================================

        self.q_network = DoubleDQN(
            self.state_size,
            self.action_size
        )

        # ==========================================
        # TARGET NETWORK
        # ==========================================

        self.target_network = DoubleDQN(
            self.state_size,
            self.action_size
        )

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.target_network.eval()

        # ==========================================
        # OPTIMIZER
        # ==========================================

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=0.0003
        )

        # ==========================================
        # EPSILON
        # ==========================================

        self.epsilon = 1.0

        self.epsilon_min = 0.05

        self.epsilon_decay = 0.997

    # ==============================================
    # CHOOSE ACTION
    # ==============================================

    def choose_action(self, state):

        if random.random() < self.epsilon:

            return random.randint(
                0,
                self.action_size - 1
            )

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.q_network(
                state_tensor
            )

        return torch.argmax(
            q_values,
            dim=1
        ).item()

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
    # DOUBLE DQN LEARNING
    # ==============================================

    def learn(self):

        if len(self.memory) < self.batch_size:

            return None

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
        # DOUBLE DQN TARGET
        # ==========================================

        with torch.no_grad():

            # Step 1:
            # Online network selects best action

            next_actions = torch.argmax(
                self.q_network(next_states),
                dim=1
            )

            # Step 2:
            # Target network evaluates that action

            next_q_values = self.target_network(
                next_states
            )

            selected_next_q_values = next_q_values.gather(
                1,
                next_actions.unsqueeze(1)
            ).squeeze(1)

            target_q_values = (
                rewards
                + self.gamma
                * selected_next_q_values
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