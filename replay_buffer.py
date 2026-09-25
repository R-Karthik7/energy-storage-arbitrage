from collections import deque
import random


class ReplayBuffer:

    def __init__(self, capacity=10000):

        self.memory = deque(
            maxlen=capacity
        )


    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        self.memory.append(experience)


    def sample(self, batch_size):

        return random.sample(
            self.memory,
            batch_size
        )


    def __len__(self):

        return len(self.memory)