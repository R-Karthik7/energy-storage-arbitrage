import gymnasium as gym
from gymnasium import spaces
import numpy as np

from battery import Battery


class EnergyStorageEnv(gym.Env):

    def __init__(self, prices):

        super(EnergyStorageEnv, self).__init__()

        # Store electricity prices
        self.prices = np.array(prices, dtype=np.float32)

        # Battery
        self.battery = Battery()

        # Current hour
        self.current_step = 0

        # Actions
        # 0 = Idle
        # 1 = Charge
        # 2 = Discharge
        self.action_space = spaces.Discrete(3)

        # State:
        # 0 = Battery SOC
        # 1 = Electricity price
        # 2 = Hour of day
        self.observation_space = spaces.Box(
            low=np.array([0.0, -10.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 2.0, 23.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Reset battery
        self.battery = Battery()

        # Start from first price
        self.current_step = 0

        state = self._get_state()

        return state, {}

    def _get_state(self):

        price = self.prices[self.current_step]

        # Normalize price
        normalized_price = price / 150.0

        # Hour of day
        hour = self.current_step % 24

        normalized_hour = hour / 23.0

        state = np.array([
            self.battery.get_soc(),
            normalized_price,
            normalized_hour
        ], dtype=np.float32)

        return state

    def step(self, action):

        price = self.prices[self.current_step]

        reward = 0.0

        # ==========================
        # ACTION 0: IDLE
        # ==========================

        if action == 0:

            reward = 0.0

        # ==========================
        # ACTION 1: CHARGE
        # ==========================

        elif action == 1:

            energy_charged = self.battery.charge(20)

            # Cost of electricity
            reward = -((energy_charged / 1000.0) * price)

        # ==========================
        # ACTION 2: DISCHARGE
        # ==========================

        elif action == 2:

            energy_discharged = self.battery.discharge(20)

            # Revenue from selling electricity
            reward = (energy_discharged / 1000.0) * price

        # Move to next hour
        self.current_step += 1

        # Check whether episode is finished
        terminated = self.current_step >= len(self.prices)
        truncated = False
        # ==========================================
        # TERMINAL BATTERY VALUE

        # ==========================================

        if terminated:

            remaining_energy = self.battery.energy_kwh

            terminal_value = (
                remaining_energy / 1000.0
            ) * price

            reward += terminal_value

        # If episode is finished, use last valid state
        if terminated:

            next_state = self._get_state_at_end()

        else:

            next_state = self._get_state()

        info = {
            "price": price,
            "soc": self.battery.get_soc(),
            "action": action,
            "reward": reward
        }

        return next_state, reward, terminated, truncated, info

    def _get_state_at_end(self):

        last_price = self.prices[-1]

        normalized_price = last_price / 150.0

        hour = (len(self.prices) - 1) % 24

        normalized_hour = hour / 23.0

        state = np.array([
            self.battery.get_soc(),
            normalized_price,
            normalized_hour
        ], dtype=np.float32)

        return state