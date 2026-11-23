import numpy as np
import gymnasium as gym
from gymnasium import spaces

PMIN, PMAX = 80, 120
SCALE = 100.0
LOOKAHEAD = 3


class FuelRouteEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_stations=15, capacity=10, start_fuel=2, reward_mode="shaped"):
        super().__init__()
        assert reward_mode in ("naive", "shaped")
        self.n = n_stations
        self.cap = capacity
        self.start_fuel = start_fuel
        self.reward_mode = reward_mode
        self.action_space = spaces.Discrete(capacity + 1)  # units to buy
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4 + 2 * LOOKAHEAD,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "instance" in options:
            inst = options["instance"]
            self.prices = np.array(inst["prices"], dtype=np.int64)
            self.segs = np.array(inst["segs"], dtype=np.int64)
        else:
            self.prices = self.np_random.integers(PMIN, PMAX + 1, size=self.n)
            self.segs = self.np_random.integers(1, 5, size=self.n - 1)
        self.i = 0
        self.fuel = self.start_fuel
        self.total_cost = 0
        self.stranded = False
        self.arrived = False
        return self._obs(), self._info()

    def _norm_price(self, p):
        return (p - PMIN) / (PMAX - PMIN)

    def _obs(self):
        i, n = self.i, self.n
        nxt_prices = [self._norm_price(self.prices[min(i + 1 + j, n - 1)]) for j in range(LOOKAHEAD)]
        nxt_needs = [(self.segs[i + j] / self.cap) if i + j < n - 1 else 0.0 for j in range(LOOKAHEAD)]
        deficit = max(0, int(self.segs[i]) - self.fuel) / self.cap if i < n - 1 else 0.0
        vec = [i / (n - 1), self.fuel / self.cap, self._norm_price(self.prices[i]), deficit] + nxt_prices + nxt_needs
        return np.clip(np.array(vec, dtype=np.float32), 0.0, 1.0)

    def _info(self):
        return {
            "station": self.i,
            "fuel": self.fuel,
            "total_cost": self.total_cost,
            "stranded": self.stranded,
            "arrived": self.arrived,
        }

    def step(self, action):
        buy = int(min(int(action), self.cap - self.fuel))
        cost = buy * int(self.prices[self.i])
        self.fuel += buy
        self.total_cost += cost
        reward = -cost / SCALE

        need = int(self.segs[self.i])
        terminated = False
        if self.fuel < need:
            self.stranded = True
            terminated = True
            if self.reward_mode == "shaped":
                # stranded agent pays for the rescue: remaining fuel at worst price + premium
                remaining = int(self.segs[self.i:].sum()) - self.fuel
                reward -= 1.5 * remaining * PMAX / SCALE
        else:
            self.fuel -= need
            self.i += 1
            if self.i == self.n - 1:
                self.arrived = True
                terminated = True
        return self._obs(), float(reward), terminated, False, self._info()