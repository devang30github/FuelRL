import numpy as np
from stable_baselines3 import PPO
from fuelrl.evaluate import evaluate

CAP = 10
rng = np.random.default_rng(0)
model = PPO.load("models/ppo_shaped_300000_s0")  # the 300k model, same held-out seeds as the baselines

policies = {
    # obs[3] is the deficit feature: max(0, need - fuel) / cap
    "random": lambda obs: int(rng.integers(0, CAP + 1)),
    "buy-just-enough (greedy)": lambda obs: int(round(obs[3] * CAP)),
    # obs[2] is the normalized price: fill the tank when cheap, otherwise buy just enough
    "fill-when-cheap heuristic": lambda obs: CAP if obs[2] < 0.33 else int(round(obs[3] * CAP)),
    "PPO (shaped, 300k)": lambda obs: int(model.predict(obs, deterministic=True)[0]),
    "PPO + safety shield": lambda obs: max(int(model.predict(obs, deterministic=True)[0]), int(round(obs[3] * CAP))),
}

print(f"{'policy':28s} {'arrival':>8s} {'gap to optimal':>16s}")
for name, pol in policies.items():
    s = evaluate(pol)
    gap = "n/a" if s["mean_gap_to_optimal_pct"] is None else f"{s['mean_gap_to_optimal_pct']:.2f}%"
    print(f"{name:28s} {s['arrival_rate']:8.2f} {gap:>16s}")