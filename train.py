import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from fuelrl.env import FuelRouteEnv
from fuelrl.evaluate import evaluate
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["naive", "shaped"], default="naive")
p.add_argument("--steps", type=int, default=100_000)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

venv = make_vec_env(lambda: FuelRouteEnv(reward_mode=a.mode), n_envs=4, seed=a.seed)
model = PPO("MlpPolicy", venv, verbose=0, seed=a.seed, ent_coef=0.01)
model.learn(total_timesteps=a.steps)
model.save(f"models/ppo_{a.mode}_{a.steps}_s{a.seed}")


stats = evaluate(lambda obs: int(model.predict(obs, deterministic=True)[0]))
print(f"\n=== PPO trained with reward_mode={a.mode}, {a.steps} steps ===")
for k, v in stats.items():
    print(f"{k:28s} {v}")

os.makedirs("results", exist_ok=True)
with open(f"results/{a.mode}_{a.steps}_s{a.seed}.json", "w") as f:
    json.dump({"mode": a.mode, "steps": a.steps, "seed": a.seed, **stats}, f, indent=2)