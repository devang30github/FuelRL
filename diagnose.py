from collections import Counter
from stable_baselines3 import PPO
from fuelrl.env import FuelRouteEnv

model = PPO.load("models/ppo_shaped")
env = FuelRouteEnv(reward_mode="shaped")
where, rows = Counter(), []

for k in range(200):
    obs, _ = env.reset(seed=10_000 + k)
    done = False
    while not done:
        st, fuel_before = env.i, env.fuel
        a = int(model.predict(obs, deterministic=True)[0])
        bought = min(a, env.cap - fuel_before)
        need = int(env.segs[st])
        obs, _, done, _, info = env.step(a)
        if info["stranded"]:
            where[st] += 1
            if len(rows) < 10:
                rows.append((st, fuel_before, need, bought, need - fuel_before - bought, int(env.prices[st])))

print("stranding station histogram:", sorted(where.items()))
print("station | fuel_before | need | bought | short_by | price")
for r in rows:
    print("   ".join(map(str, r)))