import glob
import json
import re

import numpy as np
from stable_baselines3 import PPO

from fuelrl.evaluate import evaluate

CAP = 10
raw = lambda m: (lambda obs: int(m.predict(obs, deterministic=True)[0]))
shielded = lambda m: (lambda obs: max(int(m.predict(obs, deterministic=True)[0]), int(round(obs[3] * CAP))))
fmt = lambda x: "n/a" if x is None else f"{x:.2f}"

rows = []
for path in sorted(glob.glob("models/ppo_shaped_300000_s*.zip")):
    seed = int(re.search(r"_s(\d+)\.zip$", path).group(1))
    m = PPO.load(path)
    r, s = evaluate(raw(m)), evaluate(shielded(m))
    rows.append({
        "seed": seed,
        "raw_arrival": r["arrival_rate"],
        "raw_gap_pct": r["mean_gap_to_optimal_pct"],
        "shield_arrival": s["arrival_rate"],
        "shield_gap_pct": s["mean_gap_to_optimal_pct"],
    })

greedy = evaluate(lambda obs: int(round(obs[3] * CAP)))
print(f"{'seed':>4} {'raw arrival':>12} {'raw gap %':>10} {'shield arrival':>15} {'shield gap %':>13}")
for r in rows:
    print(f"{r['seed']:>4} {r['raw_arrival']:>12.2f} {fmt(r['raw_gap_pct']):>10} "
          f"{r['shield_arrival']:>15.2f} {fmt(r['shield_gap_pct']):>13}")

gaps = np.array([r["shield_gap_pct"] for r in rows])
std = gaps.std(ddof=1) if len(gaps) > 1 else float("nan")
print(f"\nShielded PPO gap over {len(gaps)} seeds: {gaps.mean():.2f}% +/- {std:.2f} (std)")
print(f"Greedy gap: {greedy['mean_gap_to_optimal_pct']:.2f}%  (arrival {greedy['arrival_rate']:.2f})")
print(f"Seeds beating greedy: {(gaps < greedy['mean_gap_to_optimal_pct']).sum()} / {len(gaps)}")

with open("results/summary.json", "w") as f:
    json.dump({"seeds": rows, "greedy": greedy}, f, indent=2)