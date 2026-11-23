from fuelrl.env import FuelRouteEnv
from fuelrl.dp import optimal_plan


def evaluate(policy, n_episodes=200, seed0=10_000):
    """policy: obs -> int action. Fixed held-out seeds so runs are comparable."""
    env = FuelRouteEnv(reward_mode="shaped")  # reward mode doesn't matter here, we read info
    arrived, gaps, all_costs, stations = 0, [], [], []
    for k in range(n_episodes):
        obs, info = env.reset(seed=seed0 + k)
        opt, _ = optimal_plan(env.prices, env.segs, env.cap, env.start_fuel)
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy(obs))
        all_costs.append(info["total_cost"])
        stations.append(info["station"])
        if info["arrived"]:
            arrived += 1
            gaps.append(100 * (info["total_cost"] - opt) / opt)
    return {
        "arrival_rate": arrived / n_episodes,
        "strand_rate": 1 - arrived / n_episodes,
        "mean_cost_all_episodes": sum(all_costs) / n_episodes,
        "mean_station_reached": sum(stations) / n_episodes,
        "mean_gap_to_optimal_pct": (sum(gaps) / len(gaps)) if gaps else None,
    }