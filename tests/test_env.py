import numpy as np
from fuelrl.env import FuelRouteEnv, SCALE
from fuelrl.dp import optimal_plan


def run_plan(env, buys):
    total, info = 0.0, None
    for b in buys:
        _, r, term, _, info = env.step(b)
        total += r
        if term:
            break
    return total, info


def test_obs_shape_and_bounds():
    env = FuelRouteEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_dp_plan_replays_in_env():
    # env and solver must agree exactly, otherwise "optimal" means nothing
    for seed in range(50):
        env = FuelRouteEnv(reward_mode="shaped")
        env.reset(seed=seed)
        cost, buys = optimal_plan(env.prices, env.segs, env.cap, env.start_fuel)
        total, info = run_plan(env, buys)
        assert info["arrived"] and not info["stranded"]
        assert info["total_cost"] == cost
        assert abs(total + cost / SCALE) < 1e-6


def test_greedy_never_beats_optimal():
    for seed in range(50):
        env = FuelRouteEnv()
        env.reset(seed=seed)
        opt, _ = optimal_plan(env.prices, env.segs, env.cap, env.start_fuel)
        fuel, greedy = env.start_fuel, 0
        for i in range(env.n - 1):
            buy = max(0, int(env.segs[i]) - fuel)
            greedy += buy * int(env.prices[i])
            fuel += buy - int(env.segs[i])
        assert greedy >= opt


INSTANCE = {"prices": [100] * 15, "segs": [4] + [2] * 13}  # first leg needs 4, start fuel is 2


def test_naive_reward_is_hackable():
    env = FuelRouteEnv(reward_mode="naive")
    env.reset(options={"instance": INSTANCE})
    _, r, term, _, info = env.step(0)  # buy nothing, get stranded
    assert term and info["stranded"]
    assert r == 0.0  # strictly better than any honest run, which has negative reward


def test_shaped_reward_closes_the_hack():
    env = FuelRouteEnv(reward_mode="shaped")
    env.reset(options={"instance": INSTANCE})
    opt, _ = optimal_plan(env.prices, env.segs, env.cap, env.start_fuel)
    _, r, _, _, _ = env.step(0)
    assert r < -opt / SCALE  # stranding must be worse than finishing optimally