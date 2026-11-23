import pandas as pd
import streamlit as st
from stable_baselines3 import PPO
from fuelrl.env import FuelRouteEnv
from fuelrl.dp import optimal_plan

CAP = 10
st.set_page_config(page_title="FuelRL", layout="wide")


@st.cache_resource
def load_model():
    return PPO.load("models/ppo_shaped_300000_s0")


def rollout(instance, pick):
    env = FuelRouteEnv(reward_mode="shaped")
    obs, _ = env.reset(options={"instance": instance})
    rows, done, info = [], False, {}
    while not done:
        i, fuel_before = env.i, env.fuel
        a = int(pick(obs))
        obs, _, done, _, info = env.step(a)
        rows.append({
            "station": i,
            "price": int(env.prices[i]),
            "fuel_before": fuel_before,
            "bought": min(a, CAP - fuel_before),
            "cum_cost": info["total_cost"],
        })
    return pd.DataFrame(rows), info


model = load_model()
st.title("FuelRL: fuel-stop planning with a trained RL agent")

seed = st.sidebar.number_input("Route seed", min_value=0, value=10000, step=1)
show_raw = st.sidebar.checkbox("Also show unshielded PPO (can strand)", value=False)

# Build the route
env0 = FuelRouteEnv()
env0.reset(seed=int(seed))
instance = {"prices": env0.prices.tolist(), "segs": env0.segs.tolist()}
n = env0.n

opt_cost, opt_buys = optimal_plan(env0.prices, env0.segs, env0.cap, env0.start_fuel)
buys_iter = iter(opt_buys)

ppo_pick = lambda o: int(model.predict(o, deterministic=True)[0])
runs = {
    "Optimal (DP)": rollout(instance, lambda o: next(buys_iter)),
    "PPO + shield": rollout(instance, lambda o: max(ppo_pick(o), int(round(o[3] * CAP)))),
    "Greedy (buy just enough)": rollout(instance, lambda o: int(round(o[3] * CAP))),
}
if show_raw:
    runs["PPO (no shield)"] = rollout(instance, ppo_pick)

# Headline metrics
cols = st.columns(len(runs))
for col, (name, (df, info)) in zip(cols, runs.items()):
    cost = info["total_cost"]
    if info["stranded"]:
        col.metric(name, "STRANDED", f"at station {info['station']}", delta_color="inverse")
    else:
        gap = 100 * (cost - opt_cost) / opt_cost
        col.metric(name, f"{cost}", f"+{gap:.1f}% vs optimal" if gap > 0 else "optimal", delta_color="inverse")

# Charts
idx = range(n - 1)
def series(col):
    return pd.DataFrame({
        name: pd.Series(df[col].values, index=df["station"].values).reindex(idx)
        for name, (df, _) in runs.items()
    })

left, right = st.columns(2)
with left:
    st.subheader("Cumulative money spent")
    st.line_chart(series("cum_cost"))
with right:
    st.subheader("Fuel in tank on arrival at each station")
    st.line_chart(series("fuel_before"))

st.subheader("Route: price per fuel unit at each station")
st.bar_chart(pd.DataFrame({"price": instance["prices"][: n - 1]}))

st.subheader("Per-station decisions")
tabs = st.tabs(list(runs.keys()))
for tab, (name, (df, _)) in zip(tabs, runs.items()):
    tab.dataframe(df, use_container_width=True)