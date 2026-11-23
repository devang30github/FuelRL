# FuelRL: a hack-resistant RL environment for fuel-stop planning

A custom Gymnasium environment where an agent drives a route with a fuel tank and varying
prices and decides how much to buy at each station. Includes an exact DP solver used as a
verifier, tests that prove the reward can and can't be gamed, a trained PPO agent, baselines,
a Streamlit demo, and a Dockerfile.

## Task (MDP)
- 15 stations; each leg needs 1-4 fuel units; tank holds 10; start with 2 units.
- Prices are integers in [80, 120], resampled per episode.
- Action: units to buy (0-10, clipped to free tank space), then drive to the next station.
- Observation (10 floats): progress, fuel, current price, **fuel deficit for the next leg**,
  next 3 prices, next 3 leg needs.
- Episode ends on arrival or when the agent can't cover the next leg (stranded).

## Verifier
`fuelrl/dp.py` computes the exact minimum-cost plan by backward DP. Tests check that
replaying the DP plan in the environment reproduces its cost exactly, so "optimal" is
trustworthy. All agents are scored by **gap to optimal (%)** on 200 held-out routes
(seeds 10000-10199), reported together with **arrival rate**.

## Cheat log (reward hacking and evaluation pitfalls found)
1. **Early-stranding exploit.** With `reward = -money spent` and free stranding, PPO learned to
   buy nothing: arrival 0%, strand rate 100%, "cost" 0. Fix: a stranded agent pays for the rescue
   (remaining fuel at worst price x 1.5). Unit tests assert the hack works on the naive reward and
   is closed on the shaped one.
2. **Last-leg precision failure.** After fixing (1), PPO arrived on only 9% of routes. A
   diagnostic showed 83% of strandings at the final leg, short by 1-2 units. Adding the explicit
   deficit feature raised arrival to 28% at 100k steps; 300k steps reached 89%.
3. **Misleading metrics.** Mean cost over all episodes rewards stranding (it was 0 for the naive
   agent), and gap-to-optimal computed only on arrived episodes flatters an agent that fails the
   hard routes. Fix: always report arrival next to gap, and add a safety shield (below) so
   comparisons cover all 200 routes.
4. **Arrival alone is a weak signal.** A uniformly random policy arrives 85% of the time because
   it overbuys; its gap is 37.5%.

## Results (200 held-out routes)
| Policy | Arrival | Gap to optimal |
|---|---|---|
| Random | 0.85 | 37.50% |
| Buy-just-enough (greedy) | 1.00 | 13.08% |
| Fill-when-cheap heuristic (untuned threshold) | 1.00 | 15.56% |
| PPO, shaped reward, 300k steps (seed 0) | 0.89 | 6.20% (arrived routes only) |
| PPO + safety shield (seed 0) | 1.00 | 6.32% |

Across 5 training seeds (`multiseed.py`): shielded PPO gap = **5.93% +/- 0.34** (std), beating
greedy (13.08%) on 5/5 seeds, which cuts excess cost over the optimum by about 55%. Unshielded
PPO arrival ranges 0.82-0.93.

**Safety shield:** the agent proposes a purchase, and a rule raises it to the minimum needed
to cover the next leg. Stranding becomes impossible, so PPO's gap is measured on every route.
PPO's savings come from price-aware buying that greedy can't do.

## Reproduce
```bash
docker build -t fuelrl .
docker run --rm fuelrl                       # tests
docker run --rm fuelrl python baselines.py   # baselines table
python train.py --mode naive|shaped --steps 300000 --seed 0
python multiseed.py
streamlit run app.py
```

## Limitations
- One small route distribution (15 stations, integer fuel, uniform prices); no stochastic
  dynamics. The gap to optimal may not transfer to harder settings.
- Classic RL agent in a Gymnasium env, not an LLM-agent environment.
- The shield is a hand-written rule. Action masking during training would be the cleaner fix.
- 5 training seeds, all evaluated on the same 200 held-out routes, so the +/- reflects
  training-seed variance only. No significance testing.

## Next steps
- Wrap the same environment as a tool-use task for an LLM agent, with state-based grading.
- Train with action masking instead of a post-hoc shield.
- Package in a standard environment format (e.g. Harbor).