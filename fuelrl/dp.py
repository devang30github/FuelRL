import numpy as np


def optimal_plan(prices, segs, capacity, start_fuel):
    """Backward DP. Returns (min_cost, list_of_units_to_buy_per_station)."""
    n = len(prices)
    V = np.full((n, capacity + 1), np.inf)
    A = np.zeros((n, capacity + 1), dtype=int)
    V[n - 1, :] = 0.0
    for i in range(n - 2, -1, -1):
        need, p = int(segs[i]), int(prices[i])
        for f in range(capacity + 1):
            for buy in range(capacity - f + 1):
                nf = f + buy - need
                if nf < 0:
                    continue
                c = buy * p + V[i + 1, nf]
                if c < V[i, f]:
                    V[i, f] = c
                    A[i, f] = buy
    buys, f = [], start_fuel
    for i in range(n - 1):
        b = int(A[i, f])
        buys.append(b)
        f = f + b - int(segs[i])
    return float(V[0, start_fuel]), buys