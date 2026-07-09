"""A naive trading-signal simulation: translates model predictions into a position, computes
PnL against the actual mid-price series with realistic transaction costs -- closing the loop
from prediction quality to economic reality, in the same spirit as statlab's own real-market
results section (an edge that looks good on paper needs to survive contact with real costs).
"""

from dataclasses import dataclass

import numpy as np

from deeplob.data.labeling import Label

_POSITION_BY_LABEL = {Label.DOWN: -1.0, Label.STATIONARY: 0.0, Label.UP: 1.0}


@dataclass(frozen=True)
class SimulationReport:
    gross_pnl: float
    net_pnl: float
    total_transaction_cost: float
    num_trades: int
    num_steps: int

    def summary(self) -> str:
        return (
            f"steps:      {self.num_steps}\n"
            f"trades:     {self.num_trades}\n"
            f"gross PnL:  {self.gross_pnl:+.6f}\n"
            f"total cost: {self.total_transaction_cost:.6f}\n"
            f"net PnL:    {self.net_pnl:+.6f}"
        )


def predictions_to_positions(predictions: np.ndarray) -> np.ndarray:
    """Maps `Label` predictions (DOWN/STATIONARY/UP) to a naive directional position:
    short -1, flat 0, long +1. The simplest possible translation from a 3-class prediction to
    a position -- no sizing, no confidence-weighting -- deliberately, since this simulation's
    purpose is to test whether the *direction* signal survives contact with real costs, not
    to optimize a position-sizing strategy on top of it.
    """
    return np.array([_POSITION_BY_LABEL[Label(p)] for p in predictions])


def simulate_pnl(
    mid_prices: np.ndarray, positions: np.ndarray, transaction_cost_bps: float
) -> SimulationReport:
    """`mid_prices` is `[N+1]`, `positions` is `[N]` -- one more price than positions, since
    `positions[t]` is held from `mid_prices[t]` to `mid_prices[t+1]` and earns that period's
    return. Transaction cost is charged in basis points of notional per unit of position
    *change* (the standard proportional-cost backtest convention) on every step where the
    position differs from the prior one, including the very first step (entering from flat).
    """
    if mid_prices.shape[0] != positions.shape[0] + 1:
        raise ValueError("mid_prices must have exactly one more element than positions")
    if positions.shape[0] == 0:
        raise ValueError("cannot simulate an empty position series")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")

    returns = (mid_prices[1:] - mid_prices[:-1]) / mid_prices[:-1]
    gross_pnl_per_step = positions * returns

    prior_positions = np.concatenate(([0.0], positions[:-1]))
    position_changes = np.abs(positions - prior_positions)
    cost_per_step = position_changes * (transaction_cost_bps / 10_000.0)

    net_pnl_per_step = gross_pnl_per_step - cost_per_step

    return SimulationReport(
        gross_pnl=float(gross_pnl_per_step.sum()),
        net_pnl=float(net_pnl_per_step.sum()),
        total_transaction_cost=float(cost_per_step.sum()),
        num_trades=int((position_changes > 0).sum()),
        num_steps=int(positions.shape[0]),
    )
