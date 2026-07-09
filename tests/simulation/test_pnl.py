import numpy as np
import pytest

from deeplob.data.labeling import Label
from deeplob.simulation.pnl import predictions_to_positions, simulate_pnl


def test_predictions_to_positions_hand_computed() -> None:
    predictions = np.array([Label.DOWN, Label.STATIONARY, Label.UP, Label.UP, Label.DOWN])
    positions = predictions_to_positions(predictions)
    np.testing.assert_array_equal(positions, [-1.0, 0.0, 1.0, 1.0, -1.0])


def test_simulate_pnl_hand_computed() -> None:
    # mid_prices=[100, 101, 99, 99], positions=[1 (long), 1 (long), 0 (flat)].
    # returns: (101-100)/100=0.01, (99-101)/101=-2/101, (99-99)/99=0.
    # gross per-step: [0.01, -2/101, 0.0] -> gross_pnl = 0.01 - 2/101.
    # prior_positions=[0,1,1] -> position_changes=[1,0,1] -> 2 trades.
    # cost_bps=10 (=0.001/unit) -> cost_per_step=[0.001,0,0.001] -> total_cost=0.002.
    mid_prices = np.array([100.0, 101.0, 99.0, 99.0])
    positions = np.array([1.0, 1.0, 0.0])

    report = simulate_pnl(mid_prices, positions, transaction_cost_bps=10.0)

    expected_gross = 0.01 - 2.0 / 101.0
    assert report.gross_pnl == pytest.approx(expected_gross)
    assert report.total_transaction_cost == pytest.approx(0.002)
    assert report.net_pnl == pytest.approx(expected_gross - 0.002)
    assert report.num_trades == 2
    assert report.num_steps == 3


def test_simulate_pnl_zero_cost_means_net_equals_gross() -> None:
    mid_prices = np.array([100.0, 105.0, 95.0])
    positions = np.array([1.0, -1.0])

    report = simulate_pnl(mid_prices, positions, transaction_cost_bps=0.0)

    assert report.net_pnl == pytest.approx(report.gross_pnl)
    assert report.total_transaction_cost == pytest.approx(0.0)


def test_simulate_pnl_always_flat_earns_and_costs_nothing() -> None:
    mid_prices = np.array([100.0, 110.0, 90.0, 120.0])
    positions = np.array([0.0, 0.0, 0.0])

    report = simulate_pnl(mid_prices, positions, transaction_cost_bps=25.0)

    assert report.gross_pnl == pytest.approx(0.0)
    assert report.net_pnl == pytest.approx(0.0)
    assert report.total_transaction_cost == pytest.approx(0.0)
    assert report.num_trades == 0


def test_simulate_pnl_holding_the_same_position_only_pays_entry_cost() -> None:
    # Position never changes after the first step -- only 1 trade (entering from flat).
    mid_prices = np.array([100.0, 101.0, 102.0, 103.0])
    positions = np.array([1.0, 1.0, 1.0])

    report = simulate_pnl(mid_prices, positions, transaction_cost_bps=10.0)

    assert report.num_trades == 1
    assert report.total_transaction_cost == pytest.approx(0.001)


def test_simulate_pnl_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="one more element"):
        simulate_pnl(np.array([100.0, 101.0]), np.array([1.0, 1.0]), transaction_cost_bps=1.0)


def test_simulate_pnl_rejects_empty_positions() -> None:
    with pytest.raises(ValueError, match="empty"):
        simulate_pnl(np.array([100.0]), np.array([]), transaction_cost_bps=1.0)


def test_simulate_pnl_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        simulate_pnl(np.array([100.0, 101.0]), np.array([1.0]), transaction_cost_bps=-1.0)
