"""Small financial-accounting and temporal-boundary regressions; no model fitting."""

import ast
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Optimal-transport-for-Hull-Tactical.ipynb"


def load_function(name, namespace):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook-function", "exec"), namespace)
                return namespace[name]
    raise AssertionError(f"Notebook function missing: {name}")


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.backtest = load_function("backtest_positions", {
            "np": np, "pd": pd, "COST_RATE": 0.0001,
        })

    def test_first_trade_pays_cost_before_market_return(self):
        positions = pd.DataFrame({"market": [1.0], "cash": [0.0]})
        net, turnover = self.backtest(positions, pd.Series([0.1]), pd.Series([0.001]))
        self.assertAlmostEqual(turnover.loc[0, "market"], 1 / 1.0001)
        self.assertAlmostEqual(net.loc[0, "market"], 1.1 / 1.0001 - 1)
        self.assertAlmostEqual(net.loc[0, "cash"], 0.001)
        self.assertEqual(turnover.loc[0, "cash"], 0)

    def test_matching_drifted_position_needs_no_trade(self):
        drifted = 0.55 / 1.05
        positions = pd.DataFrame({"strategy": [0.5, drifted]})
        _, turnover = self.backtest(positions, pd.Series([0.1, 0.0]), pd.Series([0.0, 0.0]))
        self.assertAlmostEqual(turnover.loc[1, "strategy"], 0.0)

    def test_leveraged_cash_balance_uses_financing_rate(self):
        positions = pd.DataFrame({"strategy": [2.0]})
        net, turnover = self.backtest(positions, pd.Series([0.1]), pd.Series([0.01]))
        self.assertAlmostEqual(turnover.iloc[0, 0], 2 / 1.0002)
        self.assertAlmostEqual(net.iloc[0, 0], (1 - 0.0001 * 2 / 1.0002) * 1.19 - 1)

    def test_oof_gap_restarts_from_cash(self):
        positions = pd.DataFrame({"strategy": [1.0, np.nan, 1.0]})
        _, turnover = self.backtest(positions, pd.Series([0.0] * 3), pd.Series([0.0] * 3), restart_segments=True)
        self.assertTrue(np.isnan(turnover.iloc[1, 0]))
        self.assertAlmostEqual(turnover.iloc[2, 0], 1 / 1.0001)


class TemporalAndDistanceTests(unittest.TestCase):
    def test_history_at_maturity_boundary_is_allowed_next_day_is_excluded(self):
        valid = pd.Index([39, 40, 41, 100])
        namespace = {
            "np": np, "pd": pd, "WINDOW": 40, "HORIZON": 20, "K": 2,
            "heapq": __import__("heapq"), "valid_state_indices": valid,
            "distance_positions": {value: value for value in valid},
            "window_means": {"rolling_state": np.zeros((101, 1))},
            "window_stds": {"rolling_state": np.zeros((101, 1))},
            "state_dimension": 1,
            "market_distance_cached": lambda t, s: float(abs(t - s)),
        }
        search = load_function("find_nearest_neighbors", namespace)
        distances, _, candidates = search(100, pd.Index([39, 40, 41]))
        self.assertEqual(set(distances.index), {39, 40})
        self.assertEqual(candidates, 2)

    def test_wasserstein_matches_reordered_points(self):
        distance = load_function("wasserstein_distance_2", {
            "np": np, "linear_sum_assignment": linear_sum_assignment, "cdist": cdist,
        })
        window = np.array([[0.0, 1.0], [3.0, 2.0], [1.0, 4.0]])
        self.assertAlmostEqual(distance(window, window[[2, 0, 1]]), 0.0)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("hull_runner", ROOT / "scripts/run_notebook.py")
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_fixed_mode_skips_only_three_search_cells(self):
        fixed, skipped = self.runner.select_cells(self.notebook, "fixed", 5)
        search, _ = self.runner.select_cells(self.notebook, "search", 5)
        self.assertEqual(len(skipped), 3)
        self.assertEqual(len(search) - len(fixed), 3)
        source = "\n".join("".join(cell.get("source", [])) for cell in fixed)
        self.assertIn("def generate_oof_strategy", source)
        self.assertIn("test_market_position", source)

    def test_section_two_smoke_check_stops_before_backtest(self):
        cells, skipped = self.runner.select_cells(self.notebook, "fixed", 2)
        source = "\n".join("".join(cell.get("source", [])) for cell in cells)
        self.assertIn("def find_nearest_neighbors", source)
        self.assertNotIn("final_lgb.fit", source)
        self.assertNotIn("test_net_return", source)
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
