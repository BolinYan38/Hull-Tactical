"""Check notebook syntax, published exports and optional local-data accounting."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/six_strategy_test"
CANDIDATES = ["trend", "mean_reversion", "defensive", "lightgbm", "catboost", "elasticnet"]
ACCOUNTS = CANDIDATES + ["equal_six", "ot_six", "buy_hold"]


def read_frame(path: Path) -> dict[int, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        index_name = reader.fieldnames[0]
        result = {}
        for row in reader:
            index = int(row[index_name])
            if index in result:
                raise ValueError(f"Duplicate date {index} in {path.name}")
            result[index] = {name: float(value) for name, value in row.items() if name != index_name}
    return result


def close(actual: float, expected: float, label: str, tolerance: float = 1e-8) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(f"Non-finite value: {label}")
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{label}: saved={actual:.12g}, recomputed={expected:.12g}")


def validate(skip_data: bool = False) -> dict:
    for name in ["README.md", "LICENSE", "requirements.txt", "requirements-dev.txt"]:
        if not (ROOT / name).is_file() or not (ROOT / name).stat().st_size:
            raise ValueError(f"Missing or empty project file: {name}")
    notebook_path = ROOT / "Optimal-transport-for-Hull-Tactical.ipynb"
    snapshot = json.loads((ROOT / "scripts/research_snapshot.json").read_text(encoding="utf-8"))
    frozen_files = {snapshot["notebook"]["path"]: snapshot["notebook"]["sha256"], **snapshot["results_sha256"]}
    for relative, expected_hash in frozen_files.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Frozen research snapshot changed: {relative}")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = 0
    for number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"notebook:cell{number}")
            code_cells += 1
            if any(output.get("output_type") == "error" for output in cell.get("outputs", [])):
                raise ValueError(f"Saved error output in main notebook cell {number}")
    for directory in ["scripts", "tests"]:
        for path in (ROOT / directory).glob("*.py"):
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
    ast.parse((ROOT / "hull_six_strategy_kaggle_submission.py").read_text(encoding="utf-8"))
    with (RESULTS / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        metrics = {row[""]: {key: float(value) for key, value in row.items() if key}
                   for row in csv.DictReader(stream)}
    if set(metrics) != set(ACCOUNTS):
        raise ValueError("metrics.csv must describe all nine test accounts")
    frames = {name: read_frame(RESULTS / f"{name}.csv") for name in [
        "positions", "daily_net_returns", "weights", "scores",
    ]}
    dates = list(frames["positions"])
    if dates != list(range(7439, 9048)):
        raise ValueError("Expected the original 1609-day held-out interval, date_id 7439–9047")
    for name, frame in frames.items():
        if list(frame) != dates:
            raise ValueError(f"Date alignment differs in {name}.csv")
        if any(not math.isfinite(value) for row in frame.values() for value in row.values()):
            raise ValueError(f"Non-finite values in {name}.csv")
        expected = CANDIDATES if name in ["weights", "scores"] else ACCOUNTS
        if set(frame[dates[0]]) != set(expected):
            raise ValueError(f"Unexpected strategy columns in {name}.csv")
    selection_counts = {}
    for date in dates:
        positions, weights, scores = (frames[name][date] for name in ["positions", "weights", "scores"])
        close(sum(weights.values()), 1.0, f"weights sum at {date}")
        if any(value < 0 for value in weights.values()) or any(not 0 <= value <= 2 for value in positions.values()):
            raise ValueError(f"Invalid weight or position bound at {date}")
        close(positions["buy_hold"], 1.0, f"buy_hold position at {date}")
        close(positions["equal_six"], statistics.mean(positions[name] for name in CANDIDATES), f"equal allocation at {date}")
        close(positions["ot_six"], sum(weights[name] * positions[name] for name in CANDIDATES), f"OT position at {date}")
        logits = {name: math.exp((value - max(scores.values())) / 0.01) for name, value in scores.items()}
        total = sum(logits.values())
        ordered = sorted(CANDIDATES, key=lambda name: logits[name], reverse=True)
        selected, cumulative = [], 0.0
        for name in ordered:
            selected.append(name)
            cumulative += logits[name] / total
            if cumulative > 0.5:
                break
        normalizer = sum(logits[name] for name in selected)
        for name in CANDIDATES:
            close(weights[name], logits[name] / normalizer if name in selected else 0.0, f"selection rule {date}/{name}")
        count = sum(value > 0 for value in weights.values())
        selection_counts[count] = selection_counts.get(count, 0) + 1
    returns = frames["daily_net_returns"]
    for name in ACCOUNTS:
        series = [returns[date][name] for date in dates]
        nav, peak, drawdown = 1.0, 1.0, 0.0
        for value in series:
            nav *= 1.0 + value
            peak = max(peak, nav)
            drawdown = min(drawdown, nav / peak - 1.0)
        for key, value in {
            "total_return": nav - 1.0,
            "annual_return": nav ** (252 / len(dates)) - 1.0,
            "annual_volatility": statistics.stdev(series) * math.sqrt(252),
            "max_drawdown": drawdown,
            "mean_position": statistics.mean(frames["positions"][date][name] for date in dates),
        }.items():
            close(metrics[name][key], value, f"{name}/{key}")
    with (RESULTS / "weight_summary.csv").open(encoding="utf-8", newline="") as stream:
        summary = {row[""]: row for row in csv.DictReader(stream)}
    if set(summary) != set(CANDIDATES):
        raise ValueError("Incomplete weight_summary.csv")
    for name in CANDIDATES:
        for key, value in {
            "mean_weight": statistics.mean(frames["weights"][date][name] for date in dates),
            "selection_rate": statistics.mean(frames["weights"][date][name] > 0 for date in dates),
            "mean_score": statistics.mean(frames["scores"][date][name] for date in dates),
        }.items():
            close(float(summary[name][key]), value, f"weight_summary/{name}/{key}")
    if not (RESULTS / "test_curves.png").is_file():
        raise ValueError("Missing published test figure")
    if not skip_data:
        data_path = ROOT / "archive/train.csv"
        if not data_path.is_file():
            raise ValueError("Missing archive/train.csv; use --skip-data for source/export checks only")
        with data_path.open(encoding="utf-8", newline="") as stream:
            data = {int(row["date_id"]): row for row in csv.DictReader(stream)}
        if list(data) != list(range(9048)):
            raise ValueError("Expected original 9048 ordered and unique date_id values")
        for name in ACCOUNTS:
            before = 0.0
            excess_returns, turnovers = [], []
            for date in dates:
                position = frames["positions"][date][name]
                market, cash = float(data[date]["forward_returns"]), float(data[date]["risk_free_rate"])
                change = position - before
                sign = 1.0 if change > 0 else -1.0 if change < 0 else 0.0
                turnover = abs(change) / (1 + 0.0001 * position * sign)
                gross = position * market + (1 - position) * cash
                net = (1 - 0.0001 * turnover) * (1 + gross) - 1
                close(returns[date][name], net, f"net accounting {date}/{name}")
                turnovers.append(turnover)
                excess_returns.append(net - cash)
                before = position * (1 + market) / (1 + gross)
            close(metrics[name]["sharpe"], statistics.mean(excess_returns) / statistics.stdev(excess_returns) * math.sqrt(252), f"{name}/sharpe")
            close(metrics[name]["annual_turnover"], statistics.mean(turnovers) * 252, f"{name}/annual_turnover")
    return {"syntax": "passed", "main_code_cells": code_cells, "test_days": len(dates),
            "accounts": len(ACCOUNTS), "selection_counts": selection_counts,
            "exports": "consistent", "raw_data_accounting": "skipped" if skip_data else "consistent"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-data", action="store_true", help="Validate source and exports without Kaggle data")
    args = parser.parse_args()
    try:
        report = validate(skip_data=args.skip_data)
    except (OSError, ValueError, KeyError, SyntaxError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
