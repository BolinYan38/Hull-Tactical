"""Execute a copy of the research notebook with the invoking Python interpreter."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Optimal-transport-for-Hull-Tactical.ipynb"


def select_cells(notebook: dict, mode: str, until_section: int) -> tuple[list, list[int]]:
    """Select sections and omit parameter-search cells in fixed mode."""
    cells, skipped = [], []
    section = 0
    for index, original in enumerate(notebook["cells"]):
        source = "".join(original.get("source", []))
        if original["cell_type"] == "markdown":
            heading = re.match(r"^##\s+(\d+)\.", source)
            if heading:
                section = int(heading.group(1))
        if section > until_section:
            break
        cell = json.loads(json.dumps(original))
        if cell["cell_type"] == "code":
            tree = ast.parse(source)
            is_search = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ParameterSampler"
                for node in ast.walk(tree)
            )
            if mode == "fixed" and is_search:
                skipped.append(index)
                continue
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)
    return cells, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixed", "search"], default="fixed")
    parser.add_argument("--until-section", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--timeout", type=int, default=3600, help="Per-cell timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Print the execution plan only")
    parser.add_argument("--output", type=Path, help="Executed notebook; default: .artifacts/")
    args = parser.parse_args()
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells, skipped = select_cells(document, args.mode, args.until_section)
    code_count = sum(cell["cell_type"] == "code" for cell in cells)
    print(f"Interpreter: {sys.executable}")
    print(f"Mode: {args.mode}; sections: 1–{args.until_section}; code cells: {code_count}")
    print(f"Skipped parameter-search cell indices (zero based): {skipped}")
    print(f"Working directory: {ROOT}")
    if args.until_section == 5:
        print("Section 5 regenerates the CSV files and figure in results/six_strategy_test/.")
    if args.dry_run:
        return 0
    if not (ROOT / "archive/train.csv").is_file():
        parser.error("Missing archive/train.csv. Follow the data setup section in README.md.")
    output = args.output or ROOT / ".artifacts" / f"research-{args.mode}-section{args.until_section}.ipynb"
    output = output.expanduser().resolve()
    if output in [NOTEBOOK.resolve(), (ROOT / "Optimal-transport-for-Hull-Tactical 2.ipynb").resolve()]:
        parser.error("The output must be a separate executed notebook.")
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as error:
        parser.error(f"Install requirements-dev.txt first: {error}")

    notebook = nbformat.from_dict(document)
    notebook.cells = [nbformat.from_dict(cell) for cell in cells]
    notebook.metadata["hull_execution"] = {
        "mode": args.mode, "until_section": args.until_section,
        "skipped_source_cells": skipped,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # A temporary kernelspec keeps execution on this interpreter without changing
    # the user's Jupyter installation or relying on a globally registered kernel.
    previous_jupyter_path = os.environ.get("JUPYTER_PATH")
    with tempfile.TemporaryDirectory(prefix="hull-kernel-") as temporary:
        kernel_name = "hull-research"
        kernel_dir = Path(temporary) / "kernels" / kernel_name
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Hull research (current interpreter)", "language": "python",
        }), encoding="utf-8")
        os.environ["JUPYTER_PATH"] = temporary + (
            os.pathsep + previous_jupyter_path if previous_jupyter_path else ""
        )
        client = NotebookClient(
            notebook, timeout=args.timeout, kernel_name=kernel_name,
            resources={"metadata": {"path": str(ROOT)}}, allow_errors=False,
        )
        try:
            client.execute()
        finally:
            nbformat.write(notebook, output)
            if previous_jupyter_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous_jupyter_path
            print(f"Execution output: {output}")
    print("Notebook execution completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
