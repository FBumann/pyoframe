"""
Benchmark: pyoframe vs linopy — modeling overhead

Measures the full roundtrip from model creation to solution retrieval,
then subtracts Gurobi's self-reported solver time (RunTime) to isolate
pure modeling overhead.

Usage:
    python scripts/benchmark_vs_linopy.py
    python scripts/benchmark_vs_linopy.py --sizes 10 50 100 500 1000
    python scripts/benchmark_vs_linopy.py --runs 5 --csv results.csv
    python scripts/benchmark_vs_linopy.py --plot
    python scripts/benchmark_vs_linopy.py --problem dense_2d --sizes 10 50 100
"""

import argparse
import gc
import statistics
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

import linopy
import pyoframe as pf


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

class Problem(ABC):
    """Base class for benchmark problems.

    Subclasses define a specific LP/MIP structure and implement it for both
    pyoframe and linopy.  Each implementation returns:
      - timings dict with at least "data", "build", "solve_wall", "gurobi_runtime",
        "solution", "overhead"
      - the objective value (for correctness checks)
    """

    name: str  # short key used on CLI
    description: str  # one-liner shown in output

    @abstractmethod
    def run_pyoframe(self, N: int) -> tuple[dict[str, float], float]: ...

    @abstractmethod
    def run_linopy(self, N: int) -> tuple[dict[str, float], float]: ...

    @staticmethod
    def var_count(N: int) -> int:
        """Number of scalar variables for display purposes."""
        return 2 * N * N

    @staticmethod
    def con_count(N: int) -> int:
        """Number of scalar constraints for display purposes."""
        return 2 * N * N


class Dense2D(Problem):
    """Linopy's standard benchmark LP.

    Variables: x[i,j], y[i,j]  for i,j in range(N)   → 2*N² variables
    Constraints: x - y >= i (broadcast over j),
                 x + y >= 0                            → 2*N² constraints
    Objective: minimize 2*sum(x) + sum(y)
    """

    name = "dense_2d"
    description = "x[i,j], y[i,j] with 2*N² vars/cons"

    # -- data helpers (shared arrays, created once per N) ------------------

    @staticmethod
    def _make_data_pyoframe(N):
        s = pf.Set(i=range(N), j=range(N))
        rhs = pf.Param({"i": range(N), "rhs": range(N)})
        return s, rhs

    @staticmethod
    def _make_data_linopy(N):
        i_coords = pd.Index(range(N), name="i")
        j_coords = pd.Index(range(N), name="j")
        rhs = xr.DataArray(np.arange(N), dims=["i"], coords={"i": i_coords})
        return i_coords, j_coords, rhs

    # -- implementations ---------------------------------------------------

    def run_pyoframe(self, N):
        timings = {}

        t0 = time.perf_counter()
        s, rhs = self._make_data_pyoframe(N)
        timings["data"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        m = pf.Model("gurobi")
        m.attr.Silent = True
        m.x = pf.Variable(s)
        m.y = pf.Variable(s)
        m.c1 = m.x - m.y >= rhs.over("j")
        m.c2 = m.x + m.y >= 0
        m.minimize = 2 * m.x.sum() + m.y.sum()
        timings["build"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        m.optimize()
        timings["solve_wall"] = time.perf_counter() - t0
        timings["gurobi_runtime"] = m.attr.SolveTimeSec

        t0 = time.perf_counter()
        _ = m.x.solution
        _ = m.y.solution
        obj_val = m.minimize.value
        timings["solution"] = time.perf_counter() - t0

        timings["overhead"] = (
            timings["build"] + timings["solve_wall"] + timings["solution"]
            - timings["gurobi_runtime"]
        )
        return timings, obj_val

    def run_linopy(self, N):
        timings = {}

        t0 = time.perf_counter()
        i_coords, j_coords, rhs = self._make_data_linopy(N)
        timings["data"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        m = linopy.Model()
        x = m.add_variables(coords=[i_coords, j_coords], name="x")
        y = m.add_variables(coords=[i_coords, j_coords], name="y")
        m.add_constraints(x - y >= rhs, name="c1")
        m.add_constraints(x + y >= 0, name="c2")
        m.add_objective(2 * x.sum() + y.sum())
        timings["build"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        m.solve(solver_name="gurobi", io_api="direct", OutputFlag=0)
        timings["solve_wall"] = time.perf_counter() - t0
        timings["gurobi_runtime"] = m.solver_model.Runtime

        t0 = time.perf_counter()
        _ = x.solution
        _ = y.solution
        obj_val = m.objective.value
        timings["solution"] = time.perf_counter() - t0

        timings["overhead"] = (
            timings["build"] + timings["solve_wall"] + timings["solution"]
            - timings["gurobi_runtime"]
        )
        return timings, obj_val


# Registry — add new Problem subclasses here
PROBLEMS: dict[str, Problem] = {p.name: p for p in [Dense2D()]}

DEFAULT_SIZES = [10, 50, 100, 200, 500, 1000]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmarks(problem: Problem, sizes: list[int], num_runs: int = 3):
    print("Warming up...")
    problem.run_pyoframe(2)
    problem.run_linopy(2)
    gc.collect()

    phases = ["data", "build", "solve_wall", "gurobi_runtime", "solution", "overhead"]
    results = []

    for N in sizes:
        n_vars = problem.var_count(N)
        n_cons = problem.con_count(N)
        print(f"\nN = {N} ({n_vars:,} variables, {n_cons:,} constraints)")

        pf_runs: dict[str, list[float]] = {p: [] for p in phases}
        lp_runs: dict[str, list[float]] = {p: [] for p in phases}
        pf_obj = lp_obj = None

        for run in range(num_runs):
            gc.collect()
            t, obj = problem.run_pyoframe(N)
            pf_obj = obj
            for p in phases:
                pf_runs[p].append(t[p])

            gc.collect()
            t, obj = problem.run_linopy(N)
            lp_obj = obj
            for p in phases:
                lp_runs[p].append(t[p])

            print(f"  Run {run + 1}/{num_runs} done")

        if pf_obj is not None and lp_obj is not None:
            if abs(pf_obj - lp_obj) > 1e-4 * max(abs(pf_obj), abs(lp_obj), 1):
                print(f"  WARNING: Objective mismatch! pyoframe={pf_obj:.6f}, linopy={lp_obj:.6f}")
            else:
                print(f"  Objective: {pf_obj:.2f} (verified match)")

        for p in phases:
            pf_med = statistics.median(pf_runs[p])
            lp_med = statistics.median(lp_runs[p])
            results.append({
                "N": N,
                "phase": p,
                "pyoframe": pf_med,
                "linopy": lp_med,
                "ratio": pf_med / lp_med if lp_med > 0 else float("inf"),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt_time(t):
    if t < 0.0001:
        return f"{t * 1e6:>9.1f} us"
    if t < 0.1:
        return f"{t * 1000:>9.2f} ms"
    return f"{t:>9.4f}  s"


def _print_table(df, phase, label=None):
    rows = df[df["phase"] == phase]
    print(f"\n  {label or phase}")
    print(f"  {'N':>6}  {'pyoframe':>12}  {'linopy':>12}  {'pf / lp':>10}")
    print(f"  {'':->6}  {'':->12}  {'':->12}  {'':->10}")
    for _, r in rows.iterrows():
        print(
            f"  {int(r['N']):>6}  {fmt_time(r['pyoframe'])}  {fmt_time(r['linopy'])}  {r['ratio']:>10.2f}x"
        )


def print_results(df, problem: Problem):
    print(f"\n{'=' * 78}")
    print(f"  {problem.description}")
    print(f"  overhead = build + solve_wall + solution - gurobi_runtime")
    print(f"{'=' * 78}")

    _print_table(df, "overhead", "MODELING OVERHEAD  (everything except Gurobi solve)")
    _print_table(df, "data", "DATA CREATION  (index / param construction)")
    _print_table(df, "gurobi_runtime", "GUROBI SOLVE  (self-reported RunTime)")

    print(f"\n  {'─' * 74}")
    print(f"  ratio < 1 → pyoframe faster,  ratio > 1 → linopy faster")
    print(f"  {'─' * 74}")


def plot_results(df, problem: Problem):
    import matplotlib.pyplot as plt

    phases = ["overhead", "data", "gurobi_runtime"]
    labels = ["Modeling overhead", "Data creation", "Gurobi solve"]
    fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 4.5))

    for ax, phase, label in zip(axes, phases, labels):
        rows = df[df["phase"] == phase]
        ax.loglog(rows["N"], rows["pyoframe"], "o-", label="pyoframe", color="tab:blue")
        ax.loglog(rows["N"], rows["linopy"], "s-", label="linopy", color="tab:orange")
        ax.set_title(label)
        ax.set_xlabel("N")
        ax.set_ylabel("Time (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"pyoframe vs linopy — {problem.description}", fontsize=13)
    fig.tight_layout()
    out = "scripts/benchmark_vs_linopy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pyoframe vs linopy (modeling overhead)"
    )
    parser.add_argument(
        "--problem",
        choices=list(PROBLEMS),
        default="dense_2d",
        help=f"Problem to benchmark (default: dense_2d)",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
        help=f"Problem sizes N (default: {DEFAULT_SIZES})",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per size (default: 3)")
    parser.add_argument("--plot", action="store_true", help="Show log-log plots")
    parser.add_argument("--csv", type=str, default=None, help="Save results to CSV")
    args = parser.parse_args()

    problem = PROBLEMS[args.problem]
    print(f"Benchmark: pyoframe vs linopy — {problem.description}")
    print(f"Sizes: {args.sizes}, Runs: {args.runs}, Solver: Gurobi (direct)")

    df = run_benchmarks(problem, args.sizes, args.runs)
    print_results(df, problem)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nResults saved to {args.csv}")
    if args.plot:
        plot_results(df, problem)


if __name__ == "__main__":
    main()
