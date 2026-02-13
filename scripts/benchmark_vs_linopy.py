"""
Benchmark: pyoframe vs linopy — modeling overhead

Measures the full roundtrip from model creation to solution retrieval,
then subtracts Gurobi's self-reported solver time (RunTime) to isolate
pure modeling overhead.

Memory is measured in a separate pass (--memory) using isolated subprocesses
so that RSS captures all allocators (Python, Rust/polars, C/numpy, Gurobi).

Usage:
    python scripts/benchmark_vs_linopy.py
    python scripts/benchmark_vs_linopy.py --sizes 10 50 100 500 1000
    python scripts/benchmark_vs_linopy.py --runs 5 --csv results.csv
    python scripts/benchmark_vs_linopy.py --memory
    python scripts/benchmark_vs_linopy.py --plot
    python scripts/benchmark_vs_linopy.py --problem dense_2d --sizes 10 50 100
"""

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
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
      - timings dict with at least "data", "build", "solve_wall",
        "gurobi_runtime", "solution", "overhead"
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


class SparseNetwork(Problem):
    """Sparse graph benchmark: K random outgoing edges per node.

    N nodes, each with K=10 random outgoing edges → ~K*N edges total.
    As N grows, density (K/N) drops, so pyoframe's COO representation
    (polars DataFrame storing only existing edges) should increasingly
    outperform linopy's dense NxN xarray with boolean mask.

    Variables:   x[i,j] on edges only, lb=0          → K*N variables
    Constraints: sum_j(x[i,j]) <= 1 for each node i  → N constraints
    Objective:   minimize sum(cost * x)
    """

    name = "sparse_network"
    description = "sparse graph K=10 edges/node, K*N vars on N² grid"
    K = 10

    @staticmethod
    def var_count(N: int) -> int:
        return SparseNetwork.K * N

    @staticmethod
    def con_count(N: int) -> int:
        return N

    @staticmethod
    def _make_graph(N):
        """Generate random edges: K distinct destinations per source node.

        Returns (src, dst, cost) arrays, each of length K*N.
        """
        K = min(SparseNetwork.K, N)
        np.random.seed(42)
        src_list = []
        dst_list = []
        for i in range(N):
            destinations = np.random.choice(N, size=K, replace=False)
            src_list.append(np.full(K, i, dtype=np.int64))
            dst_list.append(destinations.astype(np.int64))
        src = np.concatenate(src_list)
        dst = np.concatenate(dst_list)
        cost = np.random.rand(len(src))
        return src, dst, cost

    @staticmethod
    def _make_data_pyoframe(N):
        src, dst, cost = SparseNetwork._make_graph(N)
        edges = pl.DataFrame({"i": src, "j": dst})
        costs = edges.with_columns(pl.Series("cost", cost))
        return edges, costs

    @staticmethod
    def _make_data_linopy(N):
        src, dst, cost = SparseNetwork._make_graph(N)
        i_coords = pd.Index(range(N), name="i")
        j_coords = pd.Index(range(N), name="j")

        mask_np = np.zeros((N, N), dtype=bool)
        mask_np[src, dst] = True
        mask = xr.DataArray(mask_np, dims=["i", "j"], coords={"i": i_coords, "j": j_coords})

        cost_np = np.full((N, N), np.nan)
        cost_np[src, dst] = cost
        cost_da = xr.DataArray(cost_np, dims=["i", "j"], coords={"i": i_coords, "j": j_coords})

        return i_coords, j_coords, mask, cost_da

    def run_pyoframe(self, N):
        timings = {}

        t0 = time.perf_counter()
        edges, costs = self._make_data_pyoframe(N)
        timings["data"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        m = pf.Model("gurobi")
        m.attr.Silent = True
        m.x = pf.Variable(edges, lb=0)
        m.cap = m.x.sum("j") <= 1
        m.minimize = (pf.Param(costs) * m.x).sum()
        timings["build"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        m.optimize()
        timings["solve_wall"] = time.perf_counter() - t0
        timings["gurobi_runtime"] = m.attr.SolveTimeSec

        t0 = time.perf_counter()
        _ = m.x.solution
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
        i_coords, j_coords, mask, cost_da = self._make_data_linopy(N)
        timings["data"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        m = linopy.Model()
        x = m.add_variables(lower=0, coords=[i_coords, j_coords], name="x", mask=mask)
        m.add_constraints(x.sum("j") <= 1, name="cap")
        m.add_objective((cost_da * x).sum())
        timings["build"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        m.solve(solver_name="gurobi", io_api="direct", OutputFlag=0)
        timings["solve_wall"] = time.perf_counter() - t0
        timings["gurobi_runtime"] = m.solver_model.Runtime

        t0 = time.perf_counter()
        _ = x.solution
        obj_val = m.objective.value
        timings["solution"] = time.perf_counter() - t0

        timings["overhead"] = (
            timings["build"] + timings["solve_wall"] + timings["solution"]
            - timings["gurobi_runtime"]
        )
        return timings, obj_val


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
PROBLEMS: dict[str, Problem] = {p.name: p for p in [Dense2D(), SparseNetwork()]}

DEFAULT_SIZES = [10, 50, 100, 200, 500, 1000]


# ---------------------------------------------------------------------------
# Performance benchmark runner
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
# Memory benchmark (subprocess-isolated)
# ---------------------------------------------------------------------------

def _get_rss_mb():
    """Current RSS of this process in MB (works on macOS and Linux)."""
    # ps reports RSS in KB on both macOS and Linux
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return int(out.strip()) / 1024


_MEMORY_WORKER_SCRIPT = """
import gc, json, os, subprocess, sys

def get_rss_mb():
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return int(out.strip()) / 1024

# ---- imports (part of baseline) ----
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
import linopy
import pyoframe as pf

problem_name = sys.argv[1]
lib = sys.argv[2]
N = int(sys.argv[3])

# Reconstruct the problem
from benchmark_vs_linopy import PROBLEMS
problem = PROBLEMS[problem_name]

# Warmup
if lib == "pyoframe":
    problem.run_pyoframe(2)
else:
    problem.run_linopy(2)

# Measure
gc.collect()
rss_before = get_rss_mb()

if lib == "pyoframe":
    _, obj = problem.run_pyoframe(N)
else:
    _, obj = problem.run_linopy(N)

# Don't gc — measure what's alive after the full roundtrip
rss_after = get_rss_mb()

gc.collect()
rss_after_gc = get_rss_mb()

print(json.dumps({
    "rss_before": rss_before,
    "rss_after": rss_after,
    "rss_after_gc": rss_after_gc,
    "rss_delta": rss_after - rss_before,
    "rss_delta_gc": rss_after_gc - rss_before,
    "obj": obj,
}))
"""


def _run_memory_worker(problem_name: str, lib: str, N: int) -> dict:
    """Spawn a subprocess to measure memory for one (lib, N) combination."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        [sys.executable, "-c", _MEMORY_WORKER_SCRIPT, problem_name, lib, str(N)],
        capture_output=True,
        text=True,
        cwd=script_dir,
        env={**os.environ, "PYTHONPATH": script_dir},
    )
    if result.returncode != 0:
        print(f"  Worker failed ({lib} N={N}):")
        # Only print last few lines of stderr to avoid Gurobi license noise
        err_lines = result.stderr.strip().splitlines()
        for line in err_lines[-5:]:
            print(f"    {line}")
        return {}
    # The JSON is on the last non-empty line (skip Gurobi license output)
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    print(f"  Worker produced no JSON ({lib} N={N})")
    return {}


def run_memory_benchmarks(problem: Problem, sizes: list[int], num_runs: int = 3):
    """Run memory benchmarks in isolated subprocesses."""
    results = []

    for N in sizes:
        n_vars = problem.var_count(N)
        n_cons = problem.con_count(N)
        print(f"\nN = {N} ({n_vars:,} variables, {n_cons:,} constraints)")

        pf_deltas = []
        lp_deltas = []

        for run in range(num_runs):
            pf_result = _run_memory_worker(problem.name, "pyoframe", N)
            lp_result = _run_memory_worker(problem.name, "linopy", N)

            if pf_result and lp_result:
                pf_deltas.append(pf_result["rss_delta"])
                lp_deltas.append(lp_result["rss_delta"])
                print(
                    f"  Run {run + 1}/{num_runs}: "
                    f"pyoframe {pf_result['rss_delta']:+.1f} MB, "
                    f"linopy {lp_result['rss_delta']:+.1f} MB"
                )
            else:
                print(f"  Run {run + 1}/{num_runs}: measurement failed")

        if pf_deltas and lp_deltas:
            pf_med = statistics.median(pf_deltas)
            lp_med = statistics.median(lp_deltas)
            results.append({
                "N": N,
                "pyoframe_mb": pf_med,
                "linopy_mb": lp_med,
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


def fmt_mem(mb):
    if mb < 1:
        return f"{mb * 1024:>8.0f} KB"
    if mb < 1024:
        return f"{mb:>8.1f} MB"
    return f"{mb / 1024:>8.2f} GB"


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


def print_memory_results(df, problem: Problem):
    print(f"\n{'=' * 78}")
    print(f"  MEMORY — {problem.description}")
    print(f"  RSS delta = process RSS after roundtrip - before (median of runs)")
    print(f"{'=' * 78}")

    print(f"\n  {'N':>6}  {'pyoframe':>12}  {'linopy':>12}  {'pf / lp':>10}")
    print(f"  {'':->6}  {'':->12}  {'':->12}  {'':->10}")
    for _, r in df.iterrows():
        print(
            f"  {int(r['N']):>6}  {fmt_mem(r['pyoframe_mb'])}  {fmt_mem(r['linopy_mb'])}  {r['ratio']:>10.2f}x"
        )

    print(f"\n  {'─' * 74}")
    print(f"  ratio < 1 → pyoframe uses less memory")
    print(f"  {'─' * 74}")


def plot_results(df, problem: Problem, mem_df=None):
    import matplotlib.pyplot as plt

    n_plots = 3 if mem_df is None else 4
    phases = ["overhead", "data", "gurobi_runtime"]
    labels = ["Modeling overhead", "Data creation", "Gurobi solve"]
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))

    for ax, phase, label in zip(axes, phases, labels):
        rows = df[df["phase"] == phase]
        ax.loglog(rows["N"], rows["pyoframe"], "o-", label="pyoframe", color="tab:blue")
        ax.loglog(rows["N"], rows["linopy"], "s-", label="linopy", color="tab:orange")
        ax.set_title(label)
        ax.set_xlabel("N")
        ax.set_ylabel("Time (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if mem_df is not None:
        ax = axes[3]
        ax.loglog(mem_df["N"], mem_df["pyoframe_mb"], "o-", label="pyoframe", color="tab:blue")
        ax.loglog(mem_df["N"], mem_df["linopy_mb"], "s-", label="linopy", color="tab:orange")
        ax.set_title("Memory (RSS delta)")
        ax.set_xlabel("N")
        ax.set_ylabel("MB")
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
        "--problem", choices=list(PROBLEMS), default="dense_2d",
        help="Problem to benchmark (default: dense_2d)",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
        help=f"Problem sizes N (default: {DEFAULT_SIZES})",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per size (default: 3)")
    parser.add_argument("--memory", action="store_true", help="Run memory benchmark (separate pass)")
    parser.add_argument("--plot", action="store_true", help="Show log-log plots")
    parser.add_argument("--csv", type=str, default=None, help="Save results to CSV")
    args = parser.parse_args()

    problem = PROBLEMS[args.problem]
    print(f"Benchmark: pyoframe vs linopy — {problem.description}")
    print(f"Sizes: {args.sizes}, Runs: {args.runs}, Solver: Gurobi (direct)")

    # Performance
    perf_df = run_benchmarks(problem, args.sizes, args.runs)
    print_results(perf_df, problem)

    # Memory (separate pass)
    mem_df = None
    if args.memory:
        print(f"\n{'─' * 78}")
        print("Running memory benchmarks (isolated subprocesses)...")
        mem_df = run_memory_benchmarks(problem, args.sizes, args.runs)
        print_memory_results(mem_df, problem)

    if args.csv:
        perf_df.to_csv(args.csv, index=False)
        if mem_df is not None:
            mem_csv = args.csv.replace(".csv", "_memory.csv")
            mem_df.to_csv(mem_csv, index=False)
            print(f"\nResults saved to {args.csv} and {mem_csv}")
        else:
            print(f"\nResults saved to {args.csv}")

    if args.plot:
        plot_results(perf_df, problem, mem_df)


if __name__ == "__main__":
    main()
