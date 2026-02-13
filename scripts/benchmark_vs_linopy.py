"""
Benchmark: pyoframe vs linopy — modeling overhead

Measures the full roundtrip from model creation to solution retrieval,
then subtracts Gurobi's self-reported solver time (RunTime) to isolate
pure modeling overhead: data structure creation, solver communication,
and solution extraction.

Problem (linopy's standard benchmark LP):
  - Variables: x[i,j] and y[i,j] for i,j in range(N) → 2*N² variables
  - Constraints: x - y >= i (broadcast over j), x + y >= 0 → 2*N² constraints
  - Objective: minimize 2*sum(x) + sum(y)
  - Solver: Gurobi (direct API for both)

Usage:
    python scripts/benchmark_vs_linopy.py
    python scripts/benchmark_vs_linopy.py --sizes 10 50 100 200 500 --runs 3
    python scripts/benchmark_vs_linopy.py --plot
    python scripts/benchmark_vs_linopy.py --csv results.csv
"""

import argparse
import gc
import statistics
import time

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

import linopy
import pyoframe as pf


def benchmark_pyoframe(N):
    """Build and solve the benchmark LP using pyoframe + gurobi."""
    timings = {}

    # Build: model + variables + constraints + objective
    t0 = time.perf_counter()
    m = pf.Model("gurobi")
    m.attr.Silent = True
    m.x = pf.Variable(pf.Set(i=range(N), j=range(N)))
    m.y = pf.Variable(pf.Set(i=range(N), j=range(N)))
    timings["variables"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rhs = pf.Param({"i": range(N), "rhs": range(N)})
    m.c1 = m.x - m.y >= rhs.over("j")
    m.c2 = m.x + m.y >= 0
    timings["constraints"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    m.minimize = 2 * m.x.sum() + m.y.sum()
    timings["objective"] = time.perf_counter() - t0

    # Solve
    t0 = time.perf_counter()
    m.optimize()
    timings["optimize_wall"] = time.perf_counter() - t0
    timings["gurobi_runtime"] = m.attr.SolveTimeSec

    # Solution extraction
    t0 = time.perf_counter()
    _ = m.x.solution
    _ = m.y.solution
    obj_val = m.minimize.value
    timings["solution"] = time.perf_counter() - t0

    timings["build"] = timings["variables"] + timings["constraints"] + timings["objective"]
    timings["total_wall"] = timings["build"] + timings["optimize_wall"] + timings["solution"]
    timings["overhead"] = timings["total_wall"] - timings["gurobi_runtime"]

    return timings, obj_val


def benchmark_linopy(N):
    """Build and solve the benchmark LP using linopy + gurobi."""
    timings = {}

    # Build: model + variables + constraints + objective
    t0 = time.perf_counter()
    m = linopy.Model()
    i_coords = pd.Index(range(N), name="i")
    j_coords = pd.Index(range(N), name="j")
    x = m.add_variables(coords=[i_coords, j_coords], name="x")
    y = m.add_variables(coords=[i_coords, j_coords], name="y")
    timings["variables"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rhs = xr.DataArray(np.arange(N), dims=["i"], coords={"i": i_coords})
    m.add_constraints(x - y >= rhs, name="c1")
    m.add_constraints(x + y >= 0, name="c2")
    timings["constraints"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    m.add_objective(2 * x.sum() + y.sum())
    timings["objective"] = time.perf_counter() - t0

    # Solve (linopy bundles solver communication + solve + solution extraction here)
    t0 = time.perf_counter()
    m.solve(solver_name="gurobi", io_api="direct", OutputFlag=0)
    timings["optimize_wall"] = time.perf_counter() - t0
    timings["gurobi_runtime"] = m.solver_model.Runtime

    # Solution access (already extracted during solve, so near-zero)
    t0 = time.perf_counter()
    _ = x.solution
    _ = y.solution
    obj_val = m.objective.value
    timings["solution"] = time.perf_counter() - t0

    timings["build"] = timings["variables"] + timings["constraints"] + timings["objective"]
    timings["total_wall"] = timings["build"] + timings["optimize_wall"] + timings["solution"]
    timings["overhead"] = timings["total_wall"] - timings["gurobi_runtime"]

    return timings, obj_val


def run_benchmarks(sizes, num_runs=3):
    """Run both benchmarks across all sizes, returning a DataFrame of results."""

    print("Warming up...")
    benchmark_pyoframe(2)
    benchmark_linopy(2)
    gc.collect()

    results = []
    phases = [
        "variables",
        "constraints",
        "objective",
        "build",
        "optimize_wall",
        "gurobi_runtime",
        "solution",
        "total_wall",
        "overhead",
    ]

    for N in sizes:
        print(f"\nN = {N} ({2 * N**2} variables, {2 * N**2} constraints)")

        pf_runs = {phase: [] for phase in phases}
        lp_runs = {phase: [] for phase in phases}
        pf_obj = None
        lp_obj = None

        for run in range(num_runs):
            gc.collect()
            timings, obj = benchmark_pyoframe(N)
            pf_obj = obj
            for phase in phases:
                pf_runs[phase].append(timings[phase])

            gc.collect()
            timings, obj = benchmark_linopy(N)
            lp_obj = obj
            for phase in phases:
                lp_runs[phase].append(timings[phase])

            print(f"  Run {run + 1}/{num_runs} done")

        # Verify correctness
        if pf_obj is not None and lp_obj is not None:
            if abs(pf_obj - lp_obj) > 1e-4 * max(abs(pf_obj), abs(lp_obj), 1):
                print(
                    f"  WARNING: Objective mismatch! pyoframe={pf_obj:.6f}, linopy={lp_obj:.6f}"
                )
            else:
                print(f"  Objective: {pf_obj:.2f} (verified match)")

        for phase in phases:
            pf_median = statistics.median(pf_runs[phase])
            lp_median = statistics.median(lp_runs[phase])
            results.append(
                {
                    "N": N,
                    "phase": phase,
                    "pyoframe": pf_median,
                    "linopy": lp_median,
                    "ratio": pf_median / lp_median if lp_median > 0 else float("inf"),
                }
            )

    return pd.DataFrame(results)


def fmt_time(t):
    if t < 0.0001:
        return f"{t * 1e6:>9.1f} us"
    elif t < 0.1:
        return f"{t * 1000:>9.2f} ms"
    else:
        return f"{t:>9.4f}  s"


def print_results(df):
    """Print a formatted comparison table."""

    def print_phase(phase, label=None):
        phase_df = df[df["phase"] == phase]
        print(f"\n--- {label or phase} ---")
        print(f"{'N':>6}  {'pyoframe':>12}  {'linopy':>12}  {'ratio (pf/lp)':>14}")
        print(f"{'':->6}  {'':->12}  {'':->12}  {'':->14}")
        for _, row in phase_df.iterrows():
            print(
                f"{int(row['N']):>6}  {fmt_time(row['pyoframe'])}  {fmt_time(row['linopy'])}  {row['ratio']:>14.2f}x"
            )

    print("\n" + "=" * 90)
    print("MODELING OVERHEAD (total wall time minus Gurobi's self-reported RunTime)")
    print("  = build + solver communication + solution extraction, WITHOUT actual solve")
    print("=" * 90)

    print_phase("overhead", "OVERHEAD (total - gurobi solve)")
    print_phase("total_wall", "TOTAL (wall clock)")
    print_phase("gurobi_runtime", "GUROBI SOLVE (self-reported RunTime)")

    print("\n" + "=" * 90)
    print("DETAILED BREAKDOWN")
    print("=" * 90)

    for phase in ["variables", "constraints", "objective", "solution"]:
        print_phase(phase)

    print("\n" + "=" * 90)
    print("ratio < 1 means pyoframe is faster, ratio > 1 means linopy is faster")
    print("=" * 90)


def plot_results(df):
    """Create log-log scaling plots."""
    import matplotlib.pyplot as plt

    phases = ["overhead", "total_wall", "gurobi_runtime"]
    labels = ["Modeling overhead\n(total - gurobi)", "Total wall clock", "Gurobi solve time"]
    fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 4.5), sharey=False)

    for ax, phase, label in zip(axes, phases, labels):
        phase_df = df[df["phase"] == phase]
        ax.loglog(
            phase_df["N"], phase_df["pyoframe"], "o-", label="pyoframe", color="tab:blue"
        )
        ax.loglog(
            phase_df["N"], phase_df["linopy"], "s-", label="linopy", color="tab:orange"
        )
        ax.set_title(label)
        ax.set_xlabel("N")
        ax.set_ylabel("Time (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("pyoframe vs linopy: modeling overhead", fontsize=14)
    fig.tight_layout()
    plt.savefig("scripts/benchmark_vs_linopy.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to scripts/benchmark_vs_linopy.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Benchmark pyoframe vs linopy")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10, 50, 100, 200, 500],
        help="Problem sizes N (default: 10 50 100 200 500)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per size (default: 3)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate log-log scaling plots",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Save results to CSV file",
    )
    args = parser.parse_args()

    print(f"Benchmark: pyoframe vs linopy (modeling overhead)")
    print(f"Sizes: {args.sizes}, Runs: {args.runs}")
    print(f"Solver: Gurobi (direct API)")

    df = run_benchmarks(args.sizes, args.runs)
    print_results(df)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nResults saved to {args.csv}")

    if args.plot:
        plot_results(df)


if __name__ == "__main__":
    main()
