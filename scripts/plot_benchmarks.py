"""Plot benchmark results from CSV files produced by benchmark_vs_linopy.py."""

import pandas as pd
import matplotlib.pyplot as plt

BENCHMARKS = {
    "dense_2d": {
        "perf": "scripts/benchmark_dense_2d.csv",
        "mem": "scripts/benchmark_dense_2d_memory.csv",
        "title": "Dense 2D (2N² vars)",
    },
    "sparse_network": {
        "perf": "scripts/benchmark_sparse_network.csv",
        "mem": "scripts/benchmark_sparse_network_memory.csv",
        "title": "Sparse network (K·N vars on N² grid)",
    },
    "many_small_vars": {
        "perf": "scripts/benchmark_many_small_vars.csv",
        "mem": "scripts/benchmark_many_small_vars_memory.csv",
        "title": "Many scalar variables (N vars)",
    },
}

fig, axes = plt.subplots(len(BENCHMARKS), 2, figsize=(10, 4 * len(BENCHMARKS)))

for row, (name, paths) in enumerate(BENCHMARKS.items()):
    perf = pd.read_csv(paths["perf"])
    mem = pd.read_csv(paths["mem"])

    # -- Overhead plot --
    ax = axes[row, 0]
    oh = perf[perf["phase"] == "overhead"]
    ax.loglog(oh["N"], oh["pyoframe"], "o-", label="pyoframe", color="tab:blue")
    ax.loglog(oh["N"], oh["linopy"], "s-", label="linopy", color="tab:orange")
    ax.set_title(f"{paths['title']} — Modeling overhead")
    ax.set_xlabel("N")
    ax.set_ylabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add ratio annotations
    for _, r in oh.iterrows():
        ax.annotate(
            f"{r['ratio']:.2f}x",
            (r["N"], r["pyoframe"]),
            textcoords="offset points",
            xytext=(0, -14),
            fontsize=7,
            ha="center",
            color="tab:blue",
        )

    # -- Memory plot --
    ax = axes[row, 1]
    ax.loglog(mem["N"], mem["pyoframe_mb"], "o-", label="pyoframe", color="tab:blue")
    ax.loglog(mem["N"], mem["linopy_mb"], "s-", label="linopy", color="tab:orange")
    ax.set_title(f"{paths['title']} — Memory (RSS delta)")
    ax.set_xlabel("N")
    ax.set_ylabel("MB")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for _, r in mem.iterrows():
        ax.annotate(
            f"{r['ratio']:.2f}x",
            (r["N"], r["pyoframe_mb"]),
            textcoords="offset points",
            xytext=(0, -14),
            fontsize=7,
            ha="center",
            color="tab:blue",
        )

fig.suptitle("pyoframe vs linopy — benchmarks", fontsize=14, fontweight="bold")
fig.tight_layout()
out = "scripts/benchmark_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
