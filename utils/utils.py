"""
Simulation and analysis utilities for a 2-component radial reaction-diffusion model.

Model summary:
- Radial domain [0, r_b] with cylindrical symmetry
- Promoter c_p and inhibitor c_i diffusing with coefficients D_p and D_i
- Constant inward flux at r = 0: F_p = flp / (2 * pi * k), F_i = fl1 / (2 * pi * k)
- Reflective (Neumann, zero-flux) boundary condition at r = r_b
- Mineral accumulation c_m when c_p > c_i up to c_max
"""

import os
import json
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import numba
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:
    yaml = None


def expo1(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Sigmoidal curve fitting function used for mineralization front analysis.
    Formula: |a| / (1 + exp(|b| * (c - x)))
    """
    return np.abs(a) / (1.0 + np.exp(np.abs(b) * (c - x)))


# ==============================================================================
# Numerical Integration Kernels (Numba JIT)
# ==============================================================================

@numba.jit(nopython=True)
def step_kernel(
    rad: int,
    Fp: float,
    Fi: float,
    c_max: float,
    c_p0: np.ndarray,
    c_i0: np.ndarray,
    c_p: np.ndarray,
    c_m: np.ndarray,
    c_i: np.ndarray,
    D_i: float,
    D_p: float,
    h: float,
    k: float,
) -> None:
    """
    Optimized 1D in-place integration step for the radial reaction-diffusion model.
    Updates c_p, c_i, and c_m in-place, taking c_p0 and c_i0 as previous concentrations.
    """
    # 1. Update concentrations at the inner boundary r=0 (index 0) due to flux
    c_p0[0] = c_p0[1] + Fp * k / D_p
    c_i0[0] = c_i0[1] + Fi * k / D_i

    inv_k2 = 1.0 / (k * k)
    coeff_i = h * D_i * inv_k2
    coeff_p = h * D_p * inv_k2

    # 2. Iterate over internal grid points (1 to rad - 2)
    for i in range(1, rad - 1):
        inv_2i = 1.0 / (2.0 * i)
        # Radial diffusion: d^2c/dr^2 + (1/r)*dc/dr
        c_i[i] = c_i0[i] + coeff_i * (
            (1.0 + inv_2i) * c_i0[i + 1] - 2.0 * c_i0[i] + (1.0 - inv_2i) * c_i0[i - 1]
        )
        c_p[i] = c_p0[i] + coeff_p * (
            (1.0 + inv_2i) * c_p0[i + 1] - 2.0 * c_p0[i] + (1.0 - inv_2i) * c_p0[i - 1]
        )

        # Reaction / mineral deposition
        if c_m[i] <= c_max:
            diff = c_p[i] - c_i[i]
            if diff > 0.0:
                c_m[i] += diff * h
                c_p[i] -= diff * h

    # 3. Reflective boundary condition at outer boundary r = r_b (index rad - 1)
    c_i[rad - 1] = c_i[rad - 2]
    c_p[rad - 1] = c_p[rad - 2]

    # Outer boundary reaction
    i = rad - 1
    if c_m[i] <= c_max:
        diff = c_p[i] - c_i[i]
        if diff > 0.0:
            c_m[i] += diff * h
            c_p[i] -= diff * h


@numba.jit(nopython=True)
def step_original(
    j: int,
    rad: int,
    Fp: float,
    Fi: float,
    c_max: float,
    c_p0: np.ndarray,
    c_i0: np.ndarray,
    c_p: np.ndarray,
    c_m1: np.ndarray,
    c_i: np.ndarray,
    D_i: float,
    D_p: float,
    h: float,
    k: float,
) -> None:
    """
    Original 2D-array implementation of the step function from the notebook.
    Kept for backwards-compatibility with existing notebook code.
    """
    c_p0[0] = c_p0[1] + Fp * k / D_p
    c_i0[0] = c_i0[1] + Fi * k / D_i

    for i in range(1, rad - 1):
        c_i[i] = c_i0[i] + h * D_i / (k * k) * (
            (1.0 + 1.0 / (2.0 * i)) * c_i0[i + 1]
            - 2.0 * c_i0[i]
            + (1.0 - 1.0 / (2.0 * i)) * c_i0[i - 1]
        )
        c_p[i] = c_p0[i] + h * D_p / (k * k) * (
            (1.0 + 1.0 / (2.0 * i)) * c_p0[i + 1]
            - 2.0 * c_p0[i]
            + (1.0 - 1.0 / (2.0 * i)) * c_p0[i - 1]
        )

        c_m1[i, j + 1] = c_m1[i, j]
        if c_m1[i, j] <= c_max:
            if c_p[i] > c_i[i]:
                c_m1[i, j + 1] += (c_p[i] - c_i[i]) * h
                c_p[i] -= (c_p[i] - c_i[i]) * h

    c_i[rad - 1] = c_i[rad - 2]
    c_p[rad - 1] = c_p[rad - 2]
    i = rad - 1

    c_m1[i, j + 1] = c_m1[i, j]
    if c_m1[i, j] <= c_max:
        if c_p[i] > c_i[i]:
            c_m1[i, j + 1] += (c_p[i] - c_i[i]) * h
            c_p[i] -= (c_p[i] - c_i[i]) * h


@numba.jit(nopython=True)
def compute_total_mineral(c_m: np.ndarray, rad: int, k: float) -> float:
    """
    Computes total integrated mineral content over the circular domain:
    Integral of 2 * pi * r * c_m(r) dr discretized as:
    sum_{i=0}^{rad-1} k^2 * (-i^2 + (i+1)^2) * pi * c_m[i]
    """
    tot = 0.0
    for i in range(rad):
        # (-i^2 + (i+1)^2) = 2*i + 1
        tot += (k * k) * (-i * i + (i + 1) * (i + 1)) * np.pi * c_m[i]
    return tot


# ==============================================================================
# Configuration Loading & Parsing
# ==============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads configuration from YAML or JSON file with default fallbacks.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith((".yaml", ".yml")):
            if yaml is None:
                raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)

    # Defaults
    defaults = {
        "physics": {
            "D_p": 0.45,
            "D_i": 0.08,
            "flp": 1.2,
            "fl1": 0.8,
            "c_max": 1.0,
        },
        "grid": {
            "k": 0.01,
            "r_max": 1.0,
            "h_factor": 0.45,
        },
        "simulation": {
            "max_timesteps": 10000001,
            "eps": 0.001,
            "save_interval": 10000,
            "track_somma_interval": 100,
        },
        "output": {
            "output_dir": "output",
            "save_plots": True,
            "plot_profiles_count": 20,
        },
    }

    # Merge top-level sections
    for section, values in defaults.items():
        if section not in cfg:
            cfg[section] = values
        else:
            for k, v in values.items():
                if k not in cfg[section]:
                    cfg[section][k] = v

    return cfg


# ==============================================================================
# Simulation Runner
# ==============================================================================

def run_simulation(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Executes the reaction-diffusion simulation based on the given configuration.
    """
    phys = config["physics"]
    grid = config["grid"]
    sim = config["simulation"]

    D_p = float(phys["D_p"])
    D_i = float(phys["D_i"])
    flp = float(phys.get("F_p", phys.get("flp", 1.2)))
    fl1 = float(phys.get("F_i", phys.get("fl1", 0.8)))
    c_max = float(phys["c_max"])

    k = float(grid["k"])
    r_max = float(grid.get("r_max", 1.0))
    h_factor = float(grid.get("h_factor", 0.45))
    h = h_factor * (k ** 2) / D_p

    rad = int(r_max / k)
    max_timesteps = int(sim["max_timesteps"])
    eps = float(sim["eps"])
    save_interval = max(1, int(sim.get("save_interval", 10000)))
    somma_interval = max(1, int(sim.get("track_somma_interval", 100)))

    Fi = fl1 / (2.0 * np.pi * k)
    Fp = flp / (2.0 * np.pi * k)

    half_capacity = (np.pi * ((rad * k) ** 2)) / 2.0

    if verbose:
        print("=" * 60)
        print("Starting Radial Symmetric Reaction-Diffusion Simulation")
        print(f"Parameters: D_p={D_p}, D_i={D_i}, flp={flp}, fl1={fl1}")
        print(f"Grid: rad={rad} (r_max={r_max}, k={k}), dt (h)={h:.6e}")
        print(f"Max Timesteps: {max_timesteps}, Snapshot interval: {save_interval}")
        print("=" * 60)

    # Initial state arrays
    c_i = np.zeros(rad, dtype=np.float64)
    c_p = np.zeros(rad, dtype=np.float64)
    c_i0 = np.zeros(rad, dtype=np.float64)
    c_p0 = np.zeros(rad, dtype=np.float64)
    c_m = np.zeros(rad, dtype=np.float64)

    # Pre-allocate snapshot tracking
    snapshots_cm = []
    snapshots_step = []
    snapshots_time = []

    somma_history = []
    somma_steps = []
    somma_time = []

    endtime = 0.0
    zero_time = 0.0
    zero_found = False

    t_start = time.time()
    last_print = t_start

    # Initial snapshot at step 0
    snapshots_cm.append(c_m.copy())
    snapshots_step.append(0)
    snapshots_time.append(0.0)

    tot0 = compute_total_mineral(c_m, rad, k)
    somma_history.append(tot0)
    somma_steps.append(0)
    somma_time.append(0.0)

    final_step = 0

    for j in range(0, max_timesteps - 1):
        final_step = j + 1
        c_i0[:] = c_i[:]
        c_p0[:] = c_p[:]

        # Numerical step
        step_kernel(rad, Fp, Fi, c_max, c_p0, c_i0, c_p, c_m, c_i, D_i, D_p, h, k)

        sim_time_days = j * h / 2.0

        # Periodic tracking of total mineral
        if j % somma_interval == 0:
            tot = compute_total_mineral(c_m, rad, k)
            somma_history.append(tot)
            somma_steps.append(j)
            somma_time.append(sim_time_days)

            # Check zero condition (reaching half maximum total capacity)
            if not zero_found:
                if abs(tot - half_capacity) <= 0.0001 or tot >= half_capacity:
                    zero_time = sim_time_days
                    zero_found = True

        # Periodic snapshot recording
        if j % save_interval == 0 and j > 0:
            snapshots_cm.append(c_m.copy())
            snapshots_step.append(j)
            snapshots_time.append(sim_time_days)

        # Periodic console logging
        now = time.time()
        if verbose and (now - last_print > 3.0 or j == 1000):
            print(
                f"  Step {j:,}/{max_timesteps:,} (t={sim_time_days:.3f} days) | "
                f"c_m range: [{c_m[1:-1].min():.3f}, {c_m[1:-1].max():.3f}]"
            )
            last_print = now

        # Check saturation termination (endtime condition from notebook)
        if np.all(c_m[1:-1] > (c_max - eps)):
            endtime = sim_time_days
            if verbose:
                print(f"--> Saturation threshold reached at step {j:,}! Endtime = {endtime:.4f} days")
            break

    # Final snapshot
    sim_time_days = final_step * h / 2.0
    if len(snapshots_step) == 0 or snapshots_step[-1] != final_step:
        snapshots_cm.append(c_m.copy())
        snapshots_step.append(final_step)
        snapshots_time.append(sim_time_days)

    tot_final = compute_total_mineral(c_m, rad, k)
    if len(somma_steps) == 0 or somma_steps[-1] != final_step:
        somma_history.append(tot_final)
        somma_steps.append(final_step)
        somma_time.append(sim_time_days)

    wall_time = time.time() - t_start

    if verbose:
        print("-" * 60)
        print(f"Simulation completed in {wall_time:.2f} s")
        print(f"Final step: {final_step:,}, Final simulated time: {sim_time_days:.4f} days")
        print(f"Endtime: {endtime:.4f} days, Zero time: {zero_time:.4f} days")
        print("=" * 60)

    # Radial coordinate values: distance from origin r
    r_coords = np.arange(rad) * k
    # Normalized radial distance 2 * r / r_max (as plotted in original notebook: 2 * arange(rad-1) / rad)
    r_normalized = 2.0 * np.arange(rad) / rad

    results = {
        "final_step": int(final_step),
        "wall_time_seconds": float(wall_time),
        "endtime": float(endtime),
        "zero_time": float(zero_time),
        "h": float(h),
        "k": float(k),
        "rad": int(rad),
        "Fi": float(Fi),
        "Fp": float(Fp),
        "half_capacity": float(half_capacity),
        "r_coords": r_coords,
        "r_normalized": r_normalized,
        "snapshots_cm": np.array(snapshots_cm),  # shape: (n_snapshots, rad)
        "snapshots_step": np.array(snapshots_step),
        "snapshots_time": np.array(snapshots_time),
        "somma_history": np.array(somma_history),
        "somma_steps": np.array(somma_steps),
        "somma_time": np.array(somma_time),
        "final_c_p": c_p,
        "final_c_i": c_i,
        "final_c_m": c_m,
    }

    return results


# ==============================================================================
# Saving Outputs and Metadata
# ==============================================================================

def save_simulation_output(
    output_dir: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
    run_name: Optional[str] = None,
) -> str:
    """
    Saves simulation data as CSV files with metadata header comments,
    compressed NumPy archive (.npz), and metadata JSON into the output directory.
    Returns the path to the created run directory.
    """
    if run_name is None:
        run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    phys = config["physics"]
    grid = config["grid"]

    D_i = phys["D_i"]
    D_p = phys["D_p"]
    F_i = phys.get("F_i", phys.get("fl1"))
    F_p = phys.get("F_p", phys.get("flp"))
    r_max = grid.get("r_max", 1.0)

    # Required comment line at the top of each CSV
    header_comment = f"# D_i: {D_i}\tD_p: {D_p}\tF_i: {F_i}\tF_p: {F_p}\tr_max: {r_max}\n"

    def write_csv_with_comment(filepath: str, df: pd.DataFrame):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header_comment)
            df.to_csv(f, index=False)

    # 1. Mineral concentration profiles (radial points vs time snapshots)
    snapshots_cm = results["snapshots_cm"]  # shape: (n_snapshots, rad)
    snapshots_time = results["snapshots_time"]
    r_coords = results["r_coords"]
    r_normalized = results["r_normalized"]

    profiles_dict = {
        "r_um": r_coords,
        "r_normalized": r_normalized,
    }
    for idx, t_day in enumerate(snapshots_time):
        col_name = f"t_{t_day:.3f}_days"
        profiles_dict[col_name] = snapshots_cm[idx]

    df_profiles = pd.DataFrame(profiles_dict)
    csv_profiles_path = os.path.join(run_dir, "mineral_profiles.csv")
    write_csv_with_comment(csv_profiles_path, df_profiles)

    # 2. Total mineral vs time
    df_somma = pd.DataFrame({
        "step": results["somma_steps"],
        "time_days": results["somma_time"],
        "total_mineral": results["somma_history"],
        "total_mineral_over_pi": results["somma_history"] / np.pi,
    })
    csv_somma_path = os.path.join(run_dir, "total_mineral_vs_time.csv")
    write_csv_with_comment(csv_somma_path, df_somma)

    # 3. Final state profiles of all components
    df_final = pd.DataFrame({
        "r_um": r_coords,
        "r_normalized": r_normalized,
        "promoter_c_p": results["final_c_p"],
        "inhibitor_c_i": results["final_c_i"],
        "mineral_c_m": results["final_c_m"],
    })
    csv_final_path = os.path.join(run_dir, "final_profiles.csv")
    write_csv_with_comment(csv_final_path, df_final)

    # 4. Simulation summary record
    df_summary = pd.DataFrame([{
        "D_i": D_i,
        "D_p": D_p,
        "F_i": F_i,
        "F_p": F_p,
        "r_max": r_max,
        "final_step": results["final_step"],
        "simulated_days": float(results["snapshots_time"][-1]),
        "endtime": results["endtime"],
        "zero_time": results["zero_time"],
        "wall_time_seconds": results["wall_time_seconds"],
    }])
    csv_summary_path = os.path.join(run_dir, "simulation_summary.csv")
    write_csv_with_comment(csv_summary_path, df_summary)

    # 5. Metadata JSON
    metadata = {
        "run_name": run_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_time_seconds": results["wall_time_seconds"],
        "final_step": results["final_step"],
        "simulated_days": float(results["snapshots_time"][-1]),
        "endtime": results["endtime"],
        "zero_time": results["zero_time"],
        "parameters": {
            "physics": config["physics"],
            "grid": config["grid"],
            "simulation": config["simulation"],
        },
        "computed_constants": {
            "h": results["h"],
            "k": results["k"],
            "rad": results["rad"],
            "Fi": results["Fi"],
            "Fp": results["Fp"],
            "half_capacity": results["half_capacity"],
        },
        "output_files": {
            "mineral_profiles_csv": "mineral_profiles.csv",
            "total_mineral_csv": "total_mineral_vs_time.csv",
            "final_profiles_csv": "final_profiles.csv",
            "simulation_summary_csv": "simulation_summary.csv",
            "simulation_data_npz": "simulation_data.npz",
            "metadata_json": "metadata.json",
            "mineral_profiles_plot": "mineral_profiles.png",
            "total_mineral_plot": "total_mineral_vs_time.png",
        },
    }

    metadata_path = os.path.join(run_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # 6. Compressed NumPy arrays
    data_path = os.path.join(run_dir, "simulation_data.npz")
    np.savez_compressed(
        data_path,
        r_coords=results["r_coords"],
        r_normalized=results["r_normalized"],
        snapshots_cm=results["snapshots_cm"],
        snapshots_step=results["snapshots_step"],
        snapshots_time=results["snapshots_time"],
        somma_history=results["somma_history"],
        somma_steps=results["somma_steps"],
        somma_time=results["somma_time"],
        final_c_p=results["final_c_p"],
        final_c_i=results["final_c_i"],
        final_c_m=results["final_c_m"],
    )

    print(f"Saved CSV outputs with top comment:\n  {header_comment.strip()}")
    print(f"  - {csv_profiles_path}")
    print(f"  - {csv_somma_path}")
    print(f"  - {csv_final_path}")
    print(f"  - {csv_summary_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved simulation arrays: {data_path}")

    return run_dir


# ==============================================================================
# Visualization Utilities
# ==============================================================================

def plot_simulation_results(
    results: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    show: bool = False,
) -> Tuple[str, str]:
    """
    Generates and saves the two key diagnostic plots from the notebook:
    1. Mineral concentration profiles vs distance from canaliculus at various times.
    2. Total integrated mineral content vs time.
    """
    phys = config["physics"]
    plot_count = config["output"].get("plot_profiles_count", 20)

    snapshots_cm = results["snapshots_cm"]
    snapshots_time = results["snapshots_time"]
    r_norm = results["r_normalized"]
    somma_history = results["somma_history"]
    somma_time = results["somma_time"]

    num_snapshots = len(snapshots_cm)
    indices = np.linspace(0, num_snapshots - 1, min(plot_count, num_snapshots), dtype=int)
    indices = np.unique(indices)

    # --------------------------------------------------------------------------
    # Plot 1: Mineral concentration profiles (Cell 3 / Cell 4)
    # --------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(indices)))

    for idx, col in zip(indices, colors):
        t_day = snapshots_time[idx]
        ax1.plot(
            r_norm[:-1],
            snapshots_cm[idx, :-1],
            color=col,
            label=f"t={t_day:.2f} d",
        )

    ax1.set_xlabel(r"Distance from the canaliculus $r$ ($\mu m$)", fontsize=13)
    ax1.set_ylabel("Mineral concentration $c_m(r)$", fontsize=13)
    ax1.set_title(
        f"Mineralization Profiles\nflp={phys['flp']}, fli={phys['fl1']}, "
        f"Dp={phys['D_p']}, Di={phys['D_i']}",
        fontsize=14,
    )
    ax1.grid(True, linestyle="--", alpha=0.5)
    if len(indices) <= 25:
        ax1.legend(loc="upper right", fontsize=9, ncol=2)
    fig1.tight_layout()

    # --------------------------------------------------------------------------
    # Plot 2: Total mineral vs time (Cell 5)
    # --------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(somma_time, somma_history / np.pi, color="navy", lw=2)
    ax2.set_xlabel("Time (days)", fontsize=13)
    ax2.set_ylabel(r"Total mineral / $\pi$", fontsize=13)
    ax2.set_title(
        f"Total Mineral Accumulation\nflp={phys['flp']}, fli={phys['fl1']}, "
        f"Dp={phys['D_p']}, Di={phys['D_i']}",
        fontsize=14,
    )
    if results["zero_time"] > 0:
        ax2.axvline(
            results["zero_time"],
            color="red",
            linestyle="--",
            label=f"Half capacity (t={results['zero_time']:.2f} d)",
        )
    if results["endtime"] > 0:
        ax2.axvline(
            results["endtime"],
            color="green",
            linestyle=":",
            label=f"Full saturation (t={results['endtime']:.2f} d)",
        )
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=11)
    fig2.tight_layout()

    plot1_path = ""
    plot2_path = ""
    if output_dir:
        plot1_path = os.path.join(output_dir, "mineral_profiles.png")
        plot2_path = os.path.join(output_dir, "total_mineral_vs_time.png")
        fig1.savefig(plot1_path, dpi=300)
        fig2.savefig(plot2_path, dpi=300)
        print(f"Saved plot: {plot1_path}")
        print(f"Saved plot: {plot2_path}")

    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

    return plot1_path, plot2_path

