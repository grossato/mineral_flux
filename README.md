# 2-Component Radial Symmetric Reaction-Diffusion Model

This project implements a modularized and optimized simulation of a radial symmetric reaction-diffusion model with two components:
- **Promoter** ($c_p$) with diffusion coefficient $D_p$
- **Inhibitor** ($c_i$) with diffusion coefficient $D_i$
- **Mineral phase** ($c_m$) formed when $c_p > c_i$ up to saturation threshold $c_{max}$

## Boundary Conditions
1. **At $r = 0$ (Inner boundary / canaliculus):**
   Constant flux boundary condition:
   $$F_p = \frac{\text{flp}}{2\pi k}, \quad F_i = \frac{\text{fl1}}{2\pi k}$$
   Discretized as:
   $$c_p(0) = c_p(1) + \frac{F_p \cdot k}{D_p}, \quad c_i(0) = c_i(1) + \frac{F_i \cdot k}{D_i}$$

2. **At $r = r_b$ (Outer boundary):**
   Reflective (zero-flux / Neumann) boundary condition:
   $$\left.\frac{\partial c}{\partial r}\right|_{r = r_b} = 0 \implies c_p(\text{rad}-1) = c_p(\text{rad}-2), \quad c_i(\text{rad}-1) = c_i(\text{rad}-2)$$

---

## Project Structure

```
mineral_flux/
├── config.yaml               # Simulation, grid, and physical parameters
├── main.py                   # Main CLI script to run the simulation and save outputs
├── utils/                    # Utility package
│   ├── __init__.py           # Package exports
│   └── utils.py              # Numerical Numba kernels, runner, I/O, plotting
├── output/                   # Directory where simulation outputs are saved
│   └── run_YYYYMMDD_HHMMSS/  # Timestamped simulation run directory
│       ├── metadata.json     # Configuration and execution metadata
│       ├── simulation_data.npz # Compressed arrays (snapshots, somma, final fields)
│       ├── mineral_profiles.png # Mineral profile evolution plot
│       └── total_mineral_vs_time.png # Total mineral vs time plot
└── README.md                 # Project documentation
```

---

## Configuration (`config.yaml`)

Edit [`config.yaml`](file:///c:/Users/giaco/Desktop/reformat_flux/config.yaml) to customize parameters:

```yaml
physics:
  D_p: 0.45       # Diffusion coefficient of promoter
  D_i: 0.08       # Diffusion coefficient of inhibitor
  flp: 1.2        # Flux parameter for promoter at r=0
  fl1: 0.8        # Flux parameter for inhibitor at r=0
  c_max: 1.0      # Maximum mineral saturation concentration

grid:
  k: 0.01         # Spatial grid step dr
  r_max: 1.0      # Outer boundary radius r_b (rad = int(r_max / k))
  h_factor: 0.45  # dt factor: h = h_factor * k^2 / D_p

simulation:
  max_timesteps: 10000001
  eps: 0.001       # Tolerance for endtime saturation check
  save_interval: 10000     # Snapshot step interval
  track_somma_interval: 100 # Total mineral calculation interval

output:
  output_dir: "output"
  save_plots: true
  plot_profiles_count: 20
```

---

## Running the Simulation

Execute the main script from the command line:

```bash
python main.py
```

### Custom Options:
```bash
# Specify custom config file and output directory
python main.py --config custom_config.yaml --output-dir my_output

# Run with a custom run folder name
python main.py --run-name trial_1

# Skip saving plot figures
python main.py --no-plots
```

---

## Output Files

Each execution creates a subfolder in `output/` containing:

### CSV Data Outputs (with top comment `# D_i: ...\tD_p: ...\tF_i: ...\tF_p: ...\tr_max: ...`):
1. **`mineral_profiles.csv`**:
   - Radial distribution of mineral concentration $c_m(r)$ at each snapshot time.
   - Rows: radial distance ($r_{\mu m}$ and $r_{\text{normalized}}$).
   - Columns: concentration profile at each recorded snapshot time $t$ (days).
2. **`total_mineral_vs_time.csv`**:
   - Time series of total integrated mineral content over the domain (`step`, `time_days`, `total_mineral`, `total_mineral_over_pi`).
3. **`final_profiles.csv`**:
   - Radial profile of all components at simulation termination (`r_um`, `r_normalized`, `promoter_c_p`, `inhibitor_c_i`, `mineral_c_m`).
4. **`simulation_summary.csv`**:
   - Summary table with parameter values, transition times (`zero_time` to half capacity, `endtime` to saturation), total steps, and wall time.

### Metadata & Archives:
5. **`metadata.json`**: Complete JSON record of configuration parameters, numerical constants ($h, k, \text{rad}, F_p, F_i$), and run statistics.
6. **`simulation_data.npz`**: Compressed NumPy archive of all spatial arrays and snapshots for fast numerical loading.
7. **`mineral_profiles.png`**: Evolution of radial mineral concentration profiles over time.
8. **`total_mineral_vs_time.png`**: Total accumulated mineral vs time in days with markers for half capacity and full saturation.

---

## Using Utilities in Jupyter Notebooks / Scripts

You can also import and use the utilities directly in Python or Jupyter notebooks:

```python
from utils import load_config, run_simulation, plot_simulation_results

# Load configuration
config = load_config("config.yaml")

# Run simulation
results = run_simulation(config)

# Plot or access results
print("Endtime (days):", results["endtime"])
print("Zero time (days):", results["zero_time"])
```

