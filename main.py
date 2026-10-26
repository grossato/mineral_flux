"""
Main entry point for running the 2-component radial symmetric reaction-diffusion simulation.

Usage:
    python main.py
    python main.py --config config.yaml --output-dir output
"""

import argparse
import sys
import os

# Ensure local packages can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import (
    load_config,
    run_simulation,
    save_simulation_output,
    plot_simulation_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run radial symmetric reaction-diffusion simulation with constant flux at r=0 and reflective outer boundary."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML or JSON configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Path to directory where simulation output and metadata are saved (overrides config if given)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom name for run subfolder (default: run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating and saving visualization plots",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively during execution",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Allow CLI override of output directory
    output_dir = args.output_dir or config.get("output", {}).get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Run simulation
    results = run_simulation(config, verbose=True)

    # 3. Save simulation outputs (.npz) and metadata (.json)
    run_dir = save_simulation_output(
        output_dir=output_dir,
        config=config,
        results=results,
        run_name=args.run_name,
    )

    # 4. Generate and save diagnostic plots
    save_plots = config.get("output", {}).get("save_plots", True) and not args.no_plots
    if save_plots or args.show_plots:
        print("Generating diagnostic plots...")
        plot_simulation_results(
            results=results,
            config=config,
            output_dir=run_dir if save_plots else None,
            show=args.show_plots,
        )

    print("\n" + "=" * 60)
    print("Simulation Run Finished Successfully!")
    print(f"Output files and metadata saved in: {os.path.abspath(run_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

