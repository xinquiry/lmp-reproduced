#!/usr/bin/env python3
"""
Plot temperature-dependent cohesive energy from CSV.
Run on Mac after transferring CSV from cluster.

Usage:
    python scripts/reproduce/plot_temperature.py output/cohesive_energy_vs_temperature.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> dict:
    """Load CSV and organize by material."""
    data = {}
    
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mat = row["Material"]
            temp = int(row["Temperature_K"])
            energy = float(row["CohesiveEnergy_eV"])
            
            if mat not in data:
                data[mat] = {"temps": [], "energies": []}
            
            data[mat]["temps"].append(temp)
            data[mat]["energies"].append(energy)
    
    # Sort by temperature
    for mat in data:
        idx = np.argsort(data[mat]["temps"])
        data[mat]["temps"] = [data[mat]["temps"][i] for i in idx]
        data[mat]["energies"] = [data[mat]["energies"][i] for i in idx]
    
    return data


def plot_cohesive_energy(data: dict, output_path: Path):
    """Create publication-quality plot."""
    
    # Style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors and markers
    styles = {
        "Al": {"color": "#1f77b4", "marker": "o", "label": "Al"},
        "Ti": {"color": "#ff7f0e", "marker": "s", "label": "Ti"},
        "SiC": {"color": "#2ca02c", "marker": "^", "label": "SiC"},
        "TiB2": {"color": "#d62728", "marker": "D", "label": "TiB₂"},
        "Ti-TiB2": {"color": "#9467bd", "marker": "v", "label": "Ti-TiB₂ Interface"},
    }
    
    for mat, values in data.items():
        style = styles.get(mat, {"color": "gray", "marker": "x", "label": mat})
        ax.plot(
            values["temps"], values["energies"],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            linewidth=2,
            markersize=8,
        )
    
    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel("Cohesive Energy (eV/atom)", fontsize=12)
    ax.set_title("Temperature-Dependent Cohesive Energy", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    
    # Also save PDF for report
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved PDF: {pdf_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot cohesive energy vs temperature")
    parser.add_argument("csv_file", type=Path, help="CSV file from postprocess_temperature.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output image path (default: same as CSV with .png)")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.csv_file.with_suffix(".png")
    
    print(f"Loading: {args.csv_file}")
    data = load_csv(args.csv_file)
    
    print(f"\nMaterials found: {list(data.keys())}")
    for mat, values in data.items():
        print(f"  {mat}: {len(values['temps'])} data points")
    
    plot_cohesive_energy(data, args.output)


if __name__ == "__main__":
    main()
