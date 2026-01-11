#!/usr/bin/env python3
"""
Post-process temperature-dependent cohesive energy results.
Run on cluster after simulations complete.

Output: CSV file with all results (easy to transfer)
"""

import csv
from pathlib import Path
import sys

# Add src to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lmp_reproduced import LAMMPSPostProcessor


def extract_results(output_dir: Path) -> list[dict]:
    """Extract cohesive energy from all completed simulations."""
    processor = LAMMPSPostProcessor()
    results = []
    
    materials = ["Al", "SiC", "Ti", "TiB2", "Ti-TiB2"]
    temperatures = [100, 300, 500, 700, 900]
    
    for mat in materials:
        for temp in temperatures:
            result_file = output_dir / mat / f"T{temp}K" / "final.result"
            
            if not result_file.exists():
                print(f"  MISSING: {mat}/T{temp}K")
                continue
            
            try:
                result = processor.calculate_cohesive_energy(result_file)
                
                # Calculate average across atom types
                avg_energy = sum(result.energies.values()) / len(result.energies)
                total_atoms = sum(result.n_atoms.values())
                
                results.append({
                    "material": mat,
                    "temperature_K": temp,
                    "cohesive_energy_eV": avg_energy,
                    "n_atoms": total_atoms,
                    "energies_by_type": result.energies,
                })
                
                print(f"  OK: {mat}/T{temp}K -> {avg_energy:.4f} eV/atom")
                
            except Exception as e:
                print(f"  ERROR: {mat}/T{temp}K - {e}")
    
    return results


def save_csv(results: list[dict], output_path: Path):
    """Save results to CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Material", "Temperature_K", "CohesiveEnergy_eV", "N_atoms"])
        
        for r in results:
            writer.writerow([
                r["material"],
                r["temperature_K"],
                f"{r['cohesive_energy_eV']:.6f}",
                r["n_atoms"],
            ])
    
    print(f"\nSaved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Post-process temperature simulations")
    parser.add_argument("--input", type=Path, default=None,
                        help="Input directory with simulation results (default: output/temperature from project root)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV file (default: in input directory)")
    
    args = parser.parse_args()
    
    # Smart default: find output/temperature relative to script or use current dir
    if args.input is None:
        script_dir = Path(__file__).parent.parent.parent
        default_input = script_dir / "output" / "temperature"
        if default_input.exists():
            args.input = default_input
        else:
            args.input = Path(".")
    
    if args.output is None:
        args.output = args.input / "cohesive_energy_vs_temperature.csv"
    
    print("=" * 60)
    print("Post-Processing Temperature-Dependent Cohesive Energy")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print()
    
    results = extract_results(args.input)
    
    if results:
        save_csv(results, args.output)
        
        # Print summary table
        print("\n" + "=" * 60)
        print("SUMMARY TABLE")
        print("=" * 60)
        print(f"{'Material':<12} {'100K':>10} {'300K':>10} {'500K':>10} {'700K':>10} {'900K':>10}")
        print("-" * 65)
        
        for mat in ["Al", "SiC", "Ti", "TiB2", "Ti-TiB2"]:
            row = f"{mat:<12}"
            for temp in [100, 300, 500, 700, 900]:
                r = next((x for x in results if x["material"] == mat and x["temperature_K"] == temp), None)
                if r:
                    row += f"{r['cohesive_energy_eV']:>10.4f}"
                else:
                    row += f"{'N/A':>10}"
            print(row)
        
        print("\n" + "=" * 60)
        print(f"Done! Transfer {args.output} to your Mac for plotting.")
    else:
        print("No results found! Check if simulations completed.")


if __name__ == "__main__":
    main()
