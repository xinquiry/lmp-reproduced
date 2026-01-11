#!/usr/bin/env python3
"""
Script to run interface energy calculations for MMC materials.

This script demonstrates the use of the interface_annealing_workflow
for calculating interface energies of metal matrix composite interfaces.

Usage:
    python scripts/reproduce/run_interface_calcs.py
"""

from pathlib import Path

from lmp_reproduced import (
    ALUMINUM,
    MAGNESIUM,
    TITANIUM,
    TIB2,
    interface_annealing_workflow,
)


def main():
    print("=" * 60)
    print("Interface Energy Calculations")
    print("=" * 60)

    # 1. Al/Mg Interface
    print("\n[1] Al/Mg Interface")
    print("-" * 40)
    
    try:
        result_al_mg = interface_annealing_workflow(
            bottom_config=ALUMINUM,
            top_config=MAGNESIUM,
            output_dir=Path("simulations/Al_Mg"),
            references={1: -3.36, 2: -1.51},
            max_temperature=600.0,
        )
        print(f"  Interface Y position: {result_al_mg.position_y:.2f} Å")
        print(f"  Interface energy: {result_al_mg.interface_energy:.2f} mJ/m²")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 2. Ti/TiB2 Interface (uses spatial reference assignment)
    print("\n[2] Ti/TiB2 Interface")
    print("-" * 40)
    
    try:
        result_ti_tib2 = interface_annealing_workflow(
            bottom_config=TITANIUM,
            top_config=TIB2,
            output_dir=Path("simulations/Ti_TiB2"),
            references={
                "B_tib2": -7.580,
                "Ti_metal": -4.873,
                "Ti_tib2": -4.497,
            },
            max_temperature=1200.0,
            spatial_refs=True,
        )
        print(f"  Interface Y position: {result_ti_tib2.position_y:.2f} Å")
        print(f"  Interface energy: {result_ti_tib2.interface_energy:.2f} mJ/m²")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
