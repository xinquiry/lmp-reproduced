import sys
from pathlib import Path
from lmp_reproduced.core.post_processor import LAMMPSPostProcessor
from lmp_reproduced.models import CohesiveEnergyResult

def show_results():
    base_dir = Path("simulations")
    
    # Al/Mg
    print("\n--- Al/Mg Interface ---")
    al_mg_res = base_dir / "Al_Mg/final.result"
    if al_mg_res.exists():
        post = LAMMPSPostProcessor()
        # Reference energies used in calculation
        ref_obj = CohesiveEnergyResult(
            energies={1: -3.36, 2: -1.51},
            n_atoms={1:1, 2:1},
            source_file=Path("manual")
        )
        try:
            res = post.calculate_interface_energy(al_mg_res, reference=ref_obj)
            print(f"Status: Complete")
            print(f"Interface Position Y: {res.position_y:.2f} Angstrom")
            print(f"Interface Energy:     {res.interface_energy:.2f} mJ/m^2")
        except Exception as e:
            print(f"Error parsing: {e}")
    else:
        print("Status: Not found or Incomplete")

    # Ti/TiB2
    print("\n--- Ti/TiB2 Interface ---")
    ti_tib2_dir = base_dir / "Ti_TiB2"
    if (ti_tib2_dir / "final.result").exists():
         print("Status: Complete (Run script to calculate spatial energy)")
    elif (ti_tib2_dir / "log.lammps").exists():
        print("Status: Running (Check log.lammps)")
        # Show last line of log
        with open(ti_tib2_dir / "log.lammps") as f:
            lines = f.readlines()
            if lines:
                print(f"Last Log Entry: {lines[-1].strip()}")
    else:
        print("Status: Not Started")

if __name__ == "__main__":
    show_results()
