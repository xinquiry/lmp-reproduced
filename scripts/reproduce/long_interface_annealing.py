#!/usr/bin/env python3
"""
Long-time interface annealing simulations for Al-Mg and Ti-TiB2.

Runs extended MD annealing (100,000+ steps) to properly relax interface structures
and get accurate interface energies.

Usage:
    # Generate input files (on Mac)
    python scripts/reproduce/long_interface_annealing.py --dry-run

    # Run on cluster
    cd output/interface_annealing
    mpirun -np 20 lmp -in Al-Mg/in.lammps
"""

import argparse
from pathlib import Path

def generate_al_mg_annealing(output_dir: Path, heat_steps: int, hold_steps: int, cool_steps: int):
    """Generate Al-Mg interface annealing simulation."""
    from lmp_reproduced import (
        MaterialConfig, StructureType, MaterialSystem,
        ALUMINUM, MAGNESIUM,
    )
    from lmp_reproduced.core.structures import create_interface, write_lammps_data
    from lmp_reproduced.core.input_generator import LAMMPSInputGenerator
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create interface
    al_config = MaterialConfig(
        element="Al", lattice_a=4.032, structure_type=StructureType.FCC,
        system=MaterialSystem.AL_MG,
    )
    mg_config = MaterialConfig(
        element="Mg", lattice_a=3.196, lattice_c=5.197, structure_type=StructureType.HCP,
        system=MaterialSystem.AL_MG,
    )
    
    structure = create_interface(al_config, mg_config, n_layers_bottom=8, n_layers_top=8)
    write_lammps_data(structure, output_dir / "interface.data")
    
    # Generate annealing script
    generator = LAMMPSInputGenerator(
        material_system=MaterialSystem.AL_MG,
        structure_file="interface.data",
    )
    
    script = generator.generate_annealing(
        max_temp=600,
        heat_steps=heat_steps,
        hold_steps=hold_steps,
        cool_steps=cool_steps,
        result_file="final.result",
    )
    (output_dir / "in.lammps").write_text(script)
    
    # SLURM script
    (output_dir / "run.sh").write_text(f"""#!/bin/bash
#SBATCH --job-name=AlMg_anneal
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --time=12:00:00

cd {output_dir.absolute()}
mpirun -np 20 lmp -in in.lammps > log.lammps 2>&1
""")
    
    print(f"Generated Al-Mg annealing: {output_dir}")
    return output_dir


def generate_ti_tib2_annealing(output_dir: Path, heat_steps: int, hold_steps: int, cool_steps: int):
    """Generate Ti-TiB2 interface annealing simulation."""
    from lmp_reproduced import (
        MaterialConfig, StructureType, MaterialSystem,
    )
    from lmp_reproduced.core.structures import create_ti_tib2_interface, write_lammps_data
    from lmp_reproduced.core.input_generator import LAMMPSInputGenerator
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create interface with default 30x30 Ti / 29x29 TiB2
    ti_config = MaterialConfig(
        element="Ti", lattice_a=2.92, lattice_c=4.772,
        structure_type=StructureType.HCP, system=MaterialSystem.TI_B,
    )
    tib2_config = MaterialConfig(
        element="TiB2", lattice_a=3.050, lattice_c=3.197,
        structure_type=StructureType.HEXAGONAL, system=MaterialSystem.TIB2,
    )
    
    # Use default parameters (n_layers adjustable)
    structure = create_ti_tib2_interface(
        ti_config, tib2_config,
        n_layers_ti=6, n_layers_tib2=6,
    )
    write_lammps_data(structure, output_dir / "interface.data")
    
    # Generate annealing script
    generator = LAMMPSInputGenerator(
        material_system=MaterialSystem.TI_B,
        structure_file="interface.data",
    )
    
    script = generator.generate_annealing(
        max_temp=1200,  # Higher for Ti-TiB2
        heat_steps=heat_steps,
        hold_steps=hold_steps,
        cool_steps=cool_steps,
        result_file="final.result",
    )
    (output_dir / "in.lammps").write_text(script)
    
    # SLURM script
    (output_dir / "run.sh").write_text(f"""#!/bin/bash
#SBATCH --job-name=TiTiB2_anneal
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --time=24:00:00

cd {output_dir.absolute()}
mpirun -np 20 lmp -in in.lammps > log.lammps 2>&1
""")
    
    print(f"Generated Ti-TiB2 annealing: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Long-time interface annealing")
    parser.add_argument("--output", type=Path, default=Path("output/interface_annealing"),
                        help="Output directory")
    parser.add_argument("--heat-steps", type=int, default=20000,
                        help="Heating steps")
    parser.add_argument("--hold-steps", type=int, default=50000,
                        help="Hold at max temperature steps")
    parser.add_argument("--cool-steps", type=int, default=30000,
                        help="Cooling steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate files only")
    parser.add_argument("--systems", nargs="+", choices=["al-mg", "ti-tib2"],
                        default=["al-mg", "ti-tib2"],
                        help="Systems to generate")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Long-Time Interface Annealing")
    print("=" * 60)
    print(f"Heat: {args.heat_steps} steps")
    print(f"Hold: {args.hold_steps} steps")  
    print(f"Cool: {args.cool_steps} steps")
    print(f"Total: {args.heat_steps + args.hold_steps + args.cool_steps} steps")
    print(f"Output: {args.output}")
    print()
    
    if "al-mg" in args.systems:
        generate_al_mg_annealing(
            args.output / "Al-Mg",
            args.heat_steps, args.hold_steps, args.cool_steps
        )
    
    if "ti-tib2" in args.systems:
        generate_ti_tib2_annealing(
            args.output / "Ti-TiB2", 
            args.heat_steps, args.hold_steps, args.cool_steps
        )
    
    print("\n" + "=" * 60)
    print("Done! Transfer to cluster and run:")
    print(f"  cd {args.output}")
    print("  sbatch Al-Mg/run.sh")
    print("  sbatch Ti-TiB2/run.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
