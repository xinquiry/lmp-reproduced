#!/usr/bin/env python3
"""
Temperature-dependent cohesive and interface binding energy calculations.

Calculates cohesive energy at different temperatures for:
- Pure materials: Al, SiC, Ti, TiB2
- Interfaces: SiC-Al, TiB2-Ti

Reference: 技术报告1-1, 1-2

Usage:
    # Generate input files only (for cluster submission)
    python scripts/reproduce/temperature_cohesive_energy.py --dry-run
    
    # Run simulations locally
    python scripts/reproduce/temperature_cohesive_energy.py
    
    # Specific temperatures:
    python scripts/reproduce/temperature_cohesive_energy.py --temps 300 500 700

    # Specific materials only:
    python scripts/reproduce/temperature_cohesive_energy.py --materials al ti
"""

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MaterialCase:
    """Configuration for a single material."""
    name: str
    element: str
    lattice_a: float
    lattice_c: Optional[float]
    structure_type: str
    system: str
    n_types: int = 1


# Material configurations
MATERIALS = {
    "al": MaterialCase(
        name="Al", element="Al", lattice_a=4.032, lattice_c=None,
        structure_type="FCC", system="AL"
    ),
    "ti": MaterialCase(
        name="Ti", element="Ti", lattice_a=2.92, lattice_c=4.772,
        structure_type="HCP", system="TI"
    ),
    "sic": MaterialCase(
        name="SiC", element="SiC", lattice_a=4.36, lattice_c=None,
        structure_type="ZINCBLENDE", system="SI_C", n_types=2
    ),
    "tib2": MaterialCase(
        name="TiB2", element="TiB2", lattice_a=3.050, lattice_c=3.197,
        structure_type="HEXAGONAL", system="TIB2", n_types=2
    ),
}


def find_lammps() -> Optional[str]:
    """Find LAMMPS executable."""
    for cmd in ["lmp_serial", "lmp", "lmp_mpi"]:
        try:
            subprocess.check_call(["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return cmd
        except subprocess.CalledProcessError:
            continue
    return None


def generate_material_inputs(
    case: MaterialCase,
    temperature: int,
    output_base: Path,
    equil_steps: int = 5000,
    prod_steps: int = 5000,
) -> Path:
    """
    Generate input files for cohesive energy calculation.
    
    Returns:
        Path to the output directory
    """
    from lmp_reproduced import (
        MaterialConfig, StructureType, MaterialSystem,
        create_structure, write_lammps_data,
    )
    from lmp_reproduced.core.input_generator import LAMMPSInputGenerator
    
    # Create output directory
    output_dir = output_base / case.name / f"T{temperature}K"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create material config
    structure_type = getattr(StructureType, case.structure_type)
    system = getattr(MaterialSystem, case.system)
    
    config = MaterialConfig(
        element=case.element,
        lattice_a=case.lattice_a,
        lattice_c=case.lattice_c,
        structure_type=structure_type,
        system=system,
    )
    
    # Generate structure
    structure = create_structure(config, n_cells=4)
    struct_file = output_dir / "data.struct"
    write_lammps_data(structure, struct_file)
    
    # Generate LAMMPS script
    generator = LAMMPSInputGenerator(
        material_system=system,
        structure_file="data.struct",
    )
    script = generator.generate_temperature_relaxation(
        temperature=temperature,
        equil_steps=equil_steps,
        prod_steps=prod_steps,
        result_file="final.result",
    )
    
    input_file = output_dir / "in.lammps"
    input_file.write_text(script)
    
    # Create run script for cluster
    run_script = output_dir / "run.sh"
    run_script.write_text(f"""#!/bin/bash
#SBATCH --job-name={case.name}_T{temperature}K
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=01:00:00

cd {output_dir}
mpirun -np 4 lmp_mpi -in in.lammps
""")
    
    return output_dir


def run_lammps_simulation(output_dir: Path, lammps_cmd: str) -> bool:
    """Run LAMMPS simulation in the given directory."""
    try:
        log_file = output_dir / "log.lammps"
        with open(log_file, "w") as log:
            subprocess.check_call(
                [lammps_cmd, "-in", "in.lammps"],
                cwd=output_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ERROR running LAMMPS: {e}")
        return False


def postprocess_cohesive(output_dir: Path) -> Optional[dict]:
    """Post-process cohesive energy from simulation output."""
    from lmp_reproduced import LAMMPSPostProcessor
    
    processor = LAMMPSPostProcessor()
    try:
        result = processor.calculate_cohesive_energy(output_dir / "final.result")
        return {
            "energies": result.energies,
            "n_atoms": result.n_atoms,
        }
    except Exception as e:
        print(f"    ERROR in post-processing: {e}")
        return None


def generate_interface_inputs(
    name: str,
    bottom_case: MaterialCase,
    top_case: MaterialCase,
    temperature: int,
    output_base: Path,
    equil_steps: int = 5000,
    prod_steps: int = 5000,
) -> Optional[Path]:
    """Generate input files for interface binding energy calculation."""
    from lmp_reproduced import (
        MaterialConfig, StructureType, MaterialSystem,
    )
    from lmp_reproduced.core.structures import create_interface, write_lammps_data
    from lmp_reproduced.core.input_generator import LAMMPSInputGenerator
    
    output_dir = output_base / name / f"T{temperature}K"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configs
    bottom_struct = getattr(StructureType, bottom_case.structure_type)
    top_struct = getattr(StructureType, top_case.structure_type)
    
    bottom_config = MaterialConfig(
        element=bottom_case.element,
        lattice_a=bottom_case.lattice_a,
        lattice_c=bottom_case.lattice_c,
        structure_type=bottom_struct,
        system=getattr(MaterialSystem, bottom_case.system),
    )
    
    top_config = MaterialConfig(
        element=top_case.element,
        lattice_a=top_case.lattice_a,
        lattice_c=top_case.lattice_c,
        structure_type=top_struct,
        system=getattr(MaterialSystem, top_case.system),
    )
    
    # Determine material system for potential
    if bottom_case.element == "Ti" and top_case.element == "TiB2":
        system = MaterialSystem.TI_B
    elif bottom_case.element == "SiC" and top_case.element == "Al":
        system = MaterialSystem.C_SI_AL
    else:
        print(f"    Unknown interface system: {bottom_case.element}-{top_case.element}")
        return None
    
    # Create interface structure
    try:
        structure = create_interface(bottom_config, top_config, n_layers_bottom=4, n_layers_top=4)
        struct_file = output_dir / "interface.data"
        write_lammps_data(structure, struct_file)
    except Exception as e:
        print(f"    ERROR creating interface: {e}")
        return None
    
    # Generate script
    generator = LAMMPSInputGenerator(
        material_system=system,
        structure_file="interface.data",
    )
    script = generator.generate_temperature_relaxation(
        temperature=temperature,
        equil_steps=equil_steps,
        prod_steps=prod_steps,
        result_file="final.result",
    )
    
    input_file = output_dir / "in.lammps"
    input_file.write_text(script)
    
    # Create run script for cluster
    run_script = output_dir / "run.sh"
    run_script.write_text(f"""#!/bin/bash
#SBATCH --job-name={name}_T{temperature}K
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=02:00:00

cd {output_dir}
mpirun -np 4 lmp_mpi -in in.lammps
""")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Temperature-dependent cohesive energy calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--temps", nargs="+", type=int,
        default=[100, 300, 500, 700, 900],
        help="Temperatures to calculate (K)",
    )
    parser.add_argument(
        "--materials", nargs="+",
        choices=list(MATERIALS.keys()),
        default=list(MATERIALS.keys()),
        help="Materials to calculate",
    )
    parser.add_argument(
        "--interfaces", nargs="+",
        choices=["ti_tib2", "sic_al"],
        default=["ti_tib2"],
        help="Interfaces to calculate",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/temperature"),
        help="Output directory",
    )
    parser.add_argument(
        "--equil-steps", type=int, default=5000,
        help="Equilibration MD steps",
    )
    parser.add_argument(
        "--prod-steps", type=int, default=5000,
        help="Production MD steps",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate input files only, do not run LAMMPS",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Temperature-Dependent Cohesive Energy Calculations")
    print("=" * 70)
    print(f"Mode: {'DRY-RUN (generate only)' if args.dry_run else 'FULL (run LAMMPS)'}")
    print(f"Temperatures: {args.temps} K")
    print(f"Materials: {args.materials}")
    print(f"Interfaces: {args.interfaces}")
    print(f"Output: {args.output}")
    print()
    
    # Find LAMMPS if not dry-run
    lammps_cmd = None
    if not args.dry_run:
        lammps_cmd = find_lammps()
        if lammps_cmd is None:
            print("ERROR: LAMMPS executable not found!")
            print("Use --dry-run to generate input files only.")
            return
        print(f"LAMMPS: {lammps_cmd}")
    
    # Results storage
    generated_dirs = []
    material_results = {}
    interface_results = {}
    
    # Generate/run material calculations
    for mat_key in args.materials:
        case = MATERIALS[mat_key]
        print(f"\n[Material: {case.name}]")
        print("-" * 40)
        
        results = []
        for temp in args.temps:
            print(f"  T = {temp}K...", end=" ", flush=True)
            
            # Generate inputs
            output_dir = generate_material_inputs(
                case, temp, args.output,
                equil_steps=args.equil_steps,
                prod_steps=args.prod_steps,
            )
            generated_dirs.append(output_dir)
            
            if args.dry_run:
                print(f"Generated: {output_dir}")
            else:
                # Run simulation
                if run_lammps_simulation(output_dir, lammps_cmd):
                    result = postprocess_cohesive(output_dir)
                    if result:
                        result["temperature"] = temp
                        results.append(result)
                        e_str = ", ".join([f"Type{k}: {v:.4f} eV" for k, v in result["energies"].items()])
                        print(f"Done. {e_str}")
                    else:
                        print("POST-PROCESS FAILED")
                else:
                    print("LAMMPS FAILED")
        
        material_results[case.name] = results
    
    # Generate/run interface calculations
    for iface_key in args.interfaces:
        if iface_key == "ti_tib2":
            name = "Ti-TiB2"
            bottom = MATERIALS["ti"]
            top = MATERIALS["tib2"]
        elif iface_key == "sic_al":
            name = "SiC-Al"
            bottom = MATERIALS["sic"]
            top = MATERIALS["al"]
        else:
            continue
        
        print(f"\n[Interface: {name}]")
        print("-" * 40)
        
        results = []
        for temp in args.temps:
            print(f"  T = {temp}K...", end=" ", flush=True)
            
            output_dir = generate_interface_inputs(
                name, bottom, top, temp, args.output,
                equil_steps=args.equil_steps,
                prod_steps=args.prod_steps,
            )
            
            if output_dir is None:
                print("FAILED to generate")
                continue
            
            generated_dirs.append(output_dir)
            
            if args.dry_run:
                print(f"Generated: {output_dir}")
            else:
                if run_lammps_simulation(output_dir, lammps_cmd):
                    # Post-process interface
                    from lmp_reproduced import LAMMPSPostProcessor
                    processor = LAMMPSPostProcessor()
                    try:
                        data = processor.read_data(str(output_dir / "final.result"))
                        total_pe = float(np.sum(data.potentials))
                        n_atoms = data.n_atoms
                        results.append({
                            "temperature": temp,
                            "pe_per_atom": total_pe / n_atoms,
                            "n_atoms": n_atoms,
                        })
                        print(f"Done. PE/atom: {total_pe/n_atoms:.4f} eV")
                    except Exception as e:
                        print(f"POST-PROCESS FAILED: {e}")
                else:
                    print("LAMMPS FAILED")
        
        interface_results[name] = results
    
    # Summary
    print("\n" + "=" * 70)
    
    if args.dry_run:
        print("DRY-RUN COMPLETE - Input files generated")
        print("=" * 70)
        print(f"\nGenerated {len(generated_dirs)} simulation directories:")
        for d in generated_dirs[:5]:
            print(f"  - {d}")
        if len(generated_dirs) > 5:
            print(f"  ... and {len(generated_dirs) - 5} more")
        
        print("\nTo run on a cluster, submit the run.sh scripts:")
        print(f"  cd {args.output}")
        print("  find . -name 'run.sh' -exec sbatch {} \\;")
        
        print("\nTo post-process results after running:")
        print("  python scripts/reproduce/temperature_cohesive_energy.py --postprocess-only")
    else:
        print("SUMMARY: Temperature-Dependent Cohesive Energy (eV/atom)")
        print("=" * 70)
        
        temp_header = "".join([f"{t:>8}K" for t in args.temps])
        print(f"{'Material':<12} {temp_header}")
        print("-" * 70)
        
        for mat_name, results in material_results.items():
            if not results:
                print(f"{mat_name:<12} (no data)")
                continue
            
            row = f"{mat_name:<12}"
            for temp in args.temps:
                temp_result = next((r for r in results if r["temperature"] == temp), None)
                if temp_result:
                    avg_e = sum(temp_result["energies"].values()) / len(temp_result["energies"])
                    row += f"{avg_e:>9.4f}"
                else:
                    row += f"{'N/A':>9}"
            print(row)
        
        if interface_results:
            print("\n" + "-" * 70)
            print("Interface Binding Energy (eV/atom)")
            print("-" * 70)
            for iface_name, results in interface_results.items():
                row = f"{iface_name:<12}"
                for temp in args.temps:
                    temp_result = next((r for r in results if r["temperature"] == temp), None)
                    if temp_result:
                        row += f"{temp_result['pe_per_atom']:>9.4f}"
                    else:
                        row += f"{'N/A':>9}"
                print(row)
    
    print("\n" + "=" * 70)
    print(f"Output directory: {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
