#!/usr/bin/env python3
"""
Reproduce defect formation energy calculations from technical report.

Reference: 金属基复合材料温度相关物性计算模块技术报告 (Report 1-2)

This script validates the lmp_reproduced package by comparing calculated
defect formation energies against published values from the technical report.

Expected Results:
- Mg solute in Al (FCC): E_f = 0.55 eV
- Ti vacancy (HCP): E_f = 1.78 eV

Usage:
    python scripts/reproduce/defect_energy.py

    # Run specific defects only:
    python scripts/reproduce/defect_energy.py --defects mg_in_al

    # Custom output directory:
    python scripts/reproduce/defect_energy.py --output ./my_output
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DefectValidationCase:
    """A single validation case for defect formation energy."""
    name: str
    defect_type: str  # "substitutional" or "vacancy"
    host_element: str
    host_lattice_a: float
    host_lattice_c: Optional[float]
    host_structure: str
    solute_element: Optional[str]  # For substitutional
    system: str  # MaterialSystem name
    expected_energy: float  # Expected formation energy in eV


# Validation cases from Report 1-2
VALIDATION_CASES = {
    "mg_in_al": DefectValidationCase(
        name="Mg in Al (FCC)",
        defect_type="substitutional",
        host_element="Al",
        host_lattice_a=4.032,
        host_lattice_c=None,
        host_structure="FCC",
        solute_element="Mg",
        system="AL_MG",
        expected_energy=0.55,
    ),
    "ti_vacancy": DefectValidationCase(
        name="Ti vacancy (HCP)",
        defect_type="vacancy",
        host_element="Ti",
        host_lattice_a=2.92,  # From MEAM library.meam
        host_lattice_c=4.772,  # c = 1.633 * a
        host_structure="HCP",
        solute_element=None,
        system="TI",
        expected_energy=1.78,
    ),
}


def calculate_cohesive_with_potential(
    element: str,
    lattice_a: float,
    lattice_c: Optional[float],
    structure_type: str,
    system: str,
    output_dir: Path,
    n_types: int = 1,
    atom_type: int = 1,
) -> float:
    """
    Calculate cohesive energy for element using the specified potential.

    This ensures reference energies come from the same potential as defect calculations.

    Args:
        element: Element symbol
        lattice_a, lattice_c: Lattice parameters
        structure_type: "FCC" or "HCP"
        system: MaterialSystem name
        output_dir: Output directory
        n_types: Number of atom types to declare
        atom_type: Atom type ID to use for atoms
    """
    from lmp_reproduced import (
        MaterialConfig,
        StructureType,
        MaterialSystem,
        MinimizationConfig,
        create_fcc_structure,
        create_hcp_structure,
        write_lammps_data,
        run_lammps,
        LAMMPSPostProcessor,
    )
    from lmp_reproduced.core.input_generator import generate_minimization_script
    from lmp_reproduced.core.structures import ATOMIC_MASSES

    structure_type_enum = getattr(StructureType, structure_type)
    system_enum = getattr(MaterialSystem, system)

    config = MaterialConfig(
        element=element,
        lattice_a=lattice_a,
        lattice_c=lattice_c,
        structure_type=structure_type_enum,
        system=system_enum,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create structure with explicit type handling for multi-element potentials
    if structure_type == "FCC":
        structure = create_fcc_structure(config, n_cells=4)
        # Override n_types and masses if needed for multi-element potential
        if n_types > 1:
            masses = {}
            if system == "AL_MG":
                masses = {1: ATOMIC_MASSES["Al"], 2: ATOMIC_MASSES["Mg"]}
            structure = structure._replace(n_types=n_types, masses=masses) if hasattr(structure, '_replace') else structure
            # StructureData is a dataclass, need to create new instance
            from lmp_reproduced.models import StructureData
            structure = StructureData(
                title=structure.title,
                atoms=structure.atoms,
                n_types=n_types,
                box_bounds=structure.box_bounds,
                masses=masses,
            )
    else:  # HCP
        structure = create_hcp_structure(config, n_cells=4, n_types=n_types, atom_type=atom_type)

    struct_file = output_dir / "data.struct"
    write_lammps_data(structure, struct_file)

    # Run minimization
    min_config = MinimizationConfig(material=config, output_dir=output_dir)
    input_script = generate_minimization_script(min_config)
    sim_result = run_lammps(input_script, output_dir)

    if not sim_result.success:
        raise RuntimeError(f"Reference calculation failed: {sim_result.error_message}")

    # Read results
    processor = LAMMPSPostProcessor()
    result_file = sim_result.result_file or (output_dir / "final.result")
    data = processor.read_data(str(result_file))

    # Calculate cohesive energy (per atom)
    total_pe = sum(data.potentials)
    n_atoms = len(data.potentials)

    return total_pe / n_atoms


def run_validation(
    case: DefectValidationCase,
    output_base: Path,
    verbose: bool = True
) -> dict:
    """
    Run a single defect validation case.

    Args:
        case: The validation case to run
        output_base: Base directory for output files
        verbose: Whether to print progress

    Returns:
        Dictionary with case name, result, and validation status
    """
    # Import here to allow script to show help without package installed
    from lmp_reproduced import (
        MaterialConfig,
        StructureType,
        MaterialSystem,
        CohesiveEnergyResult,
        MinimizationConfig,
        create_fcc_substitutional,
        create_hcp_vacancy,
        write_lammps_data,
        run_lammps,
        LAMMPSPostProcessor,
    )
    from lmp_reproduced.core.input_generator import generate_minimization_script

    # Create host material config
    structure_type = getattr(StructureType, case.host_structure)
    system = getattr(MaterialSystem, case.system)

    host_config = MaterialConfig(
        element=case.host_element,
        lattice_a=case.host_lattice_a,
        lattice_c=case.host_lattice_c,
        structure_type=structure_type,
        system=system,
    )

    # Set up output directory
    output_dir = output_base / case.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {case.name}")
        print(f"  Defect type: {case.defect_type}")
        print(f"  Host: {case.host_element} {case.host_structure}, a={case.host_lattice_a} Å")
        if case.host_lattice_c:
            print(f"        c={case.host_lattice_c} Å")
        if case.solute_element:
            print(f"  Solute: {case.solute_element}")
        print(f"  Output: {output_dir}")

    try:
        # Step 1: Create defect structure
        if case.defect_type == "substitutional":
            structure = create_fcc_substitutional(
                host_config=host_config,
                solute_element=case.solute_element,
                n_cells=4,
            )
        else:  # vacancy
            structure = create_hcp_vacancy(
                config=host_config,
                n_cells=4,
                n_types=2,  # Ti system needs 2 types for MEAM
                atom_type=2,  # Ti is type 2 in MEAM
            )

        # Step 2: Write structure to file
        struct_file = output_dir / "data.struct"
        write_lammps_data(structure, struct_file)

        n_atoms = len(structure.atoms)
        if verbose:
            print(f"  Structure: {n_atoms} atoms")

        # Step 3: Run minimization
        config = MinimizationConfig(
            material=host_config,
            output_dir=output_dir,
        )
        input_script = generate_minimization_script(config)
        sim_result = run_lammps(input_script, output_dir)

        if not sim_result.success:
            raise RuntimeError(f"Simulation failed: {sim_result.error_message}")

        # Step 4: Read results and calculate formation energy
        processor = LAMMPSPostProcessor()
        result_file = sim_result.result_file or (output_dir / "final.result")
        data = processor.read_data(str(result_file))

        # Calculate total potential energy
        total_pe = sum(data.potentials)

        # Reference cohesive energies from our validated calculations
        if case.defect_type == "substitutional":
            # Calculate reference cohesive energies using the SAME potential (almg.liu.eam.alloy)
            # This is critical - must use consistent potential for both defect and reference
            if verbose:
                print(f"  Calculating reference cohesive energies with AL_MG potential...")

            ref_dir_al = output_dir / "ref_al"
            e_coh_host = calculate_cohesive_with_potential(
                element="Al",
                lattice_a=case.host_lattice_a,
                lattice_c=None,
                structure_type="FCC",
                system="AL_MG",
                output_dir=ref_dir_al,
                n_types=2,  # Need 2 types for AL_MG potential
                atom_type=1,  # Al is type 1
            )

            ref_dir_mg = output_dir / "ref_mg"
            e_coh_solute = calculate_cohesive_with_potential(
                element="Mg",
                lattice_a=3.196,  # Mg lattice constant
                lattice_c=5.197,
                structure_type="HCP",
                system="AL_MG",
                output_dir=ref_dir_mg,
                n_types=2,  # Need 2 types for AL_MG potential
                atom_type=2,  # Mg is type 2
            )

            if verbose:
                print(f"  Reference E_coh(Al) = {e_coh_host:.4f} eV/atom")
                print(f"  Reference E_coh(Mg) = {e_coh_solute:.4f} eV/atom")

            n_host = n_atoms - 1
            n_solute = 1

            # Formation energy = E_total - (n_host * E_coh_host + n_solute * E_coh_solute)
            e_ref = n_host * e_coh_host + n_solute * e_coh_solute
            formation_energy = total_pe - e_ref

        else:  # vacancy
            # For vacancy in Ti:
            # E_f = E_total - (N-1) * E_coh_Ti = E_total - (N-1) * E_coh
            e_coh = -4.87  # Ti cohesive energy from our validation
            n_remaining = n_atoms  # Already one atom removed
            e_ref = n_remaining * e_coh
            formation_energy = total_pe - e_ref

        # Validate result
        actual = formation_energy
        expected = case.expected_energy
        error_pct = abs(actual - expected) / abs(expected) * 100
        passed = error_pct < 30.0  # 30% tolerance

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  Total PE: {total_pe:.4f} eV")
            print(f"  Reference: {e_ref:.4f} eV")
            print(f"  Formation energy: {actual:.4f} eV (expected {expected:.4f}, error {error_pct:.2f}%) [{status}]")

        return {
            "case": case.name,
            "success": True,
            "passed": passed,
            "actual": actual,
            "expected": expected,
            "error_pct": error_pct,
        }

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

        return {
            "case": case.name,
            "success": False,
            "passed": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Validate defect formation energy calculations against Report 1-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[0],
    )
    parser.add_argument(
        "--defects",
        nargs="+",
        choices=list(VALIDATION_CASES.keys()),
        default=list(VALIDATION_CASES.keys()),
        help="Defects to validate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output/validation_defects"),
        help="Base output directory (default: ./output/validation_defects)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("Defect Formation Energy Validation")
    print("=" * 60)
    print(f"Reference: Report 1-2 (金属基复合材料温度相关物性计算模块技术报告)")
    print(f"Defects: {', '.join(args.defects)}")
    print(f"Output: {args.output}")

    # Run validations
    results = []
    for defect_key in args.defects:
        case = VALIDATION_CASES[defect_key]
        result = run_validation(case, args.output, verbose=not args.quiet)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for r in results if r.get("passed", False))
    total_count = len(results)

    for r in results:
        if r["success"]:
            status = "PASS" if r["passed"] else "FAIL"
            if "actual" in r:
                print(f"  {r['case']}: {r['actual']:.4f} eV (expected {r['expected']:.4f}, error {r['error_pct']:.1f}%) [{status}]")
            else:
                print(f"  {r['case']}: {status}")
        else:
            print(f"  {r['case']}: ERROR - {r.get('error', 'Unknown error')}")

    print(f"\nResults: {passed_count}/{total_count} passed")

    # Exit with error code if any failed
    if passed_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
