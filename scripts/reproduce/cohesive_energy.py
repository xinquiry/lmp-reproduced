#!/usr/bin/env python3
"""
Reproduce cohesive energy calculations from technical report.

Reference: 金属基复合材料温度相关物性计算模块技术报告 (Report 1-2)

This script validates the lmp_reproduced package by comparing calculated
cohesive energies against published values from the technical report.

Expected Results:
- Al (FCC):  Ecoh = -3.36 eV/atom
- Mg (HCP):  Ecoh = -1.51 eV/atom
- Ti (HCP):  Ecoh = -4.873 eV/atom
- TiB2 (Hex): B = -7.580 eV, Ti = -4.497 eV

Usage:
    python scripts/reproduce/cohesive_energy.py

    # Run specific materials only:
    python scripts/reproduce/cohesive_energy.py --materials al mg

    # Custom output directory:
    python scripts/reproduce/cohesive_energy.py --output ./my_output
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ValidationCase:
    """A single validation case for cohesive energy."""
    name: str
    element: str
    lattice_a: float
    lattice_c: Optional[float]
    structure_type: str
    system: str
    expected: dict[int, float]  # atom_type -> expected energy


# Validation cases from Report 1-2
VALIDATION_CASES = {
    "al": ValidationCase(
        name="Al (FCC)",
        element="Al",
        lattice_a=4.032,
        lattice_c=None,
        structure_type="FCC",
        system="AL",
        expected={1: -3.36},
    ),
    "mg": ValidationCase(
        name="Mg (HCP)",
        element="Mg",
        lattice_a=3.196,
        lattice_c=5.197,
        structure_type="HCP",
        system="MG",
        expected={1: -1.51},
    ),
    "ti": ValidationCase(
        name="Ti (HCP)",
        element="Ti",
        lattice_a=2.92,  # From MEAM library.meam (alat parameter)
        lattice_c=4.772,  # c = 1.633 * a (ideal HCP c/a ratio)
        structure_type="HCP",
        system="TI",
        expected={2: -4.87},  # Ti is type 2 in MEAM potential, esub from library.meam
    ),
    "tib2": ValidationCase(
        name="TiB2 (Hexagonal)",
        element="TiB2",
        lattice_a=3.050,
        lattice_c=3.197,
        structure_type="HEXAGONAL",
        system="TIB2",
        expected={1: -7.580, 2: -4.497},  # B, Ti
    ),
}


def run_validation(
    case: ValidationCase,
    output_base: Path,
    verbose: bool = True
) -> dict:
    """
    Run a single validation case.

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
        cohesive_energy_workflow,
    )

    # Create material config
    structure_type = getattr(StructureType, case.structure_type)
    system = getattr(MaterialSystem, case.system)

    material = MaterialConfig(
        element=case.element,
        lattice_a=case.lattice_a,
        lattice_c=case.lattice_c,
        structure_type=structure_type,
        system=system,
    )

    # Set up output directory
    output_dir = output_base / case.name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {case.name}")
        print(f"  Lattice: a={case.lattice_a} Å", end="")
        if case.lattice_c:
            print(f", c={case.lattice_c} Å")
        else:
            print()
        print(f"  Output: {output_dir}")

    try:
        result = cohesive_energy_workflow(
            material=material,
            output_dir=output_dir,
            n_cells=4,
        )

        # Validate results
        all_passed = True
        validation_results = []

        for atom_type, expected in case.expected.items():
            actual = result.energies.get(atom_type, 0)
            error_pct = abs(actual - expected) / abs(expected) * 100
            passed = error_pct < 10.0  # 10% tolerance (different potentials may vary)

            if not passed:
                all_passed = False

            validation_results.append({
                "type": atom_type,
                "expected": expected,
                "actual": actual,
                "error_pct": error_pct,
                "passed": passed,
            })

            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"  Type {atom_type}: {actual:.4f} eV (expected {expected:.4f}, error {error_pct:.2f}%) [{status}]")

        return {
            "case": case.name,
            "success": True,
            "passed": all_passed,
            "result": result,
            "validation": validation_results,
        }

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")

        return {
            "case": case.name,
            "success": False,
            "passed": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Validate cohesive energy calculations against Report 1-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[0],
    )
    parser.add_argument(
        "--materials",
        nargs="+",
        choices=list(VALIDATION_CASES.keys()),
        default=list(VALIDATION_CASES.keys()),
        help="Materials to validate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output/validation"),
        help="Base output directory (default: ./output/validation)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("Cohesive Energy Validation")
    print("=" * 60)
    print(f"Reference: Report 1-2 (金属基复合材料温度相关物性计算模块技术报告)")
    print(f"Materials: {', '.join(args.materials)}")
    print(f"Output: {args.output}")

    # Run validations
    results = []
    for material_key in args.materials:
        case = VALIDATION_CASES[material_key]
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
        else:
            status = "ERROR"
        print(f"  {r['case']}: {status}")

    print(f"\nResults: {passed_count}/{total_count} passed")

    # Exit with error code if any failed
    if passed_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
