"""
Module for running LAMMPS simulations.

This module provides functions to execute LAMMPS simulations and
manage the simulation workflow.
"""

import subprocess
from pathlib import Path
from typing import Optional

from lmp_reproduced.models import (
    MaterialConfig,
    MinimizationConfig,
    RelaxationConfig,
    SimulationResult,
    StructureType,
)
from lmp_reproduced.core.structures import create_structure, write_lammps_data
from lmp_reproduced.core.input_generator import (
    LAMMPSInputGenerator,
    generate_minimization_script,
    generate_relaxation_script,
)


def find_lammps() -> Optional[str]:
    """
    Find available LAMMPS command.

    Searches common installation paths and PATH for LAMMPS executables.

    Returns:
        Path to LAMMPS executable, or None if not found
    """
    # Common names and paths
    commands = ["lmp_serial", "lmp_mpi", "lmp", "lammps"]
    paths = [Path("/opt/homebrew/bin"), Path("/usr/local/bin"), Path("/usr/bin")]

    # Check absolute paths first (more reliable)
    for path in paths:
        for cmd in commands:
            full_path = path / cmd
            if full_path.exists() and full_path.is_file():
                try:
                    subprocess.run(
                        [str(full_path), "-h"],
                        capture_output=True,
                        timeout=5
                    )
                    return str(full_path)
                except (subprocess.TimeoutExpired, OSError):
                    continue

    # Fallback to PATH
    for cmd in commands:
        try:
            subprocess.run([cmd, "-h"], capture_output=True, timeout=5)
            return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return None


def run_lammps(
    input_script: str,
    work_dir: Path,
    lammps_cmd: Optional[str] = None,
    timeout: int = 300,
) -> SimulationResult:
    """
    Execute a LAMMPS simulation.

    Args:
        input_script: LAMMPS input script content
        work_dir: Working directory for the simulation
        lammps_cmd: LAMMPS executable (auto-detected if None)
        timeout: Maximum runtime in seconds

    Returns:
        SimulationResult with success status and output paths
    """
    # Find LAMMPS if not provided
    if lammps_cmd is None:
        lammps_cmd = find_lammps()
        if lammps_cmd is None:
            return SimulationResult(
                success=False,
                output_dir=work_dir,
                error_message="LAMMPS executable not found"
            )

    # Ensure working directory exists
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write input script
    input_file = work_dir / "in.lammps"
    input_file.write_text(input_script)

    # Ensure results directory exists
    results_dir = work_dir / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run LAMMPS
    try:
        result = subprocess.run(
            [lammps_cmd, "-in", "in.lammps"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        log_file = work_dir / "log.lammps"
        result_file = work_dir / "final.result"

        if result.returncode != 0:
            return SimulationResult(
                success=False,
                output_dir=work_dir,
                log_file=log_file if log_file.exists() else None,
                error_message=result.stderr or "LAMMPS returned non-zero exit code",
                stdout=result.stdout,
                stderr=result.stderr
            )

        return SimulationResult(
            success=True,
            output_dir=work_dir,
            log_file=log_file if log_file.exists() else None,
            result_file=result_file if result_file.exists() else None,
            stdout=result.stdout,
            stderr=result.stderr
        )

    except subprocess.TimeoutExpired:
        return SimulationResult(
            success=False,
            output_dir=work_dir,
            error_message=f"LAMMPS timed out after {timeout} seconds"
        )
    except OSError as e:
        return SimulationResult(
            success=False,
            output_dir=work_dir,
            error_message=f"Failed to run LAMMPS: {e}"
        )


def run_minimization(
    config: MinimizationConfig,
    lammps_cmd: Optional[str] = None,
    n_cells: int = 4,
) -> SimulationResult:
    """
    Run energy minimization simulation.

    This is a high-level function that:
    1. Creates the crystal structure
    2. Generates the LAMMPS input script
    3. Runs the simulation

    Args:
        config: Minimization configuration
        lammps_cmd: LAMMPS executable (auto-detected if None)
        n_cells: Number of unit cells in each direction

    Returns:
        SimulationResult with simulation outcome
    """
    work_dir = Path(config.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create structure
    structure = create_structure(config.material, n_cells=n_cells)
    struct_file = work_dir / "data.struct"
    write_lammps_data(structure, struct_file)

    # Generate input script
    input_script = generate_minimization_script(config)

    # Run simulation
    return run_lammps(input_script, work_dir, lammps_cmd)


def run_relaxation(
    config: RelaxationConfig,
    lammps_cmd: Optional[str] = None,
    n_cells: int = 4,
) -> SimulationResult:
    """
    Run dynamic relaxation simulation.

    This is a high-level function that:
    1. Creates the crystal structure
    2. Generates the LAMMPS input script
    3. Runs the simulation

    Args:
        config: Relaxation configuration
        lammps_cmd: LAMMPS executable (auto-detected if None)
        n_cells: Number of unit cells in each direction

    Returns:
        SimulationResult with simulation outcome
    """
    work_dir = Path(config.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create structure
    structure = create_structure(config.material, n_cells=n_cells)
    struct_file = work_dir / "data.struct"
    write_lammps_data(structure, struct_file)

    # Generate input script
    input_script = generate_relaxation_script(config)

    # Run simulation
    return run_lammps(input_script, work_dir, lammps_cmd)


def run_single_material_simulation(
    material: MaterialConfig,
    output_dir: Path,
    lammps_cmd: Optional[str] = None,
    n_cells: int = 4,
    etol: float = 0.0,
    ftol: float = 3.12e-3,
) -> SimulationResult:
    """
    Run minimization simulation for a single material.

    Convenience function that creates a MinimizationConfig and runs the simulation.

    Args:
        material: Material configuration
        output_dir: Output directory for simulation files
        lammps_cmd: LAMMPS executable (auto-detected if None)
        n_cells: Number of unit cells in each direction
        etol: Energy tolerance for minimization
        ftol: Force tolerance for minimization

    Returns:
        SimulationResult with simulation outcome
    """
    config = MinimizationConfig(
        material=material,
        output_dir=output_dir,
        etol=etol,
        ftol=ftol,
    )

    return run_minimization(config, lammps_cmd, n_cells)
