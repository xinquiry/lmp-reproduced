"""
High-level workflow pipelines for LAMMPS simulations.

This module provides composable workflow functions that chain together
structure generation, simulation, and post-processing steps.
"""

from pathlib import Path
from typing import Optional

from lmp_reproduced.models import (
    CohesiveEnergyResult,
    DefectResult,
    InterfaceResult,
    MaterialConfig,
    MinimizationConfig,
    RelaxationConfig,
    SimulationResult,
)
from lmp_reproduced.core.structures import create_structure, write_lammps_data
from lmp_reproduced.core.post_processor import LAMMPSPostProcessor
from lmp_reproduced.simulations.runners import (
    find_lammps,
    run_lammps,
    run_minimization,
)
from lmp_reproduced.core.input_generator import generate_minimization_script


class SimulationError(Exception):
    """Raised when a simulation fails."""
    def __init__(self, message: str, result: Optional[SimulationResult] = None):
        super().__init__(message)
        self.result = result


def cohesive_energy_workflow(
    material: MaterialConfig,
    output_dir: Path | str,
    n_cells: int = 4,
    lammps_cmd: Optional[str] = None,
    raise_on_failure: bool = True,
) -> CohesiveEnergyResult:
    """
    Complete workflow: structure -> minimize -> post-process -> cohesive energy.

    This is the main entry point for calculating cohesive energy of a material.

    Args:
        material: Material configuration
        output_dir: Directory for simulation output
        n_cells: Number of unit cells in each direction
        lammps_cmd: LAMMPS executable (auto-detected if None)
        raise_on_failure: If True, raise SimulationError on failure

    Returns:
        CohesiveEnergyResult with calculated cohesive energy

    Raises:
        SimulationError: If simulation fails and raise_on_failure is True

    Example:
        >>> from lmp_reproduced import MaterialConfig, cohesive_energy_workflow, StructureType
        >>> config = MaterialConfig(element="Al", lattice_a=4.05, structure_type=StructureType.FCC)
        >>> result = cohesive_energy_workflow(config, Path("./output"))
        >>> print(f"Cohesive energy: {result.primary_energy:.4f} eV/atom")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create minimization config
    config = MinimizationConfig(
        material=material,
        output_dir=output_dir,
    )

    # Step 1: Create structure
    structure = create_structure(material, n_cells=n_cells)
    struct_file = output_dir / "data.struct"
    write_lammps_data(structure, struct_file)

    # Step 2: Generate and run simulation
    input_script = generate_minimization_script(config)
    sim_result = run_lammps(input_script, output_dir, lammps_cmd)

    if not sim_result.success:
        if raise_on_failure:
            raise SimulationError(
                f"Simulation failed: {sim_result.error_message}",
                result=sim_result
            )
        # Return empty result on failure
        return CohesiveEnergyResult(
            energies={},
            n_atoms={},
            source_file=output_dir / "final.result"
        )

    # Step 3: Post-process results
    processor = LAMMPSPostProcessor()
    result_file = sim_result.result_file or (output_dir / "final.result")

    return processor.calculate_cohesive_energy(result_file)


def batch_cohesive_energy(
    materials: list[MaterialConfig],
    base_output_dir: Path | str,
    n_cells: int = 4,
    lammps_cmd: Optional[str] = None,
    continue_on_failure: bool = True,
) -> dict[str, CohesiveEnergyResult | SimulationError]:
    """
    Run cohesive energy calculations for multiple materials.

    Args:
        materials: List of material configurations
        base_output_dir: Base directory for outputs (subdirs created per material)
        n_cells: Number of unit cells in each direction
        lammps_cmd: LAMMPS executable (auto-detected if None)
        continue_on_failure: If True, continue with remaining materials on failure

    Returns:
        Dictionary mapping element names to results or errors

    Example:
        >>> from lmp_reproduced import ALUMINUM, COPPER, MAGNESIUM, batch_cohesive_energy
        >>> results = batch_cohesive_energy([ALUMINUM, COPPER, MAGNESIUM], Path("./batch"))
        >>> for elem, result in results.items():
        ...     if isinstance(result, CohesiveEnergyResult):
        ...         print(f"{elem}: {result.primary_energy:.4f} eV/atom")
    """
    base_output_dir = Path(base_output_dir)
    results: dict[str, CohesiveEnergyResult | SimulationError] = {}

    for material in materials:
        output_dir = base_output_dir / material.element
        try:
            result = cohesive_energy_workflow(
                material=material,
                output_dir=output_dir,
                n_cells=n_cells,
                lammps_cmd=lammps_cmd,
                raise_on_failure=True,
            )
            results[material.element] = result
        except SimulationError as e:
            results[material.element] = e
            if not continue_on_failure:
                raise

    return results


def defect_energy_workflow(
    material: MaterialConfig,
    defect_structure_file: Path | str,
    reference_energy: CohesiveEnergyResult,
    output_dir: Path | str,
    lammps_cmd: Optional[str] = None,
    raise_on_failure: bool = True,
) -> DefectResult:
    """
    Calculate defect formation energy.

    Workflow:
    1. Run minimization on structure with defect
    2. Post-process to find defect and calculate formation energy

    Args:
        material: Material configuration (for potential selection)
        defect_structure_file: Path to structure file containing the defect
        reference_energy: Reference cohesive energy from perfect crystal
        output_dir: Directory for simulation output
        lammps_cmd: LAMMPS executable (auto-detected if None)
        raise_on_failure: If True, raise SimulationError on failure

    Returns:
        DefectResult with defect position and formation energy

    Raises:
        SimulationError: If simulation fails and raise_on_failure is True
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    defect_structure_file = Path(defect_structure_file)

    # Copy structure file to output dir
    struct_file = output_dir / "data.struct"
    struct_file.write_text(defect_structure_file.read_text())

    # Create config with existing structure
    config = MinimizationConfig(
        material=material,
        output_dir=output_dir,
    )

    # Generate and run simulation
    input_script = generate_minimization_script(config)
    sim_result = run_lammps(input_script, output_dir, lammps_cmd)

    if not sim_result.success:
        if raise_on_failure:
            raise SimulationError(
                f"Simulation failed: {sim_result.error_message}",
                result=sim_result
            )
        return DefectResult(
            position=(0.0, 0.0, 0.0),
            formation_energy=0.0,
            source_file=output_dir / "final.result"
        )

    # Post-process
    processor = LAMMPSPostProcessor()
    result_file = sim_result.result_file or (output_dir / "final.result")

    return processor.calculate_defect_energy(result_file, reference_energy)


def interface_energy_workflow(
    material: MaterialConfig,
    interface_structure_file: Path | str,
    reference_energy: CohesiveEnergyResult,
    output_dir: Path | str,
    lammps_cmd: Optional[str] = None,
    raise_on_failure: bool = True,
) -> InterfaceResult:
    """
    Calculate interface energy.

    Workflow:
    1. Run minimization on structure with interface
    2. Post-process to find interface and calculate interface energy

    Args:
        material: Material configuration (for potential selection)
        interface_structure_file: Path to structure file containing the interface
        reference_energy: Reference cohesive energy from bulk materials
        output_dir: Directory for simulation output
        lammps_cmd: LAMMPS executable (auto-detected if None)
        raise_on_failure: If True, raise SimulationError on failure

    Returns:
        InterfaceResult with interface position and energy (mJ/m^2)

    Raises:
        SimulationError: If simulation fails and raise_on_failure is True
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    interface_structure_file = Path(interface_structure_file)

    # Copy structure file to output dir
    struct_file = output_dir / "data.struct"
    struct_file.write_text(interface_structure_file.read_text())

    # Create config with existing structure
    config = MinimizationConfig(
        material=material,
        output_dir=output_dir,
    )

    # Generate and run simulation
    input_script = generate_minimization_script(config)
    sim_result = run_lammps(input_script, output_dir, lammps_cmd)

    if not sim_result.success:
        if raise_on_failure:
            raise SimulationError(
                f"Simulation failed: {sim_result.error_message}",
                result=sim_result
            )
        return InterfaceResult(
            position_y=0.0,
            interface_energy=0.0,
            source_file=output_dir / "final.result"
        )

    # Post-process
    processor = LAMMPSPostProcessor()
    result_file = sim_result.result_file or (output_dir / "final.result")

    return processor.calculate_interface_energy(result_file, reference_energy)


def relaxation_workflow(
    material: MaterialConfig,
    output_dir: Path | str,
    temperature: int = 300,
    n_steps: int = 10000,
    n_cells: int = 4,
    lammps_cmd: Optional[str] = None,
    raise_on_failure: bool = True,
) -> SimulationResult:
    """
    Run dynamic relaxation simulation.

    Args:
        material: Material configuration
        output_dir: Directory for simulation output
        temperature: Relaxation temperature in Kelvin
        n_steps: Number of MD steps
        n_cells: Number of unit cells in each direction
        lammps_cmd: LAMMPS executable (auto-detected if None)
        raise_on_failure: If True, raise SimulationError on failure

    Returns:
        SimulationResult with simulation outcome

    Raises:
        SimulationError: If simulation fails and raise_on_failure is True
    """
    from lmp_reproduced.simulations.runners import run_relaxation

    output_dir = Path(output_dir)

    config = RelaxationConfig(
        material=material,
        output_dir=output_dir,
        temperature=temperature,
        n_steps=n_steps,
    )

    result = run_relaxation(config, lammps_cmd, n_cells)

    if not result.success and raise_on_failure:
        raise SimulationError(
            f"Relaxation failed: {result.error_message}",
            result=result
        )

    return result


def interface_annealing_workflow(
    bottom_config: MaterialConfig,
    top_config: MaterialConfig,
    output_dir: Path | str,
    references: dict,
    max_temperature: float = 600.0,
    spatial_refs: bool = False,
    n_layers_bottom: int = 6,
    n_layers_top: int = 6,
    lammps_cmd: Optional[str] = None,
    raise_on_failure: bool = True,
) -> InterfaceResult:
    """
    Complete workflow: interface structure -> anneal -> minimize -> interface energy.
    
    This is the main entry point for interface energy calculations with
    annealing optimization.
    
    Args:
        bottom_config: Material configuration for bottom layer
        top_config: Material configuration for top layer
        output_dir: Directory for simulation output
        references: Reference energies for interface calculation.
            - For simple interfaces: {1: -3.36, 2: -1.51} (atom_type: energy)
            - For spatial refs: {"B_tib2": -7.58, "Ti_metal": -4.87, "Ti_tib2": -4.50}
        max_temperature: Maximum annealing temperature in K
        spatial_refs: If True, use spatial phase detection for multi-phase interfaces
        n_layers_bottom: Number of layers for bottom material
        n_layers_top: Number of layers for top material
        lammps_cmd: LAMMPS executable (auto-detected if None)
        raise_on_failure: If True, raise SimulationError on failure
        
    Returns:
        InterfaceResult with interface position and energy (mJ/m^2)
        
    Raises:
        SimulationError: If simulation fails and raise_on_failure is True
        
    Example:
        >>> from lmp_reproduced import ALUMINUM, MAGNESIUM, interface_annealing_workflow
        >>> result = interface_annealing_workflow(
        ...     bottom_config=ALUMINUM,
        ...     top_config=MAGNESIUM,
        ...     output_dir="./simulations/Al_Mg",
        ...     references={1: -3.36, 2: -1.51},
        ... )
        >>> print(f"Interface energy: {result.interface_energy:.2f} mJ/m²")
    """
    import subprocess
    from lmp_reproduced.core.structures import create_interface, write_lammps_data
    from lmp_reproduced.core.input_generator import LAMMPSInputGenerator, POTENTIAL_CONFIGS
    from lmp_reproduced.models import MaterialSystem, CohesiveEnergyResult
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create interface structure
    structure = create_interface(
        bottom_config, top_config,
        n_layers_bottom=n_layers_bottom,
        n_layers_top=n_layers_top,
    )
    data_file = output_dir / "interface.data"
    write_lammps_data(structure, data_file)
    
    # Step 2: Determine material system for potential
    # Use bottom material's system, or detect from combination
    system = bottom_config.system
    if bottom_config.element == "Al" and top_config.element == "Mg":
        system = MaterialSystem.AL_MG
    elif bottom_config.element in ("Ti", "TiB2") or top_config.element in ("Ti", "TiB2"):
        system = MaterialSystem.TI_B
    
    if system is None or system not in POTENTIAL_CONFIGS:
        raise ValueError(f"No potential configured for material system: {system}")
    
    # Step 3: Generate annealing script
    generator = LAMMPSInputGenerator(
        material_system=system,
        structure_file="interface.data",
    )
    script = generator.generate_annealing(
        max_temp=max_temperature,
        result_file="final.result",
    )
    
    input_file = output_dir / "in.anneal"
    input_file.write_text(script)
    
    # Step 4: Find and run LAMMPS
    if lammps_cmd is None:
        lammps_cmd = find_lammps()
    
    if lammps_cmd is None:
        if raise_on_failure:
            raise SimulationError("LAMMPS executable not found")
        return InterfaceResult(
            position_y=0.0,
            interface_energy=0.0,
            source_file=output_dir / "final.result"
        )
    
    try:
        log_file = output_dir / "log.lammps"
        with open(log_file, "w") as log:
            subprocess.check_call(
                [lammps_cmd, "-in", "in.anneal"],
                cwd=output_dir,
                stdout=log,
                stderr=subprocess.STDOUT
            )
    except subprocess.CalledProcessError as e:
        if raise_on_failure:
            raise SimulationError(f"LAMMPS simulation failed: {e}")
        return InterfaceResult(
            position_y=0.0,
            interface_energy=0.0,
            source_file=output_dir / "final.result"
        )
    
    # Step 5: Calculate interface energy
    processor = LAMMPSPostProcessor()
    result_file = output_dir / "final.result"
    
    if spatial_refs:
        return processor.calculate_interface_energy_spatial(
            result_file, references, metal_top=False
        )
    else:
        # Convert dict to CohesiveEnergyResult for standard method
        ref_obj = CohesiveEnergyResult(
            energies=references,
            n_atoms={k: 1 for k in references},
            source_file=Path("manual_reference")
        )
        return processor.calculate_interface_energy(result_file, ref_obj)
