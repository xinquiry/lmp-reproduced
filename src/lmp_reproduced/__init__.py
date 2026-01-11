"""
lmp_reproduced - LAMMPS Simulation Framework for Metal Matrix Composites

A Python API for running LAMMPS simulations with composable workflows
and dataclass-based configuration.

Example usage:
    >>> from lmp_reproduced import MaterialConfig, cohesive_energy_workflow, StructureType
    >>> config = MaterialConfig(element="Al", lattice_a=4.05, structure_type=StructureType.FCC)
    >>> result = cohesive_energy_workflow(config, output_dir="./output")
    >>> print(f"Cohesive energy: {result.primary_energy:.4f} eV/atom")
"""

# Models and configuration
from lmp_reproduced.models import (
    # Enums
    StructureType,
    SimulationType,
    MaterialSystem,
    # Configuration dataclasses
    MaterialConfig,
    MinimizationConfig,
    NEBConfig,
    RelaxationConfig,
    StressLoadingConfig,
    StrainLoadingConfig,
    AnnealingConfig,
    # Structure data
    Atom,
    StructureData,
    # Result dataclasses
    SimulationResult,
    CohesiveEnergyResult,
    DefectResult,
    InterfaceResult,
    DislocationResult,
    NEBResult,
    # Document extraction
    Paragraph,
    Table,
    DocumentContent,
    # Predefined materials
    ALUMINUM,
    COPPER,
    MAGNESIUM,
    TITANIUM,
    SILICON_CARBIDE,
    TIB2,
)

# High-level workflow pipelines
from lmp_reproduced.pipelines import (
    cohesive_energy_workflow,
    batch_cohesive_energy,
    defect_energy_workflow,
    interface_energy_workflow,
    interface_annealing_workflow,
    relaxation_workflow,
    SimulationError,
)

# Core functionality
from lmp_reproduced.core.structures import (
    create_structure,
    create_fcc_structure,
    create_hcp_structure,
    create_hexagonal_structure,
    create_zincblende_structure,
    create_fcc_substitutional,
    create_hcp_vacancy,
    write_lammps_data,
)

from lmp_reproduced.core.input_generator import (
    LAMMPSInputGenerator,
    PotentialConfig,
    POTENTIAL_CONFIGS,
    generate_minimization_script,
    generate_neb_script,
    generate_relaxation_script,
    generate_stress_loading_script,
    generate_strain_loading_script,
)

from lmp_reproduced.core.post_processor import (
    LAMMPSPostProcessor,
)

from lmp_reproduced.core.potentials import (
    get_potential_dir,
    resolve_potential_path,
    list_available_potentials,
    get_potential_info,
)

from lmp_reproduced.core.extraction import (
    extract_document,
    save_extracted_content,
    extract_all_documents,
)

# Simulation runners
from lmp_reproduced.simulations.runners import (
    find_lammps,
    run_lammps,
    run_minimization,
    run_relaxation,
    run_single_material_simulation,
)

__version__ = "0.2.0"

__all__ = [
    # Version
    "__version__",
    # Enums
    "StructureType",
    "SimulationType",
    "MaterialSystem",
    # Configuration
    "MaterialConfig",
    "MinimizationConfig",
    "NEBConfig",
    "RelaxationConfig",
    "StressLoadingConfig",
    "StrainLoadingConfig",
    "AnnealingConfig",
    # Structure
    "Atom",
    "StructureData",
    # Results
    "SimulationResult",
    "CohesiveEnergyResult",
    "DefectResult",
    "InterfaceResult",
    "DislocationResult",
    "NEBResult",
    # Document
    "Paragraph",
    "Table",
    "DocumentContent",
    # Predefined materials
    "ALUMINUM",
    "COPPER",
    "MAGNESIUM",
    "TITANIUM",
    "SILICON_CARBIDE",
    "TIB2",
    # Pipelines
    "cohesive_energy_workflow",
    "batch_cohesive_energy",
    "defect_energy_workflow",
    "interface_energy_workflow",
    "interface_annealing_workflow",
    "relaxation_workflow",
    "SimulationError",
    # Structure generation
    "create_structure",
    "create_fcc_structure",
    "create_hcp_structure",
    "create_hexagonal_structure",
    "create_zincblende_structure",
    "create_fcc_substitutional",
    "create_hcp_vacancy",
    "write_lammps_data",
    # Input generation
    "LAMMPSInputGenerator",
    "PotentialConfig",
    "POTENTIAL_CONFIGS",
    "generate_minimization_script",
    "generate_neb_script",
    "generate_relaxation_script",
    "generate_stress_loading_script",
    "generate_strain_loading_script",
    # Post-processing
    "LAMMPSPostProcessor",
    # Potentials
    "get_potential_dir",
    "resolve_potential_path",
    "list_available_potentials",
    "get_potential_info",
    # Document extraction
    "extract_document",
    "save_extracted_content",
    "extract_all_documents",
    # Runners
    "find_lammps",
    "run_lammps",
    "run_minimization",
    "run_relaxation",
    "run_single_material_simulation",
]
