"""
Data models for LAMMPS simulation configuration and results.

This module defines dataclasses for all inputs and outputs used throughout
the lmp_reproduced package, enabling composable, type-safe workflows.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class StructureType(Enum):
    """Crystal structure types."""
    FCC = "fcc"
    HCP = "hcp"
    BCC = "bcc"
    ZINCBLENDE = "zincblende"
    HEXAGONAL = "hexagonal"  # For compounds like TiB2


class SimulationType(Enum):
    """Simulation type selection."""
    MINIMIZE = "minimize"
    NEB = "neb"
    RELAXATION = "relaxation"
    LOADING_STRESS = "loading_stress"
    LOADING_STRAIN = "loading_strain"


class MaterialSystem(Enum):
    """Supported material systems for potential selection."""
    AL_MG = "al_mg"
    C_SI_AL = "c_si_al"
    CU_C = "cu_c"
    TI_B = "ti_b"
    SI_C = "si_c"
    AL = "al"
    CU = "cu"
    MG = "mg"
    TI = "ti"
    TIB2 = "tib2"


class PotentialType(Enum):
    """Types of interatomic potentials."""
    EAM_ALLOY = "eam/alloy"
    EAM_FS = "eam/fs"
    MEAM = "meam/c"
    TERSOFF = "tersoff"
    LJ = "lj/cut"
    MORSE = "morse"
    AIREBO = "airebo"
    HYBRID = "hybrid"


@dataclass
class PotentialConfig:
    """Configuration for interatomic potential."""
    type: PotentialType
    filename: str
    elements: list[str]  # Element mapping for pair_coeff
    parameters: Optional[dict] = None  # Extra params like morse coeffs



# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class MaterialConfig:
    """Configuration for a material system."""
    element: str
    lattice_a: float
    structure_type: StructureType
    lattice_c: Optional[float] = None  # For HCP/hex systems
    system: Optional[MaterialSystem] = None  # For potential selection

    def __post_init__(self):
        """Auto-detect material system if not provided."""
        if self.system is None:
            system_map = {
                "Al": MaterialSystem.AL,
                "Cu": MaterialSystem.CU,
                "Mg": MaterialSystem.MG,
                "Ti": MaterialSystem.TI,
                "TiB2": MaterialSystem.TIB2,
                "SiC": MaterialSystem.SI_C,
            }
            self.system = system_map.get(self.element)


@dataclass
class MinimizationConfig:
    """Configuration for energy minimization simulation."""
    material: MaterialConfig
    output_dir: Path
    etol: float = 0.0
    ftol: float = 3.12e-3
    maxiter: int = 100000
    maxeval: int = 100000
    fix_start: int = 0
    fix_end: int = 0


@dataclass
class NEBConfig:
    """Configuration for NEB (Nudged Elastic Band) calculation."""
    material: MaterialConfig
    output_dir: Path
    final_config_file: Path
    spring_constant: float = 1.0
    etol: float = 0.0
    ftol: float = 0.001
    n_images: int = 21
    n_climb: int = 500
    n_final: int = 50
    fix_start: int = 0
    fix_end: int = 0


@dataclass
class RelaxationConfig:
    """Configuration for dynamic relaxation simulation."""
    material: MaterialConfig
    output_dir: Path
    temperature: int = 300
    n_steps: int = 10000
    dump_interval: int = 0
    fix_start: int = 0
    fix_end: int = 0


@dataclass
class StressLoadingConfig:
    """Configuration for constant stress loading simulation."""
    material: MaterialConfig
    output_dir: Path
    temperature: int = 300
    stress_xx: float = 0.0
    stress_yy: float = 0.0
    stress_zz: float = 0.0
    stress_xy: float = 0.0
    stress_xz: float = 0.0
    stress_yz: float = 0.0
    n_steps: int = 100000
    dump_interval: int = 2000
    fix_start: int = 0
    fix_end: int = 0


@dataclass
class StrainLoadingConfig:
    """Configuration for constant strain rate loading simulation."""
    material: MaterialConfig
    output_dir: Path
    temperature: int = 300
    deform_direction: str = "xy"
    strain_rate: float = 5e-4
    n_steps: int = 200000
    dump_interval: int = 2000
    fix_start: int = 0
    fix_end: int = 0


@dataclass
class AnnealingConfig:
    """Configuration for interface annealing simulation.
    
    This configuration is used for interface energy calculations where
    the structure needs to be annealed before minimization.
    """
    material_bottom: MaterialConfig
    material_top: MaterialConfig
    output_dir: Path
    max_temperature: float = 600.0  # K
    heat_steps: int = 1000
    hold_steps: int = 2000
    cool_steps: int = 2000
    timestep: float = 0.001  # ps (1 fs)


# ============================================================================
# Structure Data
# ============================================================================

@dataclass
class Atom:
    """Single atom data."""
    id: int
    type: int
    x: float
    y: float
    z: float


@dataclass
class StructureData:
    """Crystal structure data for LAMMPS."""
    title: str
    atoms: list[Atom]
    n_types: int
    box_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    masses: dict[int, float]  # type -> mass


# ============================================================================
# Result Dataclasses
# ============================================================================

@dataclass
class SimulationResult:
    """Result from running a LAMMPS simulation."""
    success: bool
    output_dir: Path
    log_file: Optional[Path] = None
    result_file: Optional[Path] = None
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


@dataclass
class CohesiveEnergyResult:
    """Result from cohesive energy calculation."""
    energies: dict[int, float]  # atom_type -> energy per atom (eV)
    n_atoms: dict[int, int]  # atom_type -> count
    source_file: Path

    @property
    def primary_energy(self) -> float:
        """Get energy of the first (or only) atom type."""
        return list(self.energies.values())[0]

    @property
    def total_atoms(self) -> int:
        """Get total number of atoms."""
        return sum(self.n_atoms.values())


@dataclass
class DefectResult:
    """Result from defect formation energy calculation."""
    position: tuple[float, float, float]  # x, y, z in Angstrom
    formation_energy: float  # eV
    source_file: Path


@dataclass
class InterfaceResult:
    """Result from interface energy calculation."""
    position_y: float  # Angstrom
    interface_energy: float  # mJ/m^2
    source_file: Path


@dataclass
class DislocationResult:
    """Result from dislocation core energy calculation."""
    position: tuple[float, float]  # x, y in Angstrom
    core_width: float  # Angstrom
    core_energy: float  # eV/Angstrom
    source_file: Path


@dataclass
class NEBResult:
    """Result from NEB calculation."""
    barrier: float  # eV
    n_images: int
    source_file: Path


# ============================================================================
# Document Extraction Models
# ============================================================================

@dataclass
class Paragraph:
    """Extracted paragraph from a document."""
    text: str
    style: Optional[str] = None


@dataclass
class Table:
    """Extracted table from a document."""
    index: int
    rows: list[list[str]]


@dataclass
class DocumentContent:
    """Extracted content from a Word document."""
    paragraphs: list[Paragraph]
    tables: list[Table]
    images_count: int
    source_file: Path


# ============================================================================
# Predefined Material Configurations
# ============================================================================

# Common materials with values from technical report:
# "金属基复合材料温度相关物性计算模块技术报告" (Report 1-2)

ALUMINUM = MaterialConfig(
    element="Al",
    lattice_a=4.032,  # Report 1-2: a = 4.032 Å, Ecoh = -3.36 eV
    structure_type=StructureType.FCC,
    system=MaterialSystem.AL
)

COPPER = MaterialConfig(
    element="Cu",
    lattice_a=3.615,
    structure_type=StructureType.FCC,
    system=MaterialSystem.CU
)

MAGNESIUM = MaterialConfig(
    element="Mg",
    lattice_a=3.196,  # Report 1-2: a = 3.196 Å, c = 5.197 Å, Ecoh = -1.51 eV
    lattice_c=5.197,
    structure_type=StructureType.HCP,
    system=MaterialSystem.MG
)

TITANIUM = MaterialConfig(
    element="Ti",
    lattice_a=2.945,  # Report 1-2: a = 2.945 Å, c = 4.687 Å, Ecoh = -4.873 eV
    lattice_c=4.687,
    structure_type=StructureType.HCP,
    system=MaterialSystem.TI
)

SILICON_CARBIDE = MaterialConfig(
    element="SiC",
    lattice_a=4.36,
    structure_type=StructureType.ZINCBLENDE,
    system=MaterialSystem.SI_C
)

TIB2 = MaterialConfig(
    element="TiB2",
    lattice_a=3.050,  # Report 1-2: a = 3.050 Å, c = 3.197 Å
    lattice_c=3.197,  # Ecoh: B = -7.580 eV, Ti = -4.497 eV
    structure_type=StructureType.HEXAGONAL,
    system=MaterialSystem.TIB2
)
