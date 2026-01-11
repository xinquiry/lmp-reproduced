"""
LAMMPS Input Generator Module

Generates LAMMPS input files for metal matrix composite interface and defect
calculations.

Supports:
- Molecular Statics: Minimization, NEB calculations
- Molecular Dynamics: Relaxation, Constant stress/strain rate loading
- Material Systems: Al-Mg, C-Si-Al, Cu-C, Ti-B
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lmp_reproduced.models import (
    MaterialSystem,
    MinimizationConfig,
    NEBConfig,
    RelaxationConfig,
    StressLoadingConfig,
    StrainLoadingConfig,
)
from lmp_reproduced.core.potentials import get_potential_dir


@dataclass
class PotentialConfig:
    """Potential function configuration for each material system."""
    pair_style: str
    pair_coeffs: list[str]
    neighbor: str = "2.0 bin"
    neigh_modify: str = "every 1 delay 0 check yes"


# Potential configurations for each material system
# Note: Paths use {POT_DIR} placeholder which gets resolved at script generation time
POTENTIAL_CONFIGS = {
    MaterialSystem.AL_MG: PotentialConfig(
        pair_style="eam/alloy",
        pair_coeffs=["* * {POT_DIR}/almg.liu.eam.alloy Al Mg"]
    ),
    MaterialSystem.C_SI_AL: PotentialConfig(
        pair_style="hybrid tersoff eam/alloy morse 3.0",
        pair_coeffs=[
            "* * tersoff {POT_DIR}/SiC.tersoff C Si NULL",
            "eam/alloy {POT_DIR}/Al_wkg_MSMSE_2009.set NULL NULL Al",
            "1 3 morse 0.4691 1.738 2.246",
            "2 3 morse 0.4824 1.322 2.92"
        ]
    ),
    MaterialSystem.CU_C: PotentialConfig(
        pair_style="hybrid lj/cut 8.0 eam airebo 3.0",
        pair_coeffs=[
            "1 2 lj/cut 0.02578 3.0825",
            "* * eam {POT_DIR}/Cu_u3.eam",
            "* * airebo {POT_DIR}/CH.airebo NULL C"
        ]
    ),
    MaterialSystem.TI_B: PotentialConfig(
        pair_style="meam",
        pair_coeffs=["* * {POT_DIR}/library.meam B Ti {POT_DIR}/TiB.meam B Ti"]
    ),
    MaterialSystem.SI_C: PotentialConfig(
        pair_style="tersoff",
        pair_coeffs=["* * {POT_DIR}/SiC.tersoff Si C"]
    ),
    MaterialSystem.AL: PotentialConfig(
        pair_style="eam/alloy",
        pair_coeffs=["* * {POT_DIR}/AlCu.eam.alloy Al"]
    ),
    MaterialSystem.CU: PotentialConfig(
        pair_style="eam/alloy",
        pair_coeffs=["* * {POT_DIR}/AlCu.eam.alloy Cu"]
    ),
    MaterialSystem.MG: PotentialConfig(
        pair_style="eam/fs",
        pair_coeffs=["* * {POT_DIR}/Mg_mm.eam.fs Mg"]
    ),
    MaterialSystem.TI: PotentialConfig(
        pair_style="meam",
        pair_coeffs=["* * {POT_DIR}/library.meam B Ti {POT_DIR}/TiB.meam B Ti"]
    ),
    MaterialSystem.TIB2: PotentialConfig(
        pair_style="meam",
        pair_coeffs=["* * {POT_DIR}/library.meam B Ti {POT_DIR}/TiB.meam B Ti"]
    )
}


class LAMMPSInputGenerator:
    """
    Generate LAMMPS input files for metal matrix composite simulations.

    This class provides methods to generate various types of LAMMPS input
    scripts for molecular statics and dynamics simulations.
    """

    def __init__(
        self,
        material_system: MaterialSystem,
        structure_file: str = "data/struct/data.struct",
        potential_dir: Optional[Path] = None
    ):
        """
        Initialize generator with material system.

        Args:
            material_system: The material system to use for potential functions
            structure_file: Path to the structure data file
            potential_dir: Path to potential files directory. If None, uses
                          LMP_POTENTIAL_DIR env var or default data/pot location
        """
        self.material_system = material_system
        self.potential = POTENTIAL_CONFIGS[material_system]
        self.structure_file = structure_file
        self.potential_dir = potential_dir or get_potential_dir()

    def _write_header(self, lines: list[str], title: str) -> None:
        """Write common LAMMPS header."""
        lines.append(f"#{title}")
        lines.append("units          metal")
        lines.append("atom_style     atomic")
        lines.append("boundary       p p p")
        lines.append("timestep       0.001")
        lines.append(f"read_data      {self.structure_file}")
        lines.append(" ")

    def _write_atom_groups(
        self,
        lines: list[str],
        fix_start: int,
        fix_end: int
    ) -> None:
        """Write atom group definitions for fixed and mobile regions."""
        if fix_start == 0 and fix_end == 0:
            lines.append("group          region1 id 0:0")
        else:
            lines.append(f"group          region1 id {fix_start}:{fix_end}")
        lines.append("group          region2 subtract all region1")
        lines.append(" ")

    def _write_potential(self, lines: list[str]) -> None:
        """Write potential function settings with resolved paths."""
        lines.append(f"pair_style     {self.potential.pair_style}")
        pot_dir_str = str(self.potential_dir)
        for coeff in self.potential.pair_coeffs:
            resolved_coeff = coeff.replace("{POT_DIR}", pot_dir_str)
            lines.append(f"pair_coeff     {resolved_coeff}")
        lines.append(f"neighbor       {self.potential.neighbor}")
        lines.append(f"neigh_modify   {self.potential.neigh_modify}")

    def _write_thermo(self, lines: list[str], interval: int = 100) -> None:
        """Write thermodynamic output settings."""
        lines.append(f"thermo         {interval}")
        lines.append("thermo_style   custom step temp etotal pxx pyy pzz pxy pxz pyz")
        lines.append(" ")

    def generate_minimize(
        self,
        fix_start: int = 0,
        fix_end: int = 0,
        etol: float = 0.0,
        ftol: float = 3.12e-3,
        maxiter: int = 100000,
        maxeval: int = 100000,
        result_file: str = "data/results/final.result",
    ) -> str:
        """
        Generate LAMMPS input script for energy minimization.

        Args:
            fix_start: Starting atom ID of fixed region
            fix_end: Ending atom ID of fixed region
            etol: Energy tolerance for convergence
            ftol: Force tolerance for convergence
            maxiter: Maximum iterations
            maxeval: Maximum force evaluations
            result_file: Path for output result file

        Returns:
            Generated LAMMPS input script as string
        """
        lines = []

        self._write_header(lines, "Minimize structures")
        self._write_atom_groups(lines, fix_start, fix_end)
        self._write_potential(lines)
        lines.append(" ")

        lines.append("compute        pot all pe/atom")
        self._write_thermo(lines)

        if fix_start > 0 or fix_end > 0:
            lines.append("fix            freeze region1 setforce 0 0 0")
        lines.append("min_style      fire")
        lines.append(f"minimize       {etol} {ftol} {maxiter} {maxeval}")
        lines.append(f"write_dump     all custom {result_file} id type x y z c_pot modify sort id")
        lines.append("#End of file")

        return "\n".join(lines)

    def generate_neb(
        self,
        fix_start: int = 0,
        fix_end: int = 0,
        spring_constant: float = 1.0,
        etol: float = 0.0,
        ftol: float = 0.001,
        n_images: int = 21,
        n_climb: int = 500,
        n_final: int = 50,
        final_config: str = "data.final",
    ) -> str:
        """
        Generate LAMMPS input script for NEB (Nudged Elastic Band) calculation.

        Args:
            fix_start: Starting atom ID of fixed region
            fix_end: Ending atom ID of fixed region
            spring_constant: NEB spring constant
            etol: Energy tolerance
            ftol: Force tolerance
            n_images: Number of NEB images (including initial/final)
            n_climb: Climbing image iterations
            n_final: Final iterations
            final_config: Path to final configuration file

        Returns:
            Generated LAMMPS input script as string
        """
        lines = []

        self._write_header(lines, "Perform NEB calculation")
        self._write_atom_groups(lines, fix_start, fix_end)
        self._write_potential(lines)

        lines.append("thermo         100")
        lines.append(" ")

        if fix_start > 0 or fix_end > 0:
            lines.append("fix            freeze region1 setforce 0 0 0")
        lines.append("min_style      fire")
        lines.append(f"fix            2 all neb {spring_constant} parallel ideal")
        lines.append(f"neb            {etol} {ftol} 1000 {n_climb} {n_final} final {final_config}")
        lines.append("#End of file")

        return "\n".join(lines)

    def generate_relaxation(
        self,
        temperature: int = 300,
        fix_start: int = 0,
        fix_end: int = 0,
        n_steps: int = 10000,
        dump_interval: int = 0,
    ) -> str:
        """
        Generate LAMMPS input script for dynamic relaxation.

        Args:
            temperature: Relaxation temperature in Kelvin
            fix_start: Starting atom ID of fixed region
            fix_end: Ending atom ID of fixed region
            n_steps: Number of MD steps
            dump_interval: Dump interval (0 for no intermediate dump)

        Returns:
            Generated LAMMPS input script as string
        """
        lines = []

        self._write_header(lines, "Dynamic relaxation")
        self._write_atom_groups(lines, fix_start, fix_end)
        self._write_potential(lines)
        lines.append(" ")

        if fix_start > 0 or fix_end > 0:
            lines.append("fix            freeze region1 setforce 0 0 0")

        lines.append(f"velocity       region2 create {temperature} 123456 units box")
        lines.append(f"fix            relax region2 nvt temp {temperature} {temperature} 0.1")
        lines.append(" ")

        lines.append("compute        pot all pe/atom")
        self._write_thermo(lines)

        if dump_interval > 0:
            lines.append(f"dump           1 all custom {dump_interval} dump.*.result id type x y z c_pot")
            lines.append("dump_modify    1 sort id")

        lines.append(f"run            {n_steps}")
        lines.append("write_data     final.data")
        lines.append("#End of file")

        return "\n".join(lines)

    def generate_temperature_relaxation(
        self,
        temperature: int = 300,
        equil_steps: int = 5000,
        prod_steps: int = 5000,
        result_file: str = "final.result",
    ) -> str:
        """
        Generate LAMMPS script for temperature-dependent cohesive energy.
        
        Workflow:
        1. Equilibrate at target temperature (NPT)
        2. Production run to sample average energy
        3. Output final configuration with per-atom PE
        
        Args:
            temperature: Target temperature in Kelvin
            equil_steps: Number of equilibration steps
            prod_steps: Number of production steps
            result_file: Output file for final configuration
            
        Returns:
            Generated LAMMPS input script as string
        """
        lines = []
        
        # Header
        lines.append(f"# Temperature-dependent relaxation at {temperature}K")
        lines.append("units metal")
        lines.append("atom_style atomic")
        lines.append("boundary p p p")
        lines.append("timestep 0.001")
        lines.append(f"read_data {self.structure_file}")
        lines.append("")
        
        # Potential
        self._write_potential(lines)
        lines.append("")
        
        # Initialize velocities
        lines.append(f"velocity all create {temperature} 12345 dist gaussian")
        lines.append("")
        
        # Equilibration (NPT)
        lines.append("# Equilibration")
        lines.append(f"fix 1 all npt temp {temperature} {temperature} 0.1 iso 0 0 1.0")
        lines.append("thermo 100")
        lines.append("thermo_style custom step temp pe ke etotal press vol")
        lines.append(f"run {equil_steps}")
        lines.append("unfix 1")
        lines.append("")
        
        # Production (NVT for stable volume)
        lines.append("# Production run")
        lines.append(f"fix 2 all nvt temp {temperature} {temperature} 0.1")
        lines.append(f"run {prod_steps}")
        lines.append("unfix 2")
        lines.append("")
        
        # Output with pe/atom
        lines.append("# Output final configuration with per-atom PE")
        lines.append("compute pe all pe/atom")
        lines.append(f"dump 1 all custom 1 {result_file} id type x y z c_pe")
        lines.append("run 0")
        lines.append("undump 1")
        lines.append("write_data data.final")
        lines.append("# End of file")
        
        return "\n".join(lines)

    def generate_loading_constant_stress(
        self,
        temperature: int = 300,
        fix_start: int = 0,
        fix_end: int = 0,
        stress_xx: float = 0.0,
        stress_yy: float = 0.0,
        stress_zz: float = 0.0,
        stress_xy: float = 0.0,
        stress_xz: float = 0.0,
        stress_yz: float = 0.0,
        n_steps: int = 100000,
        dump_interval: int = 2000,
    ) -> str:
        """
        Generate LAMMPS input script for constant stress loading.

        Args:
            temperature: Loading temperature in Kelvin
            fix_start: Starting atom ID of fixed region
            fix_end: Ending atom ID of fixed region
            stress_xx, stress_yy, stress_zz: Normal stress components (bars)
            stress_xy, stress_xz, stress_yz: Shear stress components (bars)
            n_steps: Number of MD steps
            dump_interval: Dump interval for trajectory output

        Returns:
            Generated LAMMPS input script as string
        """
        lines = []

        self._write_header(lines, "Constant stress loading")
        self._write_atom_groups(lines, fix_start, fix_end)
        self._write_potential(lines)

        lines.append("change_box     all triclinic")

        if fix_start > 0 or fix_end > 0:
            lines.append("fix            freeze region1 setforce 0 0 0")

        lines.append(f"velocity       region2 create {temperature} 123456 units box")

        stress_str = (
            f"fix            load region2 npt temp {temperature} {temperature} 0.1 "
            f"x {stress_xx} {stress_xx} 0.1 "
            f"y {stress_yy} {stress_yy} 0.1 "
            f"z {stress_zz} {stress_zz} 0.1 "
            f"xy {stress_xy} {stress_xy} 0.1 "
            f"xz {stress_xz} {stress_xz} 0.1 "
            f"yz {stress_yz} {stress_yz} 0.1"
        )
        lines.append(stress_str)
        lines.append(" ")

        lines.append("compute        pot all pe/atom")
        self._write_thermo(lines)

        lines.append(f"dump           1 all custom {dump_interval} final.result id type x y z c_pot")
        lines.append("dump_modify    1 sort id")
        lines.append(f"run            {n_steps}")
        lines.append("#End of file")

        return "\n".join(lines)

    def generate_loading_constant_strain_rate(
        self,
        temperature: int = 300,
        fix_start: int = 0,
        fix_end: int = 0,
        deform_direction: str = "xy",
        strain_rate: float = 5e-4,
        n_steps: int = 200000,
        dump_interval: int = 2000,
    ) -> str:
        """
        Generate LAMMPS input script for constant strain rate loading.

        Args:
            temperature: Loading temperature in Kelvin
            fix_start: Starting atom ID of fixed region
            fix_end: Ending atom ID of fixed region
            deform_direction: Deformation direction (x, y, z, xy, xz, yz)
            strain_rate: Engineering strain rate per timestep
            n_steps: Number of MD steps
            dump_interval: Dump interval for trajectory output

        Returns:
            Generated LAMMPS input script as string
        """
        lines = []

        self._write_header(lines, "Constant strain rate loading")
        self._write_atom_groups(lines, fix_start, fix_end)
        self._write_potential(lines)

        lines.append("change_box     all triclinic")

        if fix_start > 0 or fix_end > 0:
            lines.append("fix            freeze region1 setforce 0 0 0")

        lines.append(f"velocity       region2 create {temperature} 123456 units box")
        lines.append(f"fix            nvt_fix region2 nvt temp {temperature} {temperature} 0.1")
        lines.append(f"fix            deform_fix all deform 1 {deform_direction} erate {strain_rate} remap x")
        lines.append(" ")

        lines.append("compute        pot all pe/atom")
        self._write_thermo(lines)

        lines.append(f"dump           1 all custom {dump_interval} dump.*.result id type x y z c_pot")
        lines.append("dump_modify    1 sort id")
        lines.append(f"run            {n_steps}")
        lines.append("#End of file")

        return "\n".join(lines)

    def generate_annealing(
        self,
        max_temp: float = 600.0,
        heat_steps: int = 1000,
        hold_steps: int = 2000,
        cool_steps: int = 2000,
        result_file: str = "final.result",
    ) -> str:
        """
        Generate interface annealing + minimization script.
        
        The workflow is:
        1. Heat from 10K to max_temp
        2. Hold at max_temp
        3. Cool from max_temp to 10K
        4. Final minimization with FIRE algorithm
        5. Output with per-atom potential energy
        
        Args:
            max_temp: Maximum annealing temperature in K
            heat_steps: Number of heating steps
            hold_steps: Number of hold steps at max temp
            cool_steps: Number of cooling steps
            result_file: Path for output result file
            
        Returns:
            Generated LAMMPS input script as string
        """
        lines = []
        
        # Header
        lines.append("# Interface Annealing & Minimization")
        lines.append("units metal")
        lines.append("atom_style atomic")
        lines.append("boundary p p p")
        lines.append(f"read_data {self.structure_file}")
        lines.append("")
        
        # Potential
        self._write_potential(lines)
        lines.append("")
        
        # Annealing
        lines.append("timestep 0.001")
        lines.append("velocity all create 10.0 12345 dist gaussian")
        lines.append("")
        
        # Heat
        lines.append(f"fix 1 all npt temp 10.0 {max_temp} 0.1 x 0 0 1.0 z 0 0 1.0")
        lines.append(f"run {heat_steps}")
        lines.append("")
        
        # Hold
        lines.append(f"fix 1 all npt temp {max_temp} {max_temp} 0.1 x 0 0 1.0 z 0 0 1.0")
        lines.append(f"run {hold_steps}")
        lines.append("")
        
        # Cool
        lines.append(f"fix 1 all npt temp {max_temp} 10.0 0.1 x 0 0 1.0 z 0 0 1.0")
        lines.append(f"run {cool_steps}")
        lines.append("")
        
        lines.append("unfix 1")
        lines.append("")
        
        # Final minimization
        lines.append("min_style fire")
        lines.append("minimize 1e-12 1e-12 20000 20000")
        lines.append("")
        
        # Output with pe/atom
        lines.append("compute pe all pe/atom")
        lines.append(f"dump 1 all custom 1 {result_file} id type x y z c_pe")
        lines.append("run 0")
        lines.append("undump 1")
        lines.append("write_data data.final")
        lines.append("# End of file")
        
        return "\n".join(lines)


# ============================================================================
# Convenience functions using dataclass configs
# ============================================================================

def generate_minimization_script(
    config: MinimizationConfig,
    potential_dir: Optional[Path] = None
) -> str:
    """
    Generate minimization input script from configuration.

    Args:
        config: MinimizationConfig with all parameters
        potential_dir: Optional custom potential directory

    Returns:
        LAMMPS input script as string
    """
    if config.material.system is None:
        raise ValueError("Material system must be specified")

    # Use relative filenames since LAMMPS runs from output_dir
    struct_file = "data.struct"
    result_file = "final.result"

    generator = LAMMPSInputGenerator(
        material_system=config.material.system,
        structure_file=struct_file,
        potential_dir=potential_dir
    )

    return generator.generate_minimize(
        fix_start=config.fix_start,
        fix_end=config.fix_end,
        etol=config.etol,
        ftol=config.ftol,
        maxiter=config.maxiter,
        maxeval=config.maxeval,
        result_file=result_file,
    )


def generate_neb_script(
    config: NEBConfig,
    potential_dir: Optional[Path] = None
) -> str:
    """
    Generate NEB input script from configuration.

    Args:
        config: NEBConfig with all parameters
        potential_dir: Optional custom potential directory

    Returns:
        LAMMPS input script as string
    """
    if config.material.system is None:
        raise ValueError("Material system must be specified")

    # Use relative filenames since LAMMPS runs from output_dir
    struct_file = "data.struct"

    generator = LAMMPSInputGenerator(
        material_system=config.material.system,
        structure_file=struct_file,
        potential_dir=potential_dir
    )

    return generator.generate_neb(
        fix_start=config.fix_start,
        fix_end=config.fix_end,
        spring_constant=config.spring_constant,
        etol=config.etol,
        ftol=config.ftol,
        n_images=config.n_images,
        n_climb=config.n_climb,
        n_final=config.n_final,
        final_config=config.final_config_file.name,  # Use just filename
    )


def generate_relaxation_script(
    config: RelaxationConfig,
    potential_dir: Optional[Path] = None
) -> str:
    """
    Generate relaxation input script from configuration.

    Args:
        config: RelaxationConfig with all parameters
        potential_dir: Optional custom potential directory

    Returns:
        LAMMPS input script as string
    """
    if config.material.system is None:
        raise ValueError("Material system must be specified")

    # Use relative filenames since LAMMPS runs from output_dir
    struct_file = "data.struct"

    generator = LAMMPSInputGenerator(
        material_system=config.material.system,
        structure_file=struct_file,
        potential_dir=potential_dir
    )

    return generator.generate_relaxation(
        temperature=config.temperature,
        fix_start=config.fix_start,
        fix_end=config.fix_end,
        n_steps=config.n_steps,
        dump_interval=config.dump_interval,
    )


def generate_stress_loading_script(
    config: StressLoadingConfig,
    potential_dir: Optional[Path] = None
) -> str:
    """
    Generate constant stress loading input script from configuration.

    Args:
        config: StressLoadingConfig with all parameters
        potential_dir: Optional custom potential directory

    Returns:
        LAMMPS input script as string
    """
    if config.material.system is None:
        raise ValueError("Material system must be specified")

    # Use relative filenames since LAMMPS runs from output_dir
    struct_file = "data.struct"

    generator = LAMMPSInputGenerator(
        material_system=config.material.system,
        structure_file=struct_file,
        potential_dir=potential_dir
    )

    return generator.generate_loading_constant_stress(
        temperature=config.temperature,
        fix_start=config.fix_start,
        fix_end=config.fix_end,
        stress_xx=config.stress_xx,
        stress_yy=config.stress_yy,
        stress_zz=config.stress_zz,
        stress_xy=config.stress_xy,
        stress_xz=config.stress_xz,
        stress_yz=config.stress_yz,
        n_steps=config.n_steps,
        dump_interval=config.dump_interval,
    )


def generate_strain_loading_script(
    config: StrainLoadingConfig,
    potential_dir: Optional[Path] = None
) -> str:
    """
    Generate constant strain rate loading input script from configuration.

    Args:
        config: StrainLoadingConfig with all parameters
        potential_dir: Optional custom potential directory

    Returns:
        LAMMPS input script as string
    """
    if config.material.system is None:
        raise ValueError("Material system must be specified")

    # Use relative filenames since LAMMPS runs from output_dir
    struct_file = "data.struct"

    generator = LAMMPSInputGenerator(
        material_system=config.material.system,
        structure_file=struct_file,
        potential_dir=potential_dir
    )

    return generator.generate_loading_constant_strain_rate(
        temperature=config.temperature,
        fix_start=config.fix_start,
        fix_end=config.fix_end,
        deform_direction=config.deform_direction,
        strain_rate=config.strain_rate,
        n_steps=config.n_steps,
        dump_interval=config.dump_interval,
    )
