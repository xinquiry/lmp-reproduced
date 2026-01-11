"""
LAMMPS Post-Processor Module

Extracts material properties from LAMMPS output files.

Supports:
- Cohesive energy calculation from single crystal data
- Defect formation energy calculation
- Interface energy calculation
- Dislocation core energy calculation
- NEB migration barrier calculation
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from lmp_reproduced.models import (
    CohesiveEnergyResult,
    DefectResult,
    DislocationResult,
    InterfaceResult,
    NEBResult,
)


@dataclass
class AtomData:
    """Container for atom data from LAMMPS output."""
    n_atoms: int
    atom_types: np.ndarray  # shape: (n_atoms,)
    positions: np.ndarray   # shape: (n_atoms, 3)
    potentials: np.ndarray  # shape: (n_atoms,)
    box_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    n_types: int


class LAMMPSPostProcessor:
    """
    Post-process LAMMPS output files to extract material properties.

    This class provides methods to calculate various material properties
    from LAMMPS simulation output files.
    """

    def __init__(self):
        """Initialize post-processor."""
        pass

    def _read_lammps_dump(self, filename: str) -> AtomData:
        """
        Read LAMMPS dump format file (with id type x y z pe).

        Args:
            filename: Path to LAMMPS dump file

        Returns:
            AtomData containing parsed atomic information
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        idx = 0
        n_atoms = 0
        box_bounds = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

        while idx < len(lines):
            line = lines[idx].strip()
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                idx += 1
                n_atoms = int(lines[idx].strip())
            elif line.startswith("ITEM: BOX BOUNDS"):
                idx += 1
                xlo, xhi = map(float, lines[idx].strip().split()[:2])
                idx += 1
                ylo, yhi = map(float, lines[idx].strip().split()[:2])
                idx += 1
                zlo, zhi = map(float, lines[idx].strip().split()[:2])
                box_bounds = ((xlo, xhi), (ylo, yhi), (zlo, zhi))
            elif line.startswith("ITEM: ATOMS"):
                idx += 1
                break
            idx += 1

        atom_types = np.zeros(n_atoms, dtype=int)
        positions = np.zeros((n_atoms, 3))
        potentials = np.zeros(n_atoms)
        n_types = 0

        for i in range(n_atoms):
            parts = lines[idx + i].strip().split()
            atom_id = int(parts[0]) - 1  # Convert to 0-indexed
            atom_types[atom_id] = int(parts[1])
            positions[atom_id, 0] = float(parts[2])
            positions[atom_id, 1] = float(parts[3])
            positions[atom_id, 2] = float(parts[4])
            potentials[atom_id] = float(parts[5])
            n_types = max(n_types, atom_types[atom_id])

        return AtomData(
            n_atoms=n_atoms,
            atom_types=atom_types,
            positions=positions,
            potentials=potentials,
            box_bounds=box_bounds,
            n_types=n_types
        )

    def _read_lammps_data_format(self, filename: str) -> AtomData:
        """
        Read LAMMPS data file format (final.data style).

        This format has:
        - Header with atom count and box bounds
        - Atoms section with: id type x y z pe

        Args:
            filename: Path to LAMMPS data file

        Returns:
            AtomData containing parsed atomic information
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        n_atoms = 0
        box_bounds = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if "atoms" in line.lower() and not line.startswith("#"):
                n_atoms = int(line.split()[0])
            elif "xlo xhi" in line:
                parts = line.split()
                box_bounds[0] = [float(parts[0]), float(parts[1])]
            elif "ylo yhi" in line:
                parts = line.split()
                box_bounds[1] = [float(parts[0]), float(parts[1])]
            elif "zlo zhi" in line:
                parts = line.split()
                box_bounds[2] = [float(parts[0]), float(parts[1])]
            elif line.startswith("Atoms"):
                idx += 2  # Skip "Atoms" line and blank line
                break
            idx += 1

        atom_types = np.zeros(n_atoms, dtype=int)
        positions = np.zeros((n_atoms, 3))
        potentials = np.zeros(n_atoms)
        n_types = 0

        for i in range(n_atoms):
            if idx + i >= len(lines):
                break
            parts = lines[idx + i].strip().split()
            if len(parts) < 6:
                continue
            atom_id = int(parts[0]) - 1  # Convert to 0-indexed
            if atom_id >= n_atoms:
                continue
            atom_types[atom_id] = int(parts[1])
            positions[atom_id, 0] = float(parts[2])
            positions[atom_id, 1] = float(parts[3])
            positions[atom_id, 2] = float(parts[4])
            potentials[atom_id] = float(parts[5])
            n_types = max(n_types, atom_types[atom_id])

        return AtomData(
            n_atoms=n_atoms,
            atom_types=atom_types,
            positions=positions,
            potentials=potentials,
            box_bounds=tuple((b[0], b[1]) for b in box_bounds),
            n_types=n_types
        )

    def read_data(self, filename: str) -> AtomData:
        """
        Read LAMMPS data file, auto-detecting format.

        Args:
            filename: Path to LAMMPS data/dump file

        Returns:
            AtomData containing parsed atomic information
        """
        with open(filename, 'r') as f:
            first_line = f.readline()

        if first_line.startswith("ITEM:"):
            return self._read_lammps_dump(filename)
        else:
            return self._read_lammps_data_format(filename)

    def calculate_cohesive_energy(
        self,
        filename: str | Path
    ) -> CohesiveEnergyResult:
        """
        Calculate cohesive energy per atom for each element type.

        Cohesive energy = Total potential energy / Number of atoms

        Args:
            filename: Path to LAMMPS output file

        Returns:
            CohesiveEnergyResult with energies per atom type
        """
        filename = Path(filename)
        data = self.read_data(str(filename))

        # Sum potential energy by atom type
        pot_total: dict[int, float] = {}
        n_count: dict[int, int] = {}

        for i in range(data.n_atoms):
            atype = data.atom_types[i]
            if atype not in pot_total:
                pot_total[atype] = 0.0
                n_count[atype] = 0
            pot_total[atype] += data.potentials[i]
            n_count[atype] += 1

        # Calculate cohesive energy per atom
        energies = {
            atype: pot_total[atype] / n_count[atype]
            for atype in pot_total
        }

        return CohesiveEnergyResult(
            energies=energies,
            n_atoms=n_count,
            source_file=filename
        )

    def calculate_defect_energy(
        self,
        filename: str | Path,
        reference: CohesiveEnergyResult
    ) -> DefectResult:
        """
        Calculate defect formation energy.

        Formation energy = Sum of excess energies in defect region

        Args:
            filename: Path to LAMMPS output file
            reference: Reference cohesive energies for each atom type

        Returns:
            DefectResult with position and formation energy
        """
        filename = Path(filename)
        data = self.read_data(str(filename))

        # Calculate excess potential for each atom
        excess_pot = np.zeros(data.n_atoms)
        for i in range(data.n_atoms):
            atype = data.atom_types[i]
            excess_pot[i] = data.potentials[i] - reference.energies.get(atype, 0.0)

        # Find defect position by scanning for highest excess energy region
        xmin, xmax = data.box_bounds[0]
        ymin, ymax = data.box_bounds[1]
        zmin, zmax = data.box_bounds[2]

        # Scan x direction
        dx = (xmax - xmin) / 1000.0
        max_excess = -1e10
        x_mark = 0.0

        for i in range(100, 900):
            x_lo = xmin + dx * (i - 1)
            x_hi = xmin + dx * i

            mask = (
                (data.positions[:, 0] >= x_lo) &
                (data.positions[:, 0] < x_hi) &
                (data.positions[:, 1] >= 0.9 * ymin + 0.1 * ymax) &
                (data.positions[:, 1] <= 0.1 * ymin + 0.9 * ymax)
            )
            total = np.sum(excess_pot[mask])

            if total > max_excess:
                max_excess = total
                x_mark = (x_lo + x_hi) / 2.0

        xr_def = x_mark

        # Scan y direction
        dy = (ymax - ymin) / 1000.0
        max_excess = -1e10
        y_mark = 0.0

        for i in range(100, 900):
            y_lo = ymin + dy * (i - 1)
            y_hi = ymin + dy * i

            mask = (
                (data.positions[:, 1] >= y_lo) &
                (data.positions[:, 1] < y_hi) &
                (data.positions[:, 0] >= 0.9 * xmin + 0.1 * xmax) &
                (data.positions[:, 0] <= 0.1 * xmin + 0.9 * xmax)
            )
            total = np.sum(excess_pot[mask])

            if total > max_excess:
                max_excess = total
                y_mark = (y_lo + y_hi) / 2.0

        yr_def = y_mark

        # Scan z direction
        dz = (zmax - zmin) / 1000.0
        max_excess = -1e10
        z_mark = 0.0

        for i in range(1, 1000):
            z_lo = zmin + dz * (i - 1)
            z_hi = zmin + dz * i

            mask = (
                (data.positions[:, 2] >= z_lo) &
                (data.positions[:, 2] < z_hi) &
                (data.positions[:, 0] >= 0.9 * xmin + 0.1 * xmax) &
                (data.positions[:, 0] <= 0.1 * xmin + 0.9 * xmax) &
                (data.positions[:, 1] >= 0.9 * ymin + 0.1 * ymax) &
                (data.positions[:, 1] <= 0.1 * ymin + 0.9 * ymax)
            )
            total = np.sum(excess_pot[mask])

            if total > max_excess:
                max_excess = total
                z_mark = (z_lo + z_hi) / 2.0

        zr_def = z_mark

        # Calculate formation energy in defect region
        x_bound = ((xr_def + xmin) / 2.0, (xr_def + xmax) / 2.0)
        y_bound = ((yr_def + ymin) / 2.0, (yr_def + ymax) / 2.0)

        mask = (
            (data.positions[:, 0] >= x_bound[0]) &
            (data.positions[:, 0] <= x_bound[1]) &
            (data.positions[:, 1] >= y_bound[0]) &
            (data.positions[:, 1] <= y_bound[1])
        )

        formation_energy = float(np.sum(excess_pot[mask]))

        return DefectResult(
            position=(xr_def, yr_def, zr_def),
            formation_energy=formation_energy,
            source_file=filename
        )

    def calculate_interface_energy(
        self,
        filename: str | Path,
        reference: CohesiveEnergyResult
    ) -> InterfaceResult:
        """
        Calculate interface energy for horizontal interface.

        Interface energy = Total excess energy / Interface area

        Args:
            filename: Path to LAMMPS output file
            reference: Reference cohesive energies for each atom type

        Returns:
            InterfaceResult with position and interface energy (mJ/m^2)
        """
        filename = Path(filename)
        data = self.read_data(str(filename))

        # Calculate excess potential for each atom
        excess_pot = np.zeros(data.n_atoms)
        for i in range(data.n_atoms):
            atype = data.atom_types[i]
            excess_pot[i] = data.potentials[i] - reference.energies.get(atype, 0.0)

        ymin, ymax = data.box_bounds[1]

        # Find interface by scanning y direction
        dy = (ymax - ymin) / 1000.0
        max_excess = -1e10
        second_max = -1e10
        y_mark1 = 0.0
        y_mark2 = 0.0

        for i in range(100, 900):
            y_lo = ymin + dy * (i - 1)
            y_hi = ymin + dy * i

            mask = (data.positions[:, 1] >= y_lo) & (data.positions[:, 1] < y_hi)
            total = float(np.sum(excess_pot[mask]))

            if total > max_excess:
                second_max = max_excess
                max_excess = total
                y_mark2 = y_mark1
                y_mark1 = (y_lo + y_hi) / 2.0
            elif total > second_max:
                second_max = total
                y_mark2 = (y_lo + y_hi) / 2.0

        yr_int = (y_mark1 + y_mark2) / 2.0

        # Calculate interface energy
        y_bound = ((yr_int + ymin) / 2.0, (yr_int + ymax) / 2.0)

        mask = (
            (data.positions[:, 1] >= y_bound[0]) &
            (data.positions[:, 1] <= y_bound[1])
        )

        pot_total = float(np.sum(excess_pot[mask]))

        # Convert to mJ/m^2 (factor 16021.8 = eV/Å^2 to mJ/m^2)
        xmin, xmax = data.box_bounds[0]
        zmin, zmax = data.box_bounds[2]
        area = (xmax - xmin) * (zmax - zmin)

        interface_energy = pot_total / area * 16021.8

        return InterfaceResult(
            position_y=yr_int,
            interface_energy=interface_energy,
            source_file=filename
        )

    def calculate_interface_energy_spatial(
        self,
        filename: str | Path,
        references: dict[str, float],
        metal_top: bool = False,
    ) -> InterfaceResult:
        """
        Calculate interface energy with spatial phase detection.
        
        This method is for multi-phase interfaces (e.g., Ti/TiB2) where
        atoms of the same type have different reference energies depending
        on which phase they belong to (determined by Y position).
        
        Args:
            filename: Path to LAMMPS output file
            references: Reference energies with keys like:
                - "B_tib2": B atoms in TiB2 phase
                - "Ti_metal": Ti atoms in metal phase
                - "Ti_tib2": Ti atoms in TiB2 phase
            metal_top: If True, metal is at high Y. If False, metal is at low Y.
            
        Returns:
            InterfaceResult with position and interface energy (mJ/m^2)
        """
        filename = Path(filename)
        data = self.read_data(str(filename))
        
        # 1. First pass: Find interface location using potential energy profile
        ymin, ymax = data.box_bounds[1]
        dy = (ymax - ymin) / 1000.0
        
        max_pe = -1e10
        yr_int = (ymin + ymax) / 2.0  # default to center
        
        for i in range(100, 900):
            y_lo = ymin + dy * (i - 1)
            y_hi = ymin + dy * i
            
            mask = (data.positions[:, 1] >= y_lo) & (data.positions[:, 1] < y_hi)
            if np.sum(mask) == 0:
                continue
            
            total_pe = float(np.sum(data.potentials[mask]))
            if total_pe > max_pe:
                max_pe = total_pe
                yr_int = (y_lo + y_hi) / 2.0
        
        # 2. Assign reference energies based on Y position
        ref_b = references.get("B_tib2", 0.0)
        ref_ti_metal = references.get("Ti_metal", 0.0)
        ref_ti_tib2 = references.get("Ti_tib2", 0.0)
        
        excess_pot = np.zeros(data.n_atoms)
        
        for i in range(data.n_atoms):
            atype = data.atom_types[i]
            y = data.positions[i, 1]
            
            if atype == 1:  # B (typically type 1 in Ti/TiB2)
                ref = ref_b
            elif atype == 2:  # Ti
                # Check phase based on Y position
                if not metal_top:
                    # Metal at bottom (low Y)
                    ref = ref_ti_metal if y < yr_int else ref_ti_tib2
                else:
                    # Metal at top (high Y)
                    ref = ref_ti_metal if y > yr_int else ref_ti_tib2
            else:
                ref = 0.0
            
            excess_pot[i] = data.potentials[i] - ref
        
        # 3. Calculate interface energy
        pot_total = float(np.sum(excess_pot))
        
        xmin, xmax = data.box_bounds[0]
        zmin, zmax = data.box_bounds[2]
        area = (xmax - xmin) * (zmax - zmin)
        
        # Convert to mJ/m^2
        interface_energy = pot_total / area * 16021.8
        
        return InterfaceResult(
            position_y=yr_int,
            interface_energy=interface_energy,
            source_file=filename
        )

    def calculate_core_energy(
        self,
        filename: str | Path,
        reference: CohesiveEnergyResult
    ) -> DislocationResult:
        """
        Calculate dislocation core energy and width.

        Uses E-ln(r) method to identify core region where energy
        deviates from linear elastic behavior.

        Args:
            filename: Path to LAMMPS output file
            reference: Reference cohesive energies for each atom type

        Returns:
            DislocationResult with position, core width, and core energy
        """
        filename = Path(filename)
        data = self.read_data(str(filename))

        # Calculate excess potential for each atom
        excess_pot = np.zeros(data.n_atoms)
        for i in range(data.n_atoms):
            atype = data.atom_types[i]
            excess_pot[i] = data.potentials[i] - reference.energies.get(atype, 0.0)

        xmin, xmax = data.box_bounds[0]
        ymin, ymax = data.box_bounds[1]
        zmin, zmax = data.box_bounds[2]

        # Find dislocation in x direction
        dx = (xmax - xmin) / 1000.0
        max_excess = -1e10
        second_max = -1e10
        x_mark1 = 0.0
        x_mark2 = 0.0

        for i in range(100, 900):
            x_lo = xmin + dx * (i - 1)
            x_hi = xmin + dx * i

            mask = (
                (data.positions[:, 0] >= x_lo) &
                (data.positions[:, 0] < x_hi) &
                (data.positions[:, 1] >= 0.9 * ymin + 0.1 * ymax) &
                (data.positions[:, 1] <= 0.1 * ymin + 0.9 * ymax)
            )
            total = float(np.sum(excess_pot[mask]))

            if total > max_excess:
                second_max = max_excess
                max_excess = total
                x_mark2 = x_mark1
                x_mark1 = (x_lo + x_hi) / 2.0

        xr_dis = (x_mark1 + x_mark2) / 2.0

        # Find dislocation in y direction
        dy = (ymax - ymin) / 1000.0
        max_excess = -1e10
        second_max = -1e10
        y_mark1 = 0.0
        y_mark2 = 0.0

        for i in range(1, 1000):
            y_lo = ymin + dy * (i - 1)
            y_hi = ymin + dy * i

            mask = (
                (data.positions[:, 1] >= y_lo) &
                (data.positions[:, 1] < y_hi) &
                (data.positions[:, 0] >= 0.9 * xmin + 0.1 * xmax) &
                (data.positions[:, 0] <= 0.1 * xmin + 0.9 * xmax)
            )
            total = float(np.sum(excess_pot[mask]))

            if total > max_excess:
                second_max = max_excess
                max_excess = total
                y_mark2 = y_mark1
                y_mark1 = (y_lo + y_hi) / 2.0

        yr_dis = (y_mark1 + y_mark2) / 2.0

        # Calculate E(ln(r)) data for core energy extraction
        dist_to_boundary = min(
            xmax - xr_dis, xr_dis - xmin,
            ymax - yr_dis, yr_dis - ymin
        )
        rr_max = 0.5 * dist_to_boundary

        r_values = []
        pot_line = []

        for i in range(101):
            r_tmp = 1.0 + rr_max / 100.0 * i
            r_values.append(math.log(r_tmp))

            # Sum potential within radius r_tmp, normalized by z-length
            pot_sum = 0.0
            for j in range(data.n_atoms):
                dist = math.sqrt(
                    (data.positions[j, 0] - xr_dis) ** 2 +
                    (data.positions[j, 1] - yr_dis) ** 2
                )
                if dist <= r_tmp:
                    pot_sum += excess_pot[j] / (zmax - zmin)
            pot_line.append(pot_sum)

        # Find core radius by slope deviation (> 25% change)
        r_values_arr = np.array(r_values)
        pot_line_arr = np.array(pot_line)

        # Reference slope at outer region
        tk_ref = (pot_line_arr[100] - pot_line_arr[98]) / (r_values_arr[100] - r_values_arr[98])

        core_width = rr_max
        core_energy = pot_line_arr[-1]

        for i in range(99, 0, -1):
            tk_tmp = (pot_line_arr[i + 1] - pot_line_arr[i - 1]) / (r_values_arr[i + 1] - r_values_arr[i - 1])
            deviation = abs((tk_tmp - tk_ref) / tk_ref) if tk_ref != 0 else 0

            if deviation <= 0.25:
                tk_ref = tk_tmp
            else:
                core_width = math.exp(r_values_arr[i])
                core_energy = pot_line_arr[i]
                break

        return DislocationResult(
            position=(xr_dis, yr_dis),
            core_width=core_width,
            core_energy=float(core_energy),
            source_file=filename
        )

    def calculate_neb_barrier(
        self,
        filename: str | Path,
        n_states: int
    ) -> NEBResult:
        """
        Calculate migration barrier from NEB calculation log.

        Barrier = Max energy - Min energy along MEP

        Args:
            filename: Path to LAMMPS log file from NEB run
            n_states: Number of NEB images

        Returns:
            NEBResult with migration barrier in eV
        """
        filename = Path(filename)

        with open(filename, 'r') as f:
            lines = f.readlines()

        # Read last line for final NEB energies
        last_line = lines[-1].strip()
        parts = last_line.split()

        # NEB energies are in alternating positions after the standard thermo columns
        # Format: step temp etotal ... E1 Rx1 E2 Rx2 ... En Rxn
        # Standard thermo has 9 columns before NEB data
        neb_start = 9
        energies = []

        for i in range(n_states):
            idx = neb_start + 2 * i
            if idx < len(parts):
                energies.append(float(parts[idx]))

        if energies:
            barrier = max(energies) - min(energies)
        else:
            barrier = 0.0

        return NEBResult(
            barrier=barrier,
            n_images=n_states,
            source_file=filename
        )
