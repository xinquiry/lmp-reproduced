"""
Crystal structure generation for LAMMPS simulations.

This module provides functions to generate crystal structures for various
material systems and write them in LAMMPS data file format.
"""

from pathlib import Path

import numpy as np

from lmp_reproduced.models import (
    Atom,
    MaterialConfig,
    StructureData,
    StructureType,
)


# Mass lookup table (atomic mass units)
ATOMIC_MASSES = {
    "Al": 26.9815,
    "Cu": 63.546,
    "Mg": 24.305,
    "Ti": 47.867,
    "B": 10.811,
    "Si": 28.0855,
    "C": 12.0107,
}


def create_fcc_structure(
    config: MaterialConfig,
    n_cells: int = 4,
) -> StructureData:
    """
    Create FCC (Face-Centered Cubic) crystal structure.

    Args:
        config: Material configuration with lattice parameters
        n_cells: Number of unit cells in each direction

    Returns:
        StructureData containing the generated structure
    """
    a = config.lattice_a
    box_size = n_cells * a

    # FCC basis positions (fractional)
    basis = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ]

    atoms = []
    atom_id = 1

    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for bx, by, bz in basis:
                    x = (i + bx) * a
                    y = (j + by) * a
                    z = (k + bz) * a
                    atoms.append(Atom(id=atom_id, type=1, x=x, y=y, z=z))
                    atom_id += 1

    mass = ATOMIC_MASSES.get(config.element, 26.98)

    return StructureData(
        title=f"LAMMPS {config.element} FCC crystal",
        atoms=atoms,
        n_types=1,
        box_bounds=((0.0, box_size), (0.0, box_size), (0.0, box_size)),
        masses={1: mass}
    )


def create_hcp_structure(
    config: MaterialConfig,
    n_cells: int = 4,
    n_types: int = 1,
    atom_type: int = 1,
) -> StructureData:
    """
    Create HCP (Hexagonal Close-Packed) crystal structure.

    Uses the orthogonal representation with 4 atoms per unit cell,
    compatible with LAMMPS's lattice hcp command.

    Args:
        config: Material configuration with lattice parameters (a and c)
        n_cells: Number of unit cells in each direction
        n_types: Total number of atom types to declare in header
        atom_type: The atom type ID to assign to generated atoms

    Returns:
        StructureData containing the generated structure
    """
    a = config.lattice_a
    c = config.lattice_c if config.lattice_c else a * 1.633  # Ideal c/a ratio

    # Orthogonal HCP unit cell dimensions
    # Matches LAMMPS lattice hcp command
    lx = a  # x lattice spacing
    ly = a * np.sqrt(3)  # y lattice spacing
    lz = c  # z lattice spacing

    # 4 atoms per orthogonal unit cell (fractional coordinates)
    basis = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 1/6, 0.5),
        (0.0, 2/3, 0.5),
    ]

    atoms = []
    atom_id = 1

    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for bx, by, bz in basis:
                    x = (i + bx) * lx
                    y = (j + by) * ly
                    z = (k + bz) * lz
                    atoms.append(Atom(id=atom_id, type=atom_type, x=x, y=y, z=z))
                    atom_id += 1

    xlo, xhi = 0.0, n_cells * lx
    ylo, yhi = 0.0, n_cells * ly
    zlo, zhi = 0.0, n_cells * lz

    # Set up masses
    masses = {}
    if config.element == "Ti" and n_types == 2:
        # Ti-B system: type 1 = B, type 2 = Ti
        masses = {1: ATOMIC_MASSES["B"], 2: ATOMIC_MASSES["Ti"]}
    else:
        mass = ATOMIC_MASSES.get(config.element, 24.305)
        for t in range(1, n_types + 1):
            masses[t] = mass

    return StructureData(
        title=f"LAMMPS {config.element} HCP crystal",
        atoms=atoms,
        n_types=n_types,
        box_bounds=((xlo, xhi), (ylo, yhi), (zlo, zhi)),
        masses=masses
    )


def create_hexagonal_structure(
    config: MaterialConfig,
    n_cells: int = 4,
) -> StructureData:
    """
    Create hexagonal crystal structure (AlB2-type, for TiB2).

    Structure: P6/mmm space group
    - Ti at (0, 0, 0)
    - B at (1/3, 2/3, 1/2) and (2/3, 1/3, 1/2)

    Args:
        config: Material configuration with lattice parameters (a and c)
        n_cells: Number of unit cells in each direction

    Returns:
        StructureData containing the generated structure
    """
    a = config.lattice_a
    c = config.lattice_c if config.lattice_c else a

    # Hexagonal lattice vectors
    # a1 = a * (1, 0, 0)
    # a2 = a * (-1/2, sqrt(3)/2, 0)
    # a3 = c * (0, 0, 1)

    # Basis positions for AlB2-type structure (fractional coordinates)
    # Type 1: B atoms at (1/3, 2/3, 1/2) and (2/3, 1/3, 1/2)
    # Type 2: Ti atoms at (0, 0, 0)
    b_basis = [
        (1/3, 2/3, 0.5),
        (2/3, 1/3, 0.5),
    ]
    ti_basis = [
        (0.0, 0.0, 0.0),
    ]

    atoms = []
    atom_id = 1

    sqrt3 = np.sqrt(3)

    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                # B atoms (type 1)
                for fx, fy, fz in b_basis:
                    # Convert fractional to Cartesian (hexagonal)
                    x = (i + fx) * a + (j + fy) * a * (-0.5)
                    y = (j + fy) * a * sqrt3 / 2
                    z = (k + fz) * c
                    atoms.append(Atom(id=atom_id, type=1, x=x, y=y, z=z))
                    atom_id += 1

                # Ti atoms (type 2)
                for fx, fy, fz in ti_basis:
                    x = (i + fx) * a + (j + fy) * a * (-0.5)
                    y = (j + fy) * a * sqrt3 / 2
                    z = (k + fz) * c
                    atoms.append(Atom(id=atom_id, type=2, x=x, y=y, z=z))
                    atom_id += 1

    # Box bounds for hexagonal cell
    xlo = -n_cells * a * 0.5
    xhi = n_cells * a
    ylo = 0.0
    yhi = n_cells * a * sqrt3 / 2
    zlo = 0.0
    zhi = n_cells * c

    return StructureData(
        title=f"LAMMPS {config.element} Hexagonal (AlB2-type) crystal",
        atoms=atoms,
        n_types=2,
        box_bounds=((xlo, xhi), (ylo, yhi), (zlo, zhi)),
        masses={1: ATOMIC_MASSES["B"], 2: ATOMIC_MASSES["Ti"]}
    )


def create_zincblende_structure(
    config: MaterialConfig,
    n_cells: int = 4,
) -> StructureData:
    """
    Create Zincblende crystal structure (e.g., for SiC).

    Structure: FCC lattice with basis at (0,0,0) and (0.25, 0.25, 0.25).

    Args:
        config: Material configuration with lattice parameters
        n_cells: Number of unit cells in each direction

    Returns:
        StructureData containing the generated structure
    """
    a = config.lattice_a
    box_size = n_cells * a

    # FCC basis vectors
    fcc_basis = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ]

    atoms = []
    atom_id = 1

    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for bx, by, bz in fcc_basis:
                    # Type 1: Si at FCC sites
                    x1 = (i + bx) * a
                    y1 = (j + by) * a
                    z1 = (k + bz) * a
                    atoms.append(Atom(id=atom_id, type=1, x=x1, y=y1, z=z1))
                    atom_id += 1

                    # Type 2: C at FCC sites + 0.25*(1,1,1)
                    x2 = x1 + 0.25 * a
                    y2 = y1 + 0.25 * a
                    z2 = z1 + 0.25 * a
                    atoms.append(Atom(id=atom_id, type=2, x=x2, y=y2, z=z2))
                    atom_id += 1

    return StructureData(
        title=f"LAMMPS {config.element} Zincblende crystal",
        atoms=atoms,
        n_types=2,
        box_bounds=((0.0, box_size), (0.0, box_size), (0.0, box_size)),
        masses={1: ATOMIC_MASSES["Si"], 2: ATOMIC_MASSES["C"]}
    )


def create_fcc_substitutional(
    host_config: MaterialConfig,
    solute_element: str,
    n_cells: int = 4,
) -> StructureData:
    """
    Create FCC structure with one substitutional defect at box center.

    Args:
        host_config: Host material configuration (e.g., Al)
        solute_element: Element symbol for solute atom (e.g., "Mg")
        n_cells: Number of unit cells in each direction

    Returns:
        StructureData with 2 atom types: type 1 = host, type 2 = solute
    """
    a = host_config.lattice_a
    box_size = n_cells * a

    # FCC basis positions (fractional)
    basis = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ]

    atoms = []
    atom_id = 1

    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for bx, by, bz in basis:
                    x = (i + bx) * a
                    y = (j + by) * a
                    z = (k + bz) * a
                    atoms.append(Atom(id=atom_id, type=1, x=x, y=y, z=z))
                    atom_id += 1

    # Find atom closest to box center and change its type to solute
    center = box_size / 2.0
    min_dist = float('inf')
    center_idx = 0

    for i, atom in enumerate(atoms):
        dist = (atom.x - center)**2 + (atom.y - center)**2 + (atom.z - center)**2
        if dist < min_dist:
            min_dist = dist
            center_idx = i

    # Change center atom to solute type (type 2)
    center_atom = atoms[center_idx]
    atoms[center_idx] = Atom(
        id=center_atom.id,
        type=2,
        x=center_atom.x,
        y=center_atom.y,
        z=center_atom.z
    )

    host_mass = ATOMIC_MASSES.get(host_config.element, 26.98)
    solute_mass = ATOMIC_MASSES.get(solute_element, 24.305)

    return StructureData(
        title=f"LAMMPS {host_config.element} FCC with {solute_element} substitutional",
        atoms=atoms,
        n_types=2,
        box_bounds=((0.0, box_size), (0.0, box_size), (0.0, box_size)),
        masses={1: host_mass, 2: solute_mass}
    )


def create_hcp_vacancy(
    config: MaterialConfig,
    n_cells: int = 4,
    n_types: int = 2,
    atom_type: int = 2,
) -> StructureData:
    """
    Create HCP structure with one vacancy at box center.

    Uses the orthogonal representation with 4 atoms per unit cell.

    Args:
        config: Material configuration with lattice parameters (a and c)
        n_cells: Number of unit cells in each direction
        n_types: Total number of atom types to declare in header
        atom_type: The atom type ID to assign to generated atoms

    Returns:
        StructureData with vacancy (one atom removed from center)
    """
    a = config.lattice_a
    c = config.lattice_c if config.lattice_c else a * 1.633

    # Orthogonal HCP unit cell dimensions
    lx = a
    ly = a * np.sqrt(3)
    lz = c

    # 4 atoms per orthogonal unit cell (fractional coordinates)
    basis = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 1/6, 0.5),
        (0.0, 2/3, 0.5),
    ]

    atoms = []
    atom_id = 1

    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for bx, by, bz in basis:
                    x = (i + bx) * lx
                    y = (j + by) * ly
                    z = (k + bz) * lz
                    atoms.append(Atom(id=atom_id, type=atom_type, x=x, y=y, z=z))
                    atom_id += 1

    xhi = n_cells * lx
    yhi = n_cells * ly
    zhi = n_cells * lz

    # Find atom closest to box center and remove it (vacancy)
    center_x, center_y, center_z = xhi / 2.0, yhi / 2.0, zhi / 2.0
    min_dist = float('inf')
    center_idx = 0

    for i, atom in enumerate(atoms):
        dist = (atom.x - center_x)**2 + (atom.y - center_y)**2 + (atom.z - center_z)**2
        if dist < min_dist:
            min_dist = dist
            center_idx = i

    # Remove the center atom
    del atoms[center_idx]

    # Re-index remaining atoms
    for i, atom in enumerate(atoms):
        atoms[i] = Atom(id=i + 1, type=atom.type, x=atom.x, y=atom.y, z=atom.z)

    # Set up masses (same as create_hcp_structure for Ti)
    masses = {}
    if config.element == "Ti" and n_types == 2:
        masses = {1: ATOMIC_MASSES["B"], 2: ATOMIC_MASSES["Ti"]}
    else:
        mass = ATOMIC_MASSES.get(config.element, 24.305)
        for t in range(1, n_types + 1):
            masses[t] = mass

    return StructureData(
        title=f"LAMMPS {config.element} HCP with vacancy",
        atoms=atoms,
        n_types=n_types,
        box_bounds=((0.0, xhi), (0.0, yhi), (0.0, zhi)),
        masses=masses
    )


def create_structure(
    config: MaterialConfig,
    n_cells: int = 4,
) -> StructureData:
    """
    Create crystal structure based on the material configuration.

    This is a convenience function that dispatches to the appropriate
    structure generation function based on the structure type.

    Args:
        config: Material configuration
        n_cells: Number of unit cells in each direction

    Returns:
        StructureData containing the generated structure

    Raises:
        ValueError: If structure type is not supported
    """
    if config.structure_type == StructureType.FCC:
        return create_fcc_structure(config, n_cells)
    elif config.structure_type == StructureType.HCP:
        # Special handling for Ti (needs 2 types for MEAM potential)
        if config.element == "Ti":
            return create_hcp_structure(config, n_cells, n_types=2, atom_type=2)
        return create_hcp_structure(config, n_cells)
    elif config.structure_type == StructureType.ZINCBLENDE:
        return create_zincblende_structure(config, n_cells)
    elif config.structure_type == StructureType.HEXAGONAL:
        return create_hexagonal_structure(config, n_cells)
    else:
        raise ValueError(f"Unsupported structure type: {config.structure_type}")


def write_lammps_data(
    structure: StructureData,
    path: Path,
) -> Path:
    """
    Write structure data to LAMMPS data file format.

    Args:
        structure: StructureData to write
        path: Output file path

    Returns:
        Path to the written file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [structure.title, ""]
    lines.append(f"{len(structure.atoms)} atoms")
    lines.append(f"{structure.n_types} atom types")
    lines.append("")

    # Box bounds
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = structure.box_bounds
    lines.append(f"{xlo:.6f} {xhi:.6f} xlo xhi")
    lines.append(f"{ylo:.6f} {yhi:.6f} ylo yhi")
    lines.append(f"{zlo:.6f} {zhi:.6f} zlo zhi")
    lines.append("")

    # Masses
    lines.append("Masses")
    lines.append("")
    for atom_type, mass in sorted(structure.masses.items()):
        lines.append(f"{atom_type} {mass}")
    lines.append("")

    # Atoms
    lines.append("Atoms")
    lines.append("")
    for atom in structure.atoms:
        lines.append(f"{atom.id} {atom.type} {atom.x:.6f} {atom.y:.6f} {atom.z:.6f}")

    content = "\n".join(lines)
    path.write_text(content)

    return path
    return path


def create_interface(
    config_bottom: MaterialConfig,
    config_top: MaterialConfig,
    n_layers_bottom: int = 6,
    n_layers_top: int = 6,
    vacuum: float = 20.0,
    mismatch_tol: float = 0.05,
) -> StructureData:
    """
    Create an interface structure by stacking two material slabs.
    
    This function dispatches to specific interface builders based on the
    material combination. It handles:
    1. Crystal orientation alignment
    2. Lattice mismatch minimization (finding supercell sizes)
    3. Stacking with vacuum padding
    
    Args:
        config_bottom: Material configuration for the bottom slab
        config_top: Material configuration for the top slab
        n_layers_bottom: Number of atomic layers for bottom slab
        n_layers_top: Number of atomic layers for top slab
        vacuum: Vacuum spacing in Angstroms (total vacuum in Z)
        mismatch_tol: Maximum allowed lattice mismatch strain
        
    Returns:
        StructureData containing the interface model
    """
    sys_pair = (config_bottom.element, config_top.element)
    
    if sys_pair == ("Al", "Mg"):
        return create_al_mg_interface(config_bottom, config_top, n_layers_bottom, n_layers_top, vacuum)
    elif sys_pair == ("Ti", "TiB2"):
        return create_ti_tib2_interface(config_bottom, config_top, n_layers_bottom, n_layers_top, vacuum)
    else:
        raise NotImplementedError(f"Interface builder for {sys_pair} not implemented yet.")


def create_al_mg_interface(
    al_config: MaterialConfig,
    mg_config: MaterialConfig,
    n_layers_al: int = 6,
    n_layers_mg: int = 6,
    vacuum: float = 20.0
) -> StructureData:
    """
    Create Al(111)//Mg(0001) semi-coherent interface stacked along Y.
    
    Orientation Relationship:
    - Plane: Al(111) // Mg(0001) -> Normal to Y
    - Direction: Al[1-10] // Mg[1-100] -> X axis
    """
    # Al (111) setup
    a_al = al_config.lattice_a
    # X direction [1-10]
    lx_al = a_al / np.sqrt(2)
    # Z direction [11-2] (ortho to X and Normal)
    lz_al = a_al * np.sqrt(1.5)
    # Y direction [111] (Stacking)
    d_al_111 = a_al / np.sqrt(3) # spacing
    
    nx_al, nz_al = 9, 5 
    n_repeat_al = 9
    n_repeat_mg = 8
    
    # 1. Generate Al atoms
    atoms_al = []
    atom_id = 1
    
    # Basis in X-Z plane (periodic)
    # Rectangular cell basis: (0,0), (0.5, 0.5)
    rect_basis_al = [(0.0, 0.0), (0.5, 0.5)]
    
    # Shifts now along Z (the periodic B direction)
    # ABC stacking shifts relative to Z?
    # FCC (111) stacking ABC involves shifts in the plane.
    # Shift vector (x_shift, z_shift).
    # X shift: 0. 
    # Z shift: 1/3 * Lz?
    shift_z = [0.0, 1.0/3.0, 2.0/3.0]
    
    # Supercell dims
    final_lx_al = n_repeat_al * lx_al
    final_lz_al = n_repeat_al * lz_al 
    
    # Mg (0001) setup
    lx_mg = 3.196
    lz_mg = 3.196 * np.sqrt(3)
    
    avg_lx = (n_repeat_al * lx_al + n_repeat_mg * lx_mg) / 2.0
    avg_lz = (n_repeat_al * lz_al + n_repeat_mg * lz_mg) / 2.0
    
    scale_x_al = avg_lx / (n_repeat_al * lx_al)
    scale_z_al = avg_lz / (n_repeat_al * lz_al)
    
    scale_x_mg = avg_lx / (n_repeat_mg * lx_mg)
    scale_z_mg = avg_lz / (n_repeat_mg * lz_mg)
    
    y_current = 0.0
    
    # Al Slab
    for k in range(n_layers_al):
        layer_idx = k % 3
        z_shift = shift_z[layer_idx]
        
        for i in range(n_repeat_al):
            for j in range(n_repeat_al):
                for bax, baz in rect_basis_al:
                    x = (i + bax) * lx_al * scale_x_al
                    # Z is second periodic dim
                    z = (j + baz + z_shift) * lz_al * scale_z_al
                    z = z % avg_lz
                    y = y_current
                    
                    atoms_al.append(Atom(atom_id, 1, x, y, z))
                    atom_id += 1
        y_current += d_al_111
        
    y_interface = y_current + 2.0
    
    # Mg Slab
    c_mg = 5.197
    d_mg_0001 = c_mg / 2.0
    rect_basis_mg = [(0.0, 0.0), (0.5, 0.5)]
    shift_z_mg = [0.0, 1.0/3.0]
    
    atoms_mg = []
    y_current = y_interface
    
    for k in range(n_layers_mg):
        layer_idx = k % 2
        z_shift = shift_z_mg[layer_idx]
        
        for i in range(n_repeat_mg):
            for j in range(n_repeat_mg):
                for bax, baz in rect_basis_mg:
                    x = (i + bax) * lx_mg * scale_x_mg
                    z = (j + baz + z_shift) * lz_mg * scale_z_mg
                    z = z % avg_lz
                    y = y_current
                    
                    atoms_mg.append(Atom(atom_id, 2, x, y, z))
                    atom_id += 1
        y_current += d_mg_0001
        
    final_atoms = atoms_al + atoms_mg
    total_y = y_current + vacuum
    
    masses = {1: ATOMIC_MASSES.get("Al"), 2: ATOMIC_MASSES.get("Mg")}
    
    return StructureData(
        title="Al(111)/Mg(0001) interface Y-stack",
        atoms=final_atoms,
        n_types=2,
        box_bounds=((0.0, avg_lx), (0.0, total_y), (0.0, avg_lz)),
        masses=masses
    )

def create_ti_tib2_interface(
    ti_config: MaterialConfig,
    tib2_config: MaterialConfig,
    n_layers_ti: int = 6,
    n_layers_tib2: int = 6,
    vacuum: float = 20.0
) -> StructureData:
    """
    Create Ti(0001)//TiB2(0001) interface stacked along Y.
    2 Types only: 1=B, 2=Ti.
    """
    a_ti = 2.945
    c_ti = 4.687
    
    a_tib2 = 3.050
    c_tib2 = 3.197
    
    n_ti = 30
    n_tib2 = 29
    
    # X and Z (periodic)
    lx_ti = a_ti
    lz_ti = a_ti * np.sqrt(3)
    
    lx_tib2 = a_tib2
    lz_tib2 = a_tib2 * np.sqrt(3)
    
    avg_lx = (n_ti * lx_ti + n_tib2 * lx_tib2) / 2.0
    avg_lz = (n_ti * lz_ti + n_tib2 * lz_tib2) / 2.0
    
    scale_x_ti = avg_lx / (n_ti * lx_ti)
    scale_z_ti = avg_lz / (n_ti * lz_ti)
    
    scale_x_tib2 = avg_lx / (n_tib2 * lx_tib2)
    scale_z_tib2 = avg_lz / (n_tib2 * lz_tib2)
    
    atom_id = 1
    final_atoms = []
    
    # 1. Ti Slab (Bottom)
    d_ti = c_ti / 2.0
    basis_ti = [(0.0, 0.0), (0.5, 0.5)]
    shift_z_ti = [0.0, 1.0/3.0]
    
    TYPE_B = 1
    TYPE_TI = 2
    
    y_current = 0.0
    
    for k in range(n_layers_ti):
        layer_idx = k % 2
        z_shift = shift_z_ti[layer_idx]
        
        for i in range(n_ti):
            for j in range(n_ti):
                for bax, baz in basis_ti:
                    x = (i + bax) * lx_ti * scale_x_ti
                    z = (j + baz + z_shift) * lz_ti * scale_z_ti
                    z = z % avg_lz
                    y = y_current
                    
                    final_atoms.append(Atom(atom_id, TYPE_TI, x, y, z))
                    atom_id += 1
        y_current += d_ti
        
    y_interface = y_current + 2.0
    y_current = y_interface
    
    # 2. TiB2 Slab (Top)
    basis_ti_layer = [(0.0, 0.0), (0.5, 0.5)]
    basis_b_layer = [
        (0.5, 1.0/6.0),
        (0.5, 5.0/6.0),
        (0.0, 1.0/3.0),
        (0.0, 2.0/3.0)
    ]
    
    z_layer_spacing = c_tib2 / 2.0 # ??? Wait
    # TiB2 alternating planes: Ti ... B ... Ti ...
    # Spacing is c/2 between Ti and Ti? No.
    # Unit cell height c.
    # Ti at 0. B at c/2. Ti at c.
    # Spacing between planes is c/2.
    
    for k in range(n_layers_tib2):
        # Ti layer
        for i in range(n_tib2):
            for j in range(n_tib2):
                for bax, baz in basis_ti_layer:
                    x = (i + bax) * lx_tib2 * scale_x_tib2
                    z = (j + baz) * lz_tib2 * scale_z_tib2
                    z = z % avg_lz
                    # Ti is Type 2
                    final_atoms.append(Atom(atom_id, TYPE_TI, x, y_current, z))
                    atom_id += 1
        
        # B layer
        y_b = y_current + c_tib2 * 0.5
        
        for i in range(n_tib2):
            for j in range(n_tib2):
                for bax, baz in basis_b_layer:
                    x = (i + bax) * lx_tib2 * scale_x_tib2
                    z = (j + baz) * lz_tib2 * scale_z_tib2
                    z = z % avg_lz
                    final_atoms.append(Atom(atom_id, TYPE_B, x, y_b, z))
                    atom_id += 1
                    
        y_current += c_tib2
        
    total_y = y_current + vacuum
    
    masses = {1: ATOMIC_MASSES["B"], 2: ATOMIC_MASSES["Ti"]}
    
    return StructureData(
        title="Ti(0001)/TiB2(0001) interface Y-stack",
        atoms=final_atoms,
        n_types=2,
        box_bounds=((0.0, avg_lx), (0.0, total_y), (0.0, avg_lz)),
        masses=masses
    )
