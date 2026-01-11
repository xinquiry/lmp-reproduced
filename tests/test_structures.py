"""
Tests for crystal structure generation module.
"""

import tempfile
from pathlib import Path

from lmp_reproduced import (
    MaterialConfig,
    StructureType,
    StructureData,
    create_structure,
    create_fcc_structure,
    create_hcp_structure,
    create_zincblende_structure,
    write_lammps_data,
    ALUMINUM,
    MAGNESIUM,
    SILICON_CARBIDE,
)


def test_create_fcc_structure():
    """Test FCC structure generation."""
    structure = create_fcc_structure(ALUMINUM, n_cells=2)

    # FCC has 4 atoms per unit cell, so 2^3 * 4 = 32 atoms
    assert len(structure.atoms) == 32
    assert structure.n_types == 1

    # Check box size
    expected_size = 2 * ALUMINUM.lattice_a
    assert structure.box_bounds[0] == (0.0, expected_size)
    assert structure.box_bounds[1] == (0.0, expected_size)
    assert structure.box_bounds[2] == (0.0, expected_size)

    # Check masses
    assert 1 in structure.masses
    assert abs(structure.masses[1] - 26.9815) < 0.01

    print("test_create_fcc_structure passed")


def test_create_hcp_structure():
    """Test HCP structure generation."""
    structure = create_hcp_structure(MAGNESIUM, n_cells=2)

    # Orthogonal HCP has 4 atoms per unit cell, so 2^3 * 4 = 32 atoms
    assert len(structure.atoms) == 32
    assert structure.n_types == 1

    # Check masses
    assert 1 in structure.masses
    assert abs(structure.masses[1] - 24.305) < 0.01

    print("test_create_hcp_structure passed")


def test_create_zincblende_structure():
    """Test Zincblende structure generation."""
    structure = create_zincblende_structure(SILICON_CARBIDE, n_cells=2)

    # Zincblende has 8 atoms per unit cell (4 of each type), so 2^3 * 8 = 64
    assert len(structure.atoms) == 64
    assert structure.n_types == 2

    # Count atoms of each type
    type_counts = {}
    for atom in structure.atoms:
        type_counts[atom.type] = type_counts.get(atom.type, 0) + 1

    assert type_counts[1] == 32  # Si
    assert type_counts[2] == 32  # C

    # Check masses
    assert 1 in structure.masses  # Si
    assert 2 in structure.masses  # C
    assert abs(structure.masses[1] - 28.0855) < 0.01
    assert abs(structure.masses[2] - 12.0107) < 0.01

    print("test_create_zincblende_structure passed")


def test_create_structure_dispatch():
    """Test automatic structure type dispatch."""
    # FCC
    al_struct = create_structure(ALUMINUM, n_cells=2)
    assert len(al_struct.atoms) == 32

    # HCP (orthogonal representation with 4 atoms per cell)
    mg_struct = create_structure(MAGNESIUM, n_cells=2)
    assert len(mg_struct.atoms) == 32

    # Zincblende
    sic_struct = create_structure(SILICON_CARBIDE, n_cells=2)
    assert len(sic_struct.atoms) == 64

    print("test_create_structure_dispatch passed")


def test_write_lammps_data():
    """Test writing structure to LAMMPS data file."""
    structure = create_fcc_structure(ALUMINUM, n_cells=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "data.struct"
        result_path = write_lammps_data(structure, output_path)

        assert result_path.exists()

        # Read and verify content
        content = result_path.read_text()

        assert "32 atoms" in content
        assert "1 atom types" in content
        assert "xlo xhi" in content
        assert "Masses" in content
        assert "26.9815" in content
        assert "Atoms" in content

    print("test_write_lammps_data passed")


def test_custom_material_config():
    """Test structure generation with custom MaterialConfig."""
    custom_material = MaterialConfig(
        element="Cu",
        lattice_a=3.6,
        structure_type=StructureType.FCC,
    )

    structure = create_structure(custom_material, n_cells=3)

    # 3^3 * 4 = 108 atoms
    assert len(structure.atoms) == 108

    # Check box size
    expected_size = 3 * 3.6
    assert abs(structure.box_bounds[0][1] - expected_size) < 0.001

    print("test_custom_material_config passed")


def test_structure_data_integrity():
    """Test that StructureData contains valid data."""
    structure = create_fcc_structure(ALUMINUM, n_cells=2)

    # All atoms should have valid IDs
    ids = [atom.id for atom in structure.atoms]
    assert len(set(ids)) == len(ids)  # All unique
    assert min(ids) == 1
    assert max(ids) == len(structure.atoms)

    # All atoms should be inside box
    xmin, xmax = structure.box_bounds[0]
    ymin, ymax = structure.box_bounds[1]
    zmin, zmax = structure.box_bounds[2]

    for atom in structure.atoms:
        assert xmin <= atom.x <= xmax
        assert ymin <= atom.y <= ymax
        assert zmin <= atom.z <= zmax

    print("test_structure_data_integrity passed")


def run_all_tests():
    """Run all structure tests."""
    print("\n=== Testing Structure Generation ===\n")

    test_create_fcc_structure()
    test_create_hcp_structure()
    test_create_zincblende_structure()
    test_create_structure_dispatch()
    test_write_lammps_data()
    test_custom_material_config()
    test_structure_data_integrity()

    print("\n=== All structure tests passed! ===\n")


if __name__ == "__main__":
    run_all_tests()
