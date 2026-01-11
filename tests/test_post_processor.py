"""
Tests for LAMMPS Post-Processor module.
"""

import tempfile
from pathlib import Path

from lmp_reproduced import (
    LAMMPSPostProcessor,
    CohesiveEnergyResult,
    DefectResult,
)


def create_sample_dump_file(n_atoms: int = 100) -> str:
    """Create a sample LAMMPS dump file for testing."""
    content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
100
ITEM: BOX BOUNDS pp pp pp
0.0 20.0
0.0 20.0
0.0 20.0
ITEM: ATOMS id type x y z c_pot
"""

    # Create a simple FCC-like structure with uniform potential
    lines = []
    atom_id = 1
    for i in range(5):
        for j in range(5):
            for k in range(4):
                x = i * 4.0 + 1.0
                y = j * 4.0 + 1.0
                z = k * 5.0 + 1.0
                pot = -3.36  # Al cohesive energy
                lines.append(f"{atom_id} 1 {x:.2f} {y:.2f} {z:.2f} {pot:.4f}")
                atom_id += 1

    return content + "\n".join(lines[:100])


def create_sample_data_file() -> str:
    """Create a sample LAMMPS data file for testing."""
    content = """LAMMPS data file

100 atoms
1 atom types

0.0 20.0 xlo xhi
0.0 20.0 ylo yhi
0.0 20.0 zlo zhi

Atoms

"""

    lines = []
    atom_id = 1
    for i in range(5):
        for j in range(5):
            for k in range(4):
                x = i * 4.0 + 1.0
                y = j * 4.0 + 1.0
                z = k * 5.0 + 1.0
                pot = -3.36
                lines.append(f"{atom_id} 1 {x:.2f} {y:.2f} {z:.2f} {pot:.4f}")
                atom_id += 1

    return content + "\n".join(lines[:100])


def test_read_dump_format():
    """Test reading LAMMPS dump file format."""
    processor = LAMMPSPostProcessor()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as f:
        f.write(create_sample_dump_file())
        temp_path = f.name

    try:
        data = processor.read_data(temp_path)
        assert data.n_atoms == 100
        assert data.n_types == 1
        assert data.box_bounds[0] == (0.0, 20.0)
        print("test_read_dump_format passed")
    finally:
        Path(temp_path).unlink()


def test_cohesive_energy():
    """Test cohesive energy calculation returns CohesiveEnergyResult."""
    processor = LAMMPSPostProcessor()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as f:
        f.write(create_sample_dump_file())
        temp_path = f.name

    try:
        result = processor.calculate_cohesive_energy(temp_path)

        # Check return type is dataclass
        assert isinstance(result, CohesiveEnergyResult)

        # Check energies dict
        assert 1 in result.energies
        assert abs(result.energies[1] - (-3.36)) < 0.01

        # Check n_atoms dict
        assert 1 in result.n_atoms
        assert result.n_atoms[1] == 100

        # Check convenience properties
        assert abs(result.primary_energy - (-3.36)) < 0.01
        assert result.total_atoms == 100

        # Check source_file is set
        assert result.source_file == Path(temp_path)

        print(f"test_cohesive_energy passed: {result.primary_energy:.4f} eV")
    finally:
        Path(temp_path).unlink()


def test_defect_with_vacancy():
    """Test defect energy calculation returns DefectResult."""
    processor = LAMMPSPostProcessor()

    # Create structure with one high-energy region (simulating vacancy)
    content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
100
ITEM: BOX BOUNDS pp pp pp
0.0 20.0
0.0 20.0
0.0 20.0
ITEM: ATOMS id type x y z c_pot
"""

    lines = []
    atom_id = 1
    for i in range(5):
        for j in range(5):
            for k in range(4):
                x = i * 4.0 + 1.0
                y = j * 4.0 + 1.0
                z = k * 5.0 + 1.0
                # Create high energy region around (10, 10, 10) to simulate defect
                dist = ((x - 10)**2 + (y - 10)**2 + (z - 10)**2)**0.5
                if dist < 3.0:
                    pot = -3.0  # Higher energy (less negative)
                else:
                    pot = -3.36
                lines.append(f"{atom_id} 1 {x:.2f} {y:.2f} {z:.2f} {pot:.4f}")
                atom_id += 1

    full_content = content + "\n".join(lines[:100])

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as f:
        f.write(full_content)
        temp_path = f.name

    try:
        # Create reference energy result
        reference = CohesiveEnergyResult(
            energies={1: -3.36},
            n_atoms={1: 100},
            source_file=Path(temp_path)
        )

        result = processor.calculate_defect_energy(temp_path, reference)

        # Check return type is dataclass
        assert isinstance(result, DefectResult)

        # Defect should be found near center
        assert 5.0 < result.position[0] < 15.0
        assert 5.0 < result.position[1] < 15.0
        assert result.formation_energy > 0  # Defect has positive formation energy

        # Check source_file is set
        assert result.source_file == Path(temp_path)

        print(f"test_defect_with_vacancy passed: position=({result.position[0]:.1f}, "
              f"{result.position[1]:.1f}, {result.position[2]:.1f}), "
              f"Ef={result.formation_energy:.3f} eV")
    finally:
        Path(temp_path).unlink()


def test_cohesive_energy_multi_type():
    """Test cohesive energy calculation with multiple atom types."""
    processor = LAMMPSPostProcessor()

    # Create structure with two atom types
    content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
50
ITEM: BOX BOUNDS pp pp pp
0.0 20.0
0.0 20.0
0.0 20.0
ITEM: ATOMS id type x y z c_pot
"""

    lines = []
    atom_id = 1
    for i in range(5):
        for j in range(5):
            for k in range(2):
                x = i * 4.0 + 1.0
                y = j * 4.0 + 1.0
                z = k * 10.0 + 1.0
                # Type 1 (Si-like) and Type 2 (C-like)
                atom_type = 1 if k == 0 else 2
                pot = -4.63 if atom_type == 1 else -7.37  # Approximate SiC energies
                lines.append(f"{atom_id} {atom_type} {x:.2f} {y:.2f} {z:.2f} {pot:.4f}")
                atom_id += 1

    full_content = content + "\n".join(lines[:50])

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as f:
        f.write(full_content)
        temp_path = f.name

    try:
        result = processor.calculate_cohesive_energy(temp_path)

        # Should have two types
        assert len(result.energies) == 2
        assert 1 in result.energies
        assert 2 in result.energies

        # Check energies are roughly correct
        assert abs(result.energies[1] - (-4.63)) < 0.01
        assert abs(result.energies[2] - (-7.37)) < 0.01

        # Check atom counts
        assert result.n_atoms[1] == 25
        assert result.n_atoms[2] == 25
        assert result.total_atoms == 50

        print(f"test_cohesive_energy_multi_type passed: "
              f"Type1={result.energies[1]:.2f}, Type2={result.energies[2]:.2f}")
    finally:
        Path(temp_path).unlink()


def run_all_tests():
    """Run all post-processor tests."""
    print("\n=== Testing LAMMPS Post-Processor ===\n")

    test_read_dump_format()
    test_cohesive_energy()
    test_defect_with_vacancy()
    test_cohesive_energy_multi_type()

    print("\n=== All post-processor tests passed! ===\n")


if __name__ == "__main__":
    run_all_tests()
