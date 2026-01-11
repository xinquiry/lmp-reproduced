"""
Tests for data models module.
"""

from pathlib import Path

from lmp_reproduced import (
    # Enums
    StructureType,
    SimulationType,
    MaterialSystem,
    # Configs
    MaterialConfig,
    MinimizationConfig,
    RelaxationConfig,
    # Results
    CohesiveEnergyResult,
    SimulationResult,
    # Predefined materials
    ALUMINUM,
    COPPER,
    MAGNESIUM,
    TITANIUM,
    SILICON_CARBIDE,
    TIB2,
)


def test_material_config_creation():
    """Test MaterialConfig dataclass creation."""
    config = MaterialConfig(
        element="Al",
        lattice_a=4.05,
        structure_type=StructureType.FCC,
    )

    assert config.element == "Al"
    assert config.lattice_a == 4.05
    assert config.structure_type == StructureType.FCC
    assert config.lattice_c is None

    print("test_material_config_creation passed")


def test_material_config_with_c():
    """Test MaterialConfig with c lattice parameter."""
    config = MaterialConfig(
        element="Mg",
        lattice_a=3.21,
        lattice_c=5.21,
        structure_type=StructureType.HCP,
    )

    assert config.element == "Mg"
    assert config.lattice_a == 3.21
    assert config.lattice_c == 5.21
    assert config.structure_type == StructureType.HCP

    print("test_material_config_with_c passed")


def test_material_config_auto_system():
    """Test automatic material system detection."""
    al_config = MaterialConfig(
        element="Al",
        lattice_a=4.05,
        structure_type=StructureType.FCC,
    )
    assert al_config.system == MaterialSystem.AL

    cu_config = MaterialConfig(
        element="Cu",
        lattice_a=3.615,
        structure_type=StructureType.FCC,
    )
    assert cu_config.system == MaterialSystem.CU

    mg_config = MaterialConfig(
        element="Mg",
        lattice_a=3.21,
        structure_type=StructureType.HCP,
    )
    assert mg_config.system == MaterialSystem.MG

    print("test_material_config_auto_system passed")


def test_predefined_materials():
    """Test predefined material constants."""
    # Aluminum (Report 1-2: a = 4.032 Å)
    assert ALUMINUM.element == "Al"
    assert ALUMINUM.lattice_a == 4.032
    assert ALUMINUM.structure_type == StructureType.FCC
    assert ALUMINUM.system == MaterialSystem.AL

    # Copper
    assert COPPER.element == "Cu"
    assert COPPER.lattice_a == 3.615
    assert COPPER.structure_type == StructureType.FCC

    # Magnesium (Report 1-2: a = 3.196 Å, c = 5.197 Å)
    assert MAGNESIUM.element == "Mg"
    assert MAGNESIUM.lattice_a == 3.196
    assert MAGNESIUM.lattice_c == 5.197
    assert MAGNESIUM.structure_type == StructureType.HCP

    # Titanium (Report 1-2: a = 2.945 Å, c = 4.687 Å)
    assert TITANIUM.element == "Ti"
    assert TITANIUM.lattice_a == 2.945
    assert TITANIUM.lattice_c == 4.687
    assert TITANIUM.structure_type == StructureType.HCP

    # Silicon Carbide
    assert SILICON_CARBIDE.element == "SiC"
    assert SILICON_CARBIDE.structure_type == StructureType.ZINCBLENDE

    # TiB2 (Report 1-2: a = 3.050 Å, c = 3.197 Å)
    assert TIB2.element == "TiB2"
    assert TIB2.lattice_a == 3.050
    assert TIB2.lattice_c == 3.197
    assert TIB2.structure_type == StructureType.HEXAGONAL

    print("test_predefined_materials passed")


def test_minimization_config():
    """Test MinimizationConfig dataclass."""
    config = MinimizationConfig(
        material=ALUMINUM,
        output_dir=Path("./output"),
        etol=1e-10,
        ftol=1e-10,
        maxiter=50000,
    )

    assert config.material == ALUMINUM
    assert config.output_dir == Path("./output")
    assert config.etol == 1e-10
    assert config.ftol == 1e-10
    assert config.maxiter == 50000
    # Check defaults
    assert config.maxeval == 100000
    assert config.fix_start == 0
    assert config.fix_end == 0

    print("test_minimization_config passed")


def test_relaxation_config():
    """Test RelaxationConfig dataclass."""
    config = RelaxationConfig(
        material=COPPER,
        output_dir=Path("./output"),
        temperature=500,
        n_steps=20000,
    )

    assert config.material == COPPER
    assert config.temperature == 500
    assert config.n_steps == 20000
    # Check defaults
    assert config.dump_interval == 0
    assert config.fix_start == 0

    print("test_relaxation_config passed")


def test_cohesive_energy_result():
    """Test CohesiveEnergyResult dataclass."""
    result = CohesiveEnergyResult(
        energies={1: -3.36, 2: -3.54},
        n_atoms={1: 100, 2: 100},
        source_file=Path("./test.result"),
    )

    # Test properties
    assert result.primary_energy == -3.36
    assert result.total_atoms == 200

    # Test direct access
    assert result.energies[1] == -3.36
    assert result.energies[2] == -3.54
    assert result.n_atoms[1] == 100

    print("test_cohesive_energy_result passed")


def test_simulation_result():
    """Test SimulationResult dataclass."""
    # Successful result
    result = SimulationResult(
        success=True,
        output_dir=Path("./output"),
        log_file=Path("./output/log.lammps"),
        result_file=Path("./output/final.result"),
    )

    assert result.success is True
    assert result.error_message is None

    # Failed result
    failed_result = SimulationResult(
        success=False,
        output_dir=Path("./output"),
        error_message="LAMMPS not found",
    )

    assert failed_result.success is False
    assert failed_result.error_message == "LAMMPS not found"
    assert failed_result.result_file is None

    print("test_simulation_result passed")


def test_structure_type_enum():
    """Test StructureType enum values."""
    assert StructureType.FCC.value == "fcc"
    assert StructureType.HCP.value == "hcp"
    assert StructureType.BCC.value == "bcc"
    assert StructureType.ZINCBLENDE.value == "zincblende"
    assert StructureType.HEXAGONAL.value == "hexagonal"

    print("test_structure_type_enum passed")


def test_simulation_type_enum():
    """Test SimulationType enum values."""
    assert SimulationType.MINIMIZE.value == "minimize"
    assert SimulationType.NEB.value == "neb"
    assert SimulationType.RELAXATION.value == "relaxation"
    assert SimulationType.LOADING_STRESS.value == "loading_stress"
    assert SimulationType.LOADING_STRAIN.value == "loading_strain"

    print("test_simulation_type_enum passed")


def run_all_tests():
    """Run all model tests."""
    print("\n=== Testing Data Models ===\n")

    test_material_config_creation()
    test_material_config_with_c()
    test_material_config_auto_system()
    test_predefined_materials()
    test_minimization_config()
    test_relaxation_config()
    test_cohesive_energy_result()
    test_simulation_result()
    test_structure_type_enum()
    test_simulation_type_enum()

    print("\n=== All model tests passed! ===\n")


if __name__ == "__main__":
    run_all_tests()
