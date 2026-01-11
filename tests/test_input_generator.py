"""
Tests for LAMMPS Input Generator module.
"""

import tempfile
from pathlib import Path

from lmp_reproduced import (
    LAMMPSInputGenerator,
    MaterialSystem,
    MaterialConfig,
    StructureType,
    MinimizationConfig,
    RelaxationConfig,
    generate_minimization_script,
    generate_relaxation_script,
)


def test_minimize_al():
    """Test minimization input generation for Al system."""
    gen = LAMMPSInputGenerator(MaterialSystem.AL)
    content = gen.generate_minimize(
        fix_start=0,
        fix_end=0,
    )

    # Verify key elements are present
    assert "pair_style     eam/alloy" in content
    assert "AlCu.eam.alloy" in content
    assert "min_style      fire" in content
    assert "minimize" in content

    print("test_minimize_al passed")


def test_minimize_tib():
    """Test minimization input generation for Ti-B system."""
    gen = LAMMPSInputGenerator(MaterialSystem.TI_B)
    content = gen.generate_minimize(
        fix_start=1,
        fix_end=1000,
    )

    # Verify key elements
    assert "pair_style     meam" in content
    assert "library.meam B Ti" in content
    assert "group          region1 id 1:1000" in content
    assert "fix            freeze region1 setforce 0 0 0" in content

    print("test_minimize_tib passed")


def test_neb():
    """Test NEB input generation."""
    gen = LAMMPSInputGenerator(MaterialSystem.AL)
    content = gen.generate_neb(
        fix_start=0,
        fix_end=1942,
        spring_constant=1.0,
        final_config="data.final",
    )

    # Verify NEB-specific elements
    assert "fix            2 all neb 1.0 parallel ideal" in content
    assert "neb" in content
    assert "data.final" in content

    print("test_neb passed")


def test_relaxation():
    """Test dynamic relaxation input generation."""
    gen = LAMMPSInputGenerator(MaterialSystem.TI_B)
    content = gen.generate_relaxation(
        temperature=300,
        fix_start=0,
        fix_end=0,
        n_steps=10000,
    )

    # Verify MD elements
    assert "velocity       region2 create 300" in content
    assert "fix            relax region2 nvt temp 300 300" in content
    assert "run            10000" in content

    print("test_relaxation passed")


def test_constant_stress_loading():
    """Test constant stress loading input generation."""
    gen = LAMMPSInputGenerator(MaterialSystem.TI_B)
    content = gen.generate_loading_constant_stress(
        temperature=300,
        fix_start=0,
        fix_end=1429,
        stress_xx=0,
        stress_yy=0,
        stress_zz=0,
        stress_xy=0.8,
        stress_xz=0,
        stress_yz=0,
        n_steps=100000,
    )

    # Verify stress loading elements
    assert "change_box     all triclinic" in content
    assert "npt temp 300 300" in content
    assert "xy 0.8 0.8" in content

    print("test_constant_stress_loading passed")


def test_constant_strain_rate_loading():
    """Test constant strain rate loading input generation."""
    gen = LAMMPSInputGenerator(MaterialSystem.TI_B)
    content = gen.generate_loading_constant_strain_rate(
        temperature=300,
        fix_start=0,
        fix_end=43559,
        deform_direction="xy",
        strain_rate=5e-4,
        n_steps=200000,
    )

    # Verify strain rate elements
    assert "deform" in content
    assert "xy erate 0.0005" in content

    print("test_constant_strain_rate_loading passed")


def test_generate_minimization_script_with_config():
    """Test minimization script generation using dataclass config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        material = MaterialConfig(
            element="Al",
            lattice_a=4.05,
            structure_type=StructureType.FCC,
        )

        config = MinimizationConfig(
            material=material,
            output_dir=output_dir,
            etol=1e-10,
            ftol=1e-10,
        )

        content = generate_minimization_script(config)

        assert "minimize" in content
        assert "1e-10" in content
        assert "eam/alloy" in content

    print("test_generate_minimization_script_with_config passed")


def test_generate_relaxation_script_with_config():
    """Test relaxation script generation using dataclass config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        material = MaterialConfig(
            element="Cu",
            lattice_a=3.615,
            structure_type=StructureType.FCC,
        )

        config = RelaxationConfig(
            material=material,
            output_dir=output_dir,
            temperature=500,
            n_steps=5000,
        )

        content = generate_relaxation_script(config)

        assert "nvt temp 500 500" in content
        assert "run            5000" in content

    print("test_generate_relaxation_script_with_config passed")


def run_all_tests():
    """Run all input generator tests."""
    print("\n=== Testing LAMMPS Input Generator ===\n")

    test_minimize_al()
    test_minimize_tib()
    test_neb()
    test_relaxation()
    test_constant_stress_loading()
    test_constant_strain_rate_loading()
    test_generate_minimization_script_with_config()
    test_generate_relaxation_script_with_config()

    print("\n=== All input generator tests passed! ===\n")


if __name__ == "__main__":
    run_all_tests()
