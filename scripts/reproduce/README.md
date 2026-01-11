# Reproduction Scripts

Scripts to reproduce simulation results from technical reports.

## Reference Documents

1. **Report 1-1**: 金属基复合材料界面与缺陷计算模块技术报告
   - Interface and defect calculation module
   - Covers: static minimization, NEB, dynamic relaxation, loading simulations

2. **Report 1-2**: 金属基复合材料温度相关物性计算模块技术报告
   - Temperature-dependent property calculation module
   - Covers: cohesive energy, defect energy, interface energy, dislocation core energy

## Available Scripts

### cohesive_energy.py

Validates cohesive energy calculations against Report 1-2.

**Expected Results:**

| Material | Structure | Lattice Parameters | Expected Ecoh | Actual | Error |
|----------|-----------|-------------------|---------------|--------|-------|
| Al | FCC | a = 4.032 Å | -3.36 eV/atom | -3.32 eV | 1.3% |
| Mg | HCP | a = 3.196 Å, c = 5.197 Å | -1.51 eV/atom | -1.53 eV | 1.2% |
| Ti | HCP | a = 2.92 Å, c = 4.77 Å | -4.87 eV/atom | -4.87 eV | 0.0% |
| TiB2 | Hex | a = 3.050 Å, c = 3.197 Å | B: -7.58 eV, Ti: -4.50 eV | B: -6.97 eV, Ti: -4.22 eV | 6-8% |

Note: Ti lattice parameters use values from the MEAM potential (library.meam) rather than Report 1-2
to match the potential's equilibrium structure.

**Usage:**

```bash
# Run all validations
python scripts/reproduce/cohesive_energy.py

# Run specific materials
python scripts/reproduce/cohesive_energy.py --materials al mg

# Custom output directory
python scripts/reproduce/cohesive_energy.py --output ./my_output

# Quiet mode
python scripts/reproduce/cohesive_energy.py -q
```

### defect_energy.py

Validates defect formation energy calculations against Report 1-2.

**Expected Results:**

| Defect | Host | Structure | Expected E_f | Actual | Error | Status |
|--------|------|-----------|--------------|--------|-------|--------|
| Mg substitutional | Al | FCC | 0.55 eV | 0.07 eV | 87% | See note |
| Ti vacancy | Ti | HCP | 1.78 eV | 1.83 eV | 2.8% | PASS |

**Note on Mg in Al:** The Liu Al-Mg EAM potential (`almg.liu.eam.alloy`) predicts a much lower
substitution energy than the report value. This is a known limitation of EAM potentials - they
are fitted to reproduce specific properties (cohesive energy, lattice constant, elastic moduli)
but may not accurately predict all defect configurations. The report value (0.55 eV) likely
comes from DFT calculations or experimental data.

**Formula used:**
- Substitutional: `E_f = E_total - (N_host × E_coh_host + N_solute × E_coh_solute)`
- Vacancy: `E_f = E_total - (N-1) × E_coh`

**Usage:**

```bash
# Run all defect validations
python scripts/reproduce/defect_energy.py

# Run specific defects
python scripts/reproduce/defect_energy.py --defects ti_vacancy

# Custom output directory
python scripts/reproduce/defect_energy.py --output ./my_output
```

## Prerequisites

1. LAMMPS must be installed and accessible (via PATH or `LMP_COMMAND` env var)
2. Potential files must be in `data/pot/` or specified via `LMP_POTENTIAL_DIR`
3. The `lmp_reproduced` package must be installed:

```bash
pip install -e .
# or
uv pip install -e .
```

## Output

Results are saved to `./output/validation/` by default:

```
output/validation/
├── al_fcc/
│   ├── data.struct      # Input structure
│   ├── in.lammps         # LAMMPS input script
│   ├── log.lammps        # LAMMPS log
│   └── final.result      # Final atomic configuration
├── mg_hcp/
│   └── ...
└── ...
```
