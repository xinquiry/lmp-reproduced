# lmp-reproduced

**金属基复合材料界面与缺陷计算模块** - A Python package for LAMMPS-based molecular simulations of metal matrix composites.

## 📁 Project Structure

```
lmp-reproduced/
├── src/lmp_reproduced/       # Core Python package
│   ├── core/                 # Structure, input generation, post-processing
│   ├── simulations/          # LAMMPS runners
│   └── visualization/        # Plotting utilities
├── scripts/reproduce/        # Reproduction scripts for technical reports
│   ├── cohesive_energy.py          # Validate cohesive energies
│   ├── defect_energy.py            # Calculate defect formation energies
│   ├── run_interface_calcs.py      # Al-Mg, Ti-TiB2 interface energies
│   └── temperature_cohesive_energy.py  # Temperature-dependent calculations
├── data/
│   ├── pot/                  # Interatomic potential files
│   └── struct/               # Pre-built structure files
├── output/                   # Simulation results
├── reports/                  # Technical reports (reference)
└── tests/                    # Unit tests
```

## 🚀 Quick Start

```bash
# Install
pip install -e .

# Run cohesive energy validation
python scripts/reproduce/cohesive_energy.py

# Generate temperature-dependent simulation inputs
python scripts/reproduce/temperature_cohesive_energy.py --dry-run
```

## 📊 Capabilities

| Simulation Type | Description | Workflow |
|-----------------|-------------|----------|
| Cohesive Energy | 0K binding energy | `cohesive_energy_workflow()` |
| Defect Energy | Vacancy/substitutional | `defect_energy_workflow()` |
| Interface Energy | Heterogeneous interfaces | `interface_annealing_workflow()` |
| Temperature-dependent | MD at various T | `--dry-run` + cluster |

## 🧪 Supported Materials

- **Metals**: Al, Cu, Ti, Mg
- **Ceramics**: SiC, TiB₂
- **Interfaces**: Al-Mg, Ti-TiB₂, SiC-Al

## 📖 Usage Examples

### 1. Cohesive Energy (0K)
```python
from lmp_reproduced import ALUMINUM, cohesive_energy_workflow

result = cohesive_energy_workflow(ALUMINUM, output_dir="output/al")
print(f"Al cohesive energy: {result.primary_energy:.3f} eV/atom")
```

### 2. Interface Energy
```python
from lmp_reproduced import TITANIUM, TIB2, interface_annealing_workflow

result = interface_annealing_workflow(
    bottom_config=TITANIUM,
    top_config=TIB2,
    output_dir="output/ti_tib2",
    references={"B_tib2": -7.58, "Ti_metal": -4.87, "Ti_tib2": -4.50},
    spatial_refs=True,
)
print(f"Interface energy: {result.interface_energy:.2f} mJ/m²")
```

### 3. Temperature-Dependent (Cluster)
```bash
# Generate input files
python scripts/reproduce/temperature_cohesive_energy.py --dry-run \
    --materials al ti sic tib2 \
    --temps 100 300 500 700 900

# Submit to cluster
cd output/temperature
find . -name 'run.sh' -exec sbatch {} \;
```

## 🔧 Requirements

- Python 3.10+
- LAMMPS (for running simulations)
- numpy, matplotlib

## 📚 References

- 技术报告1-1: 金属基复合材料界面与缺陷计算模块
- 技术报告1-2: 金属基复合材料温度相关物性计算模块

## 📄 License

MIT
