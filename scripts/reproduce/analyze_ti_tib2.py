import sys
import numpy as np
from pathlib import Path
from lmp_reproduced.core.post_processor import LAMMPSPostProcessor
from lmp_reproduced.models import InterfaceResult

def calculate_interface_energy_spatial(filename, references, metal_top=False):
    """
    Custom interface energy calculation with spatial reference assignment.
    """
    filename = Path(filename)
    if not filename.exists():
        print(f"File not found: {filename}")
        return

    post = LAMMPSPostProcessor()
    data = post.read_data(str(filename))
    
    # 1. Interface Detection (Scanning Peak PE)
    ymin, ymax = data.box_bounds[1]
    dy = (ymax - ymin) / 1000.0
    
    max_pe = -1e10
    yr_int = (ymin + ymax) / 2.0 
    
    print("  Scanning for interface peak...")
    for i in range(100, 900):
        y_lo = ymin + dy * (i - 1)
        y_hi = ymin + dy * i
        
        mask = (data.positions[:, 1] >= y_lo) & (data.positions[:, 1] < y_hi)
        if np.sum(mask) == 0: continue
        
        # Total PE in this slice
        total_pe = float(np.sum(data.potentials[mask]))
        if total_pe > max_pe:
            max_pe = total_pe
            yr_int = (y_lo + y_hi) / 2.0
            
    print(f"  Detected Interface at Y = {yr_int:.2f} A")
    
    # 2. Assign References
    excess_pot = np.zeros(data.n_atoms)
    
    ref_b = references.get("B_tib2", 0.0)
    ref_ti_metal = references.get("Ti_metal", 0.0)
    ref_ti_tib2 = references.get("Ti_tib2", 0.0)
    
    # Metal is Bottom (Y < Y_int) for our setup?
    # Setup was: Ti Slab (Bottom) then TiB2 (Top).
    # So Y < Y_int is Ti Metal. Y > Y_int is TiB2.
    
    for i in range(data.n_atoms):
        atype = data.atom_types[i]
        y = data.positions[i, 1]
        
        if atype == 1: # B (Only in TiB2)
            ref = ref_b
        elif atype == 2: # Ti (Could be Metal or TiB2)
            if y < yr_int:
                ref = ref_ti_metal
            else:
                ref = ref_ti_tib2
        else:
            ref = 0.0
            
        excess_pot[i] = data.potentials[i] - ref
        
    # 3. Integrate
    pot_total = float(np.sum(excess_pot))
    
    # Area
    xmin, xmax = data.box_bounds[0]
    zmin, zmax = data.box_bounds[2]
    area = (xmax - xmin) * (zmax - zmin)
    
    interface_energy = pot_total / area * 16021.8
    
    return InterfaceResult(
        position_y=yr_int,
        interface_energy=interface_energy,
        source_file=filename
    )

def main():
    print("--- Analyzing Ti/TiB2 Interface ---")
    
    # References from Report
    refs = {
        "B_tib2": -7.580,
        "Ti_metal": -4.873,
        "Ti_tib2": -4.497
    }
    
    res = calculate_interface_energy_spatial("simulations/Ti_TiB2/final.result", refs)
    if res:
        print(f"Final Interface Energy: {res.interface_energy:.2f} mJ/m^2")

if __name__ == "__main__":
    main()
