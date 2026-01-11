import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

def read_lammps_data(filename):
    """Simple parser for atomic positions."""
    atoms = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    reading_atoms = False
    for line in lines:
        if "Atoms" in line:
            reading_atoms = True
            continue
            
        if reading_atoms and len(line.strip()) > 0:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    atype = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    atoms.append([atype, x, y, z])
                except:
                    pass
    return np.array(atoms)

def plot_interface_3d(system_name, data_file, output_file):
    print(f"Rendering 3D Model for {system_name}...")
    atoms = read_lammps_data(data_file)
    if len(atoms) == 0: return

    # Configuration
    # Colors: Al=Silver, Mg=Green, B=Pink, Ti=Cyan/Grey
    # System specific mappings
    if "Al" in system_name:
        # Al/Mg: Type 1=Al, Type 2=Mg (Check structures.py)
        # Actually in create_al_mg: Al is bottom, Mg is top.
        # But wait, looking at structures.py:
        # create_al_mg_interface: slab1(Al) -> Type 1. slab2(Mg) -> shift types by 1 -> Type 2.
        # So 1=Al, 2=Mg.
        color_map = {1: '#C0C0C0', 2: '#32CD32'} # Silver, LimeGreen
        radius_map = {1: 1.43, 2: 1.60} # Atomic radii (approx scaling)
        label_map = {1: 'Al', 2: 'Mg'}
    else:
        # Ti/TiB2: 
        # Type 1=B, Type 2=Ti
        color_map = {1: '#FFB6C1', 2: '#87CEEB'} # LightPink, SkyBlue
        radius_map = {1: 0.82, 2: 1.47}
        label_map = {1: 'B', 2: 'Ti'}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot atoms
    # Use scatter, but sort by depth (Z) or just standard z-buffer?
    # Matplotlib 3D has poor z-ordering, so manual sorting helps slightly? No, complicated.
    # Just plot.
    
    unique_types = np.unique(atoms[:, 0])
    
    for t in unique_types:
        mask = atoms[:, 0] == t
        xs = atoms[mask, 1]
        ys = atoms[mask, 2] # Use Z as up? No, Y is stacking.
        zs = atoms[mask, 3]
        
        # Scaling size for visual appeal. Adjust multiplier.
        rad = radius_map.get(t, 1.0)
        s = (rad * 10) ** 2  
        
        ax.scatter(xs, ys, zs, 
                  c=color_map.get(t, 'gray'), 
                  s=s, 
                  edgecolors='black', 
                  linewidth=0.2,
                  alpha=0.9,
                  label=label_map.get(t, f'Type {int(t)}'))

    # Visual Tweaks
    ax.set_title(f"{system_name} Interface Structure", fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    
    # Hide Axes for cleaner look
    ax.set_axis_off()
    
    # View Angle
    ax.view_init(elev=15, azim=45) # Isometric-ish
    
    # Auto-scale
    # Matplotlib 3d aspect ratio is tricky.
    # We want equal aspect ratio.
    x_range = atoms[:, 1].max() - atoms[:, 1].min()
    y_range = atoms[:, 2].max() - atoms[:, 2].min()
    z_range = atoms[:, 3].max() - atoms[:, 3].min()
    max_range = max(x_range, y_range, z_range)
    
    mid_x = (atoms[:, 1].max() + atoms[:, 1].min()) * 0.5
    mid_y = (atoms[:, 2].max() + atoms[:, 2].min()) * 0.5
    mid_z = (atoms[:, 3].max() + atoms[:, 3].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight') # High DPI
    print(f"  Saved high-res to {output_file}")
    plt.close()

def main():
    # Al/Mg
    almg_path = "simulations/Al_Mg/interface.data"
    if Path(almg_path).exists():
        plot_interface_3d("Al-Mg", almg_path, "al_mg_model.png")
        
    # Ti/TiB2
    titib2_path = "simulations/Ti_TiB2/interface.data"
    if Path(titib2_path).exists():
        plot_interface_3d("Ti-TiB2", titib2_path, "ti_tib2_model.png")

if __name__ == "__main__":
    main()
