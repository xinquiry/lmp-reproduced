"""
Module for generating publication-quality charts for the report.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def create_combined_comparison_chart(output_path: Path):
    """Create an improved combined comparison chart including Mg."""
    # Data points (collected from report 1-2 and our simulations)
    MATERIALS = {
        "Al": {"calc": -3.3183, "ref": -3.360, "color": "#3498db", "bg": "#ebf5fb"},
        "Cu": {"calc": -3.5246, "ref": -3.540, "color": "#e74c3c", "bg": "#fdedec"},
        "Mg": {"calc": -1.3815, "ref": -1.510, "color": "#27ae60", "bg": "#eafaf1"}
    }

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.4)
    
    for i, (name, data) in enumerate(MATERIALS.items()):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor(data['bg'])
        
        calc, ref = abs(data['calc']), abs(data['ref'])
        
        ax.barh([0.6], [calc], height=0.3, color=data['color'], edgecolor='black', linewidth=1.5, label='Calc')
        ax.barh([0.2], [ref], height=0.3, color=data['color'], alpha=0.5, edgecolor='black', linewidth=1.5, label='Ref')
        
        ax.text(calc + 0.1, 0.6, f'{data["calc"]:.3f}', va='center', fontweight='bold')
        ax.text(ref + 0.1, 0.2, f'{data["ref"]:.2f}', va='center', alpha=0.7)
        
        error = abs((data['calc'] - data['ref']) / data['ref']) * 100
        ax.annotate(f'Err: {error:.1f}%', xy=(ref/2, 0.4), ha='center', fontweight='bold', color='darkgreen')
        
        ax.set_xlim(0, 4.5)
        ax.set_ylim(-0.1, 1)
        ax.set_yticks([0.6, 0.2])
        ax.set_yticklabels(['Calc', 'Ref'])
        ax.set_title(f'{name} (eV/atom)', fontweight='bold')
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    # Right Panel: Results Table
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')
    table_data = [
        ['Material', 'Calc (eV)', 'Ref (eV)', 'Error'],
        ['Aluminum', '-3.318', '-3.36', '1.2%'],
        ['Copper', '-3.525', '-3.54', '0.4%'],
        ['Magnesium', '-1.382', '-1.51', '8.5%']
    ]
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.5)
    
    plt.suptitle('Cohesive Energy Validation: Al, Cu, Mg Systems', fontsize=16, fontweight='bold', y=1.05)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

def create_module_flowchart(output_path: Path):
    """Create an expanded module architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    
    colors = {
        'cli': '#34495e',
        'core': '#3498db',
        'sim': '#9b59b6',
        'viz': '#27ae60',
        'data': '#f1c40f'
    }
    
    def box(x, y, w, h, color, text):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # CLI Layer
    box(3, 8.5, 4, 1, colors['cli'], 'CLI (typer)\n[extract|simulate|visualize|report]')
    
    # Core Logic
    box(0.5, 6, 2.5, 1, colors['core'], 'Extraction\n(python-docx)')
    box(3.75, 6, 2.5, 1, colors['sim'], 'Simulations\n(LAMMPS Python API)')
    box(7, 6, 2.5, 1, colors['viz'], 'Visualization\n(Matplotlib)')
    
    # Sub-components
    box(3.75, 4, 2.5, 1, colors['core'], 'Input Generator\n(Material Logic)')
    box(3.75, 2, 2.5, 1, colors['core'], 'Post-Processor\n(Result Parsing)')
    
    # Arrows (simplified)
    arrow_props = dict(arrowstyle='->', lw=1.5, color='#7f8c8d')
    ax.annotate('', xy=(1.75, 7.0), xytext=(3.5, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 7.0), xytext=(5, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(8.25, 7.0), xytext=(6.5, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 5.0), xytext=(5, 6.0), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 3.0), xytext=(5, 4.0), arrowprops=arrow_props)

    plt.title('MMC Reproduction System Architecture', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
