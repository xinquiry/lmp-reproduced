#!/usr/bin/env python3
"""
Visualize cohesive energy validation results.
"""

import matplotlib.pyplot as plt
import numpy as np

# Validation results
materials = ['Al\n(FCC)', 'Mg\n(HCP)', 'Ti\n(HCP)', 'TiB2-B\n(Hex)', 'TiB2-Ti\n(Hex)']
expected = [-3.36, -1.51, -4.87, -7.58, -4.50]
actual = [-3.32, -1.53, -4.87, -6.97, -4.22]
errors = [1.26, 1.22, 0.00, 8.05, 6.11]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Bar comparison
x = np.arange(len(materials))
width = 0.35

bars1 = ax1.bar(x - width/2, expected, width, label='Expected (Report 1-2)', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, actual, width, label='Calculated', color='coral', alpha=0.8)

ax1.set_ylabel('Cohesive Energy (eV/atom)')
ax1.set_title('Cohesive Energy: Expected vs Calculated')
ax1.set_xticks(x)
ax1.set_xticklabels(materials)
ax1.legend()
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax1.set_ylim(min(expected + actual) * 1.1, 0.5)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -12),
                textcoords="offset points",
                ha='center', va='top', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -12),
                textcoords="offset points",
                ha='center', va='top', fontsize=8)

# Plot 2: Error percentage
colors = ['green' if e < 5 else 'orange' if e < 10 else 'red' for e in errors]
bars3 = ax2.bar(x, errors, color=colors, alpha=0.8)
ax2.set_ylabel('Error (%)')
ax2.set_title('Validation Error by Material')
ax2.set_xticks(x)
ax2.set_xticklabels(materials)
ax2.axhline(y=5, color='green', linestyle='--', linewidth=1, label='5% threshold')
ax2.axhline(y=10, color='red', linestyle='--', linewidth=1, label='10% threshold')
ax2.legend()
ax2.set_ylim(0, 12)

# Add value labels on error bars
for bar, err in zip(bars3, errors):
    height = bar.get_height()
    ax2.annotate(f'{err:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('output/validation/cohesive_energy_results.png', dpi=150, bbox_inches='tight')
plt.savefig('output/validation/cohesive_energy_results.pdf', bbox_inches='tight')
print("Saved: output/validation/cohesive_energy_results.png")
print("Saved: output/validation/cohesive_energy_results.pdf")
plt.show()
