# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 00:12:07 2024

@author: Amirreza
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(8, 6))

# Full Domain
full_domain = patches.Rectangle((0, 0), 10, 10, linewidth=2, edgecolor='black', facecolor='none')
ax.add_patch(full_domain)
ax.text(5, 10.5, 'Full Domain (\u03A9)', ha='center')

# Near-Wall Zone (\u03A9_1)
near_wall = patches.Rectangle((0, 0), 10, 4, linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.5)
ax.add_patch(near_wall)
ax.text(5, 2, 'Near-Wall Zone (\u03A9₁)', ha='center')

# Far-Wall Zone (\u03A9_2)
far_wall = patches.Rectangle((0, 4), 10, 6, linewidth=1, edgecolor='green', facecolor='lightgreen', alpha=0.5)
ax.add_patch(far_wall)
ax.text(5, 7, 'Far-Wall Zone (\u03A9₂)', ha='center')

# Interface (\u0393)
ax.axhline(y=4, color='red', linestyle='--', linewidth=2)
ax.text(10.2, 4, 'Interface (\u0393)', va='center', color='red')

# Formatting
ax.set_xlim(-1, 12)
ax.set_ylim(-1, 12)
ax.set_aspect('equal')
ax.axis('off')

plt.show()
