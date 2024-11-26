# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 02:49:53 2024

@author: Amirreza
"""

import numpy as np
import pyvista as pv
import pandas as pd
pv.global_theme.allow_empty_mesh = True
# Load the STL file
stl_mesh = pv.read("mesh/buildings.stl")

# Save as a VTK file
#stl_mesh.save("mesh/geometery_caseE.vtk")

print("STL file successfully converted to VTK format.")

# Load the building geometry from the VTK file
#geometry_mesh = pv.read("mesh/geometery_caseE.vtk")
geometry_mesh = pv.read("mesh/buildings.stl")

# Define the bounds of the domain
x_min, x_max = -503, 503  # Replace with your domain's x-bounds
y_min, y_max = -502, 502  # Replace with your domain's y-bounds
z_min, z_max = 0, 300    # Replace with your domain's z-bounds

# Number of random points to generate
num_points = 5000

# Generate random points uniformly within the domain bounds
x_rand = np.random.uniform(x_min, x_max, num_points)
y_rand = np.random.uniform(y_min, y_max, num_points)
 
# bias in the direction of z (boundary layer consideration)
z_rand = np.random.exponential(scale=1, size=num_points) 

z_rand = (z_rand / max(z_rand)) * z_max

random_points = np.column_stack((x_rand, y_rand, z_rand))

# Convert random points to a PyVista PolyData object
points_cloud = pv.PolyData(random_points)

# Check if points are inside the building geometries
# `inside_out=False` ensures "inside" points are marked
selected = geometry_mesh.select_enclosed_points(points_cloud, tolerance=1e-6, inside_out=False)

# Extract the mask for points outside the building geometries
inside_mask = selected["SelectedPoints"]  # 1 = Inside, 0 = Outside

# Keep only points outside the geometry
valid_points = random_points[inside_mask.all() == 0]

# Output the results
print(f"Total random points generated: {num_points}")
print(f"Points outside the buildings: {len(valid_points)}")


#np.savetxt("valid_points.csv" ,valid_points[0 , : , :] ,  delimiter=',' , fmt='%1.4e')

"""
pl = pv.Plotter()
pl.add_mesh(geometry_mesh, color="gray", opacity=0.35, label="Buildings")
pl.add_points(valid_points, color="green", point_size=10, label="Filtered Points")
pl.add_points(random_points, color="red", point_size=2, label="total Points")
#pl.save_graphic("3dplots.eps")
pl.add_legend()
pl.show()
"""

for i in range(len(valid_points[0 , : , :])):
    if valid_points[0 , i , 0]  < -60.7 and valid_points[0 , i , 0]  > -67.5 and valid_points[0 , i , 1]  > -17.1 and valid_points[0 , i , 1]  < 18.3  and valid_points[0 , i , 2]  <60:
        print("ok")
        
for i in range(len(valid_points[0 , : , :])):
    if valid_points[0 , i , 0]  < -60.7 and valid_points[0 , i , 0]  > -67.5 :
        #print(f" point:[0 , {i} , 0] in X direction overlaped")
        if valid_points[0 , i , 1]  > -17.1 and valid_points[0 , i , 1]  < 18.3:
            print("in Y direction overlaped")
            if valid_points[0 , i , 2]  <60:
                valid_points[0 , i , 2] = valid_points[0 , i , 2] + 60
                print(f" point:[0 , {i} , ]in X, Y , Z directions the point is overlaped")