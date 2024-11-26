# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 10:41:36 2024

# This codecreatess random points in the domain 
  and generates a sampleDict for postPricessin by OPENFoam 2012.
@author: Amirreza
"""

import numpy as np
import os
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.Execution.BasicRunner import BasicRunner

# Specify the case directory
case_dir = r"E:\FOAM_PINN\cavHeat\twoD_lamin_over_box"
folder_name = "2D_FoamCase"

# Sampling bounds (define the domain limits)
x_min, x_max = 0.0, 10.0
y_min, y_max = 0.0, 5.0
z_min, z_max = 0.0, 0.0

# Number of random points
num_points = 50

# Generate random points
random_points = np.column_stack((
    np.random.uniform(x_min, x_max, num_points),
    np.random.uniform(y_min, y_max, num_points),
    np.random.uniform(z_min, z_max, num_points),
))

# Create the sampleDict file
sample_dict_content = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      sampleDict;
}}
type sets;
interpolationScheme cellPoint;

setFormat csv;

sets
(
    randomPoints
    {{
        type cloud;
        axis xyz;
        points
        (
"""

for point in random_points:
    sample_dict_content += f"            ({point[0]} {point[1]} {point[2]})\n"

sample_dict_content += """
        );
    }
);
fields (U T p); // Add more fields if needed
"""

# Write sampleDict to the case directory
sample_dict_path = os.path.join(case_dir, folder_name ,"system", "sampleDict")
with open(sample_dict_path, "w") as f:
    f.write(sample_dict_content)

