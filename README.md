# AW-PINN for solving 3D flow over buildings
Adoptive weights Physics-Informed Neural networks for solving 3D turbulent flow in steady-state condition. flow over a big cities in Japan.
two types of optimization is used simultaneously , Adam and L-BFGS. A kind of novel adaptive method for weight balance is applied for making the trainer more robust.
L2 regularization in Adam optimization is considered. 

@author: Amirreza
contact me : arezayan87@gmail.com

### This case solves 3D flow over a city.
the benchmark is Case E in Aij institute:
    Guidebook for CFD Predictions
    of Urban Wind Environment
    Architectural Institute of Japan

follow this site for data of the case E : 
https://www.aij.or.jp/jpn/publish/cfdguide/index_e.htm




### Step 1 - How to run this example
* provide data and boundary condition points file in csv format.
* share your data in your gdrive
* load your drive in colbab
* open the repository github link with google colab
* Run All

### Step 2 - consideration
* Epochs number is up to your case
* it's recommended to run it with GPU 

### Step 3 - Libraries
* torch
* pandas
* numpy
* matplotlib

