# -*- coding: utf-8 -*-
"""
3D caseE - incompressible flow over city
    
"""

file_data = r"caseE_Data.csv"
#file_data = r"data_2D_Lamin.csv"
fileBC = r"caseE_BC.csv"
#fileTest = r"2D_newTest.csv"
save_to_path = 'E:/FOAM_PINN/cavHeat/CaseE/Main_results/'

#######################################################################
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import os
import time
from scipy.interpolate import griddata
import scipy
from torch.utils.data import DataLoader, random_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Swish activation function
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x = x * torch.sigmoid(x)  # inplace modification removed for safer execution
            return x
        else:
            return x * torch.sigmoid(x)

# Define the PINN model
class PINN_u(nn.Module):
    def __init__(self, layers):
        super(PINN_u, self).__init__()

        # Initialize neural network layers
        self.layers = nn.ModuleList()
        # Dynamically create layers based on input list 'layers'
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        # Apply Xavier initialization to all weights
        self.apply(self.xavier_init)

    # Xavier initialization function
    def xavier_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    # Forward pass using Swish activation
    def forward(self, x):
        for layer in self.layers[:-1]:  # Apply Swish activation for all layers except the last
            x = Swish()(layer(x))
        x = self.layers[-1](x)  # Output layer with no activation
        return x
    
class PINN_v(nn.Module):
    def __init__(self, layers):
        super(PINN_v, self).__init__()

        # Initialize neural network layers
        self.layers = nn.ModuleList()
        # Dynamically create layers based on input list 'layers'
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        # Apply Xavier initialization to all weights
        self.apply(self.xavier_init)

    # Xavier initialization function
    def xavier_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    # Forward pass using Swish activation
    def forward(self, x):
        for layer in self.layers[:-1]:  # Apply Swish activation for all layers except the last
            x = Swish()(layer(x))
        x = self.layers[-1](x)  # Output layer with no activation
        return x

class PINN_w(nn.Module):
    def __init__(self, layers):
        super(PINN_w, self).__init__()

        # Initialize neural network layers
        self.layers = nn.ModuleList()
        # Dynamically create layers based on input list 'layers'
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        # Apply Xavier initialization to all weights
        self.apply(self.xavier_init)

    # Xavier initialization function
    def xavier_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    # Forward pass using Swish activation
    def forward(self, x):
        for layer in self.layers[:-1]:  # Apply Swish activation for all layers except the last
            x = Swish()(layer(x))
        x = self.layers[-1](x)  # Output layer with no activation
        return x


class PINN_p(nn.Module):
    def __init__(self, layers):
        super(PINN_p, self).__init__()

        # Initialize neural network layers
        self.layers = nn.ModuleList()
        # Dynamically create layers based on input list 'layers'
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        # Apply Xavier initialization to all weights
        self.apply(self.xavier_init)
    # Xavier initialization function
    
    def xavier_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    # Forward pass using Swish activation
    def forward(self, x):
        for layer in self.layers[:-1]:  # Apply Swish activation for all layers except the last
            x = Swish()(layer(x))
        x = self.layers[-1](x)  # Output layer with no activation
        return x


class PINN_T(nn.Module):
    def __init__(self, layers):
        super(PINN_T, self).__init__()

        # Initialize neural network layers
        self.layers = nn.ModuleList()


        # Dynamically create layers based on input list 'layers'
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        # Apply Xavier initialization to all weights
        self.apply(self.xavier_init)

    # Xavier initialization function
    def xavier_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    # Forward pass using Swish activation
    def forward(self, x):
        for layer in self.layers[:-1]:  # Apply Swish activation for all layers except the last
            x = Swish()(layer(x))

        x = self.layers[-1](x)  # Output layer with no activation
        return x


# use the modules apply function to recursively apply the initialization
#net1 = Net1().to(device)
# Example usage
NUM_NEURONS = int(40)
NUM_LAYER = 8
dim_input = int(3)
dim_output = int(1)
layers = np.zeros(NUM_LAYER)
layers = [dim_input ]

for i in range(1 , NUM_LAYER+1):
    layers.append(NUM_NEURONS)
    if i==NUM_LAYER:
        layers.append(dim_output)


#print(layers)
model_u = PINN_u(layers).to(device)
model_v = PINN_v(layers).to(device)
model_w = PINN_w(layers).to(device)
model_p = PINN_p(layers).to(device)
model_T = PINN_T(layers).to(device)


##############################################################################
##############################################################################

#Add Gaussian noise to tensors
def add_gaussian_noise(tensor, noise = 0.02):
    return (tensor  * noise * torch.randn_like(tensor)) + tensor

#Normalize inputs
def normal_inputs(df): #df is a dataframe
    normal_df = (2 * (df - df.min()) / (df.max() - df.min() + 1e-10  )) - 1
    return normal_df

#calculate PDE loss ( Momentum & Continuity & Energy equation)
def pde_residuals(model_u , model_v , model_w , model_p  , model_T, x , y , z ):

    u = model_u(torch.cat((x,y , z) , dim = 1))
    v = model_v(torch.cat((x,y , z) , dim = 1))
    w = model_w(torch.cat((x,y , z) , dim = 1))
    p = model_p(torch.cat((x,y , z) , dim = 1))
    T = model_T(torch.cat((x,y , z) , dim = 1))
    
    
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True )[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True )[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True )[0]
    u_xx = torch.autograd.grad(u_x , x, grad_outputs=torch.ones_like(u_x), create_graph=True )[0]
    u_yy = torch.autograd.grad(u_y , y, grad_outputs=torch.ones_like(u_y), create_graph=True )[0]
    u_zz = torch.autograd.grad(u_z , z, grad_outputs=torch.ones_like(u_z), create_graph=True )[0]

    v_x = torch.autograd.grad(v , x , grad_outputs=torch.ones_like(v), create_graph=True )[0]
    v_y = torch.autograd.grad(v , y , grad_outputs=torch.ones_like(v), create_graph=True )[0]
    v_z = torch.autograd.grad(v , z , grad_outputs=torch.ones_like(v), create_graph=True )[0]
    v_xx = torch.autograd.grad(v_x , x, grad_outputs=torch.ones_like(v_x), create_graph=True )[0]
    v_yy = torch.autograd.grad(v_y , y, grad_outputs=torch.ones_like(v_y), create_graph=True )[0]
    v_zz = torch.autograd.grad(v_z , z, grad_outputs=torch.ones_like(v_z), create_graph=True )[0]
    
    w_x = torch.autograd.grad(w , x , grad_outputs=torch.ones_like(w), create_graph=True )[0]
    w_y = torch.autograd.grad(w , y , grad_outputs=torch.ones_like(w), create_graph=True )[0]
    w_z = torch.autograd.grad(w , z , grad_outputs=torch.ones_like(w), create_graph=True )[0]
    w_xx = torch.autograd.grad(w_x , x, grad_outputs=torch.ones_like(w_x), create_graph=True )[0]
    w_yy = torch.autograd.grad(w_y , y, grad_outputs=torch.ones_like(w_y), create_graph=True )[0]
    w_zz = torch.autograd.grad(w_z , z, grad_outputs=torch.ones_like(w_z), create_graph=True )[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True )[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True )[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True )[0]

    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True )[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True )[0]
    T_z = torch.autograd.grad(T, z, grad_outputs=torch.ones_like(T), create_graph=True )[0]

    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True )[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True )[0]
    T_zz = torch.autograd.grad(T_z, z, grad_outputs=torch.ones_like(T_z), create_graph=True )[0]

    alpha = 0.002
    mu = 0.01
    continuity_residual = u_x + v_y + w_z
    momentum_u_residual =  lambda_1 * (u*u_x + v*u_y + w*u_z) + p_x - mu * lambda_2 * (u_xx + u_yy + u_zz)
    momentum_v_residual =  lambda_1 * (u*v_x + v*v_y + w*v_z) + p_y - mu * lambda_2 * (v_xx + v_yy + v_zz)
    momentum_w_residual =  lambda_1 * (u*w_x + v*w_y + w*w_z) + p_z - mu * lambda_2 * (w_xx + w_yy + w_zz)
    energy_residual = lambda_3 *(u * T_x + v * T_y  + w * T_z)  - alpha * lambda_4 * (T_xx + T_yy + T_zz ) #- alpha_t_diff

    loss_mse = nn.MSELoss()
    #Note our target is zero. It is residual so we use zeros_like
    loss_pde = loss_mse(continuity_residual,torch.zeros_like(continuity_residual)) + loss_mse(momentum_u_residual,torch.zeros_like(momentum_u_residual)) + loss_mse(momentum_v_residual,torch.zeros_like(momentum_v_residual))  + loss_mse(momentum_w_residual,torch.zeros_like(momentum_w_residual)) + loss_mse(energy_residual,torch.zeros_like(energy_residual))
    return loss_pde

#Loading data without splitting
def simple_data_loader(file_name):
    df =  normal_inputs(pd.read_csv(file_name))
    x = torch.tensor(df[['x']].values, dtype=torch.float32 , requires_grad = True , device = device).reshape(-1 , 1)
    y = torch.tensor(df[['y']].values, dtype=torch.float32 , requires_grad = True , device = device).reshape(-1 , 1)
    z = torch.tensor(df[['z']].values, dtype=torch.float32 , requires_grad = True , device = device).reshape(-1 , 1)
    truth = torch.tensor(df[['u' ,'v' , 'w' ,'T' , 'p']].values, dtype=torch.float32 , requires_grad = True , device = device).reshape(-1 , 5)

    return x , y  ,z , truth


#Loading data considering splitting data (train & test)
def data_loader(file_data , test_postion):

    data =  normal_inputs(pd.read_csv(file_data))
    X = data[['x' , 'y' , 'z']].values
    y = data[['y']].values
    truth = data[['u' ,'v' , 'w','T' , 'p']].values

    # Split data into training and validation sets
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(X , truth , test_size=test_postion, random_state=42)
    # Convert to PyTorch tensors
    inputs_train_tensor = torch.tensor(inputs_train, dtype=torch.float32 , requires_grad= True , device = device)
    inputs_val_tensor = torch.tensor(inputs_val, dtype=torch.float32 , requires_grad= True , device = device)
    targets_train_tensor = torch.tensor(targets_train, dtype=torch.float32 , requires_grad= True , device = device)
    targets_val_tensor = torch.tensor(targets_val, dtype=torch.float32 , requires_grad= True , device = device)

    # Create TensorDataset for training and validation
    train_dataset = TensorDataset(inputs_train_tensor, targets_train_tensor)
    val_dataset = TensorDataset(inputs_val_tensor, targets_val_tensor)

    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    return train_loader , val_loader


# calculation of Data loss
def data_loss(model_u , model_v, model_w , model_p , model_T , x , y ,z,  truth):
    loss_mse = nn.MSELoss()

    u_pred = model_u(torch.cat((x, y , z) , dim = 1))
    v_pred = model_v(torch.cat((x, y , z) , dim = 1))
    w_pred = model_w(torch.cat((x, y , z) , dim = 1))
    
    p_pred = model_p(torch.cat((x , y , z) , dim = 1))
    T_pred = model_T(torch.cat((x , y , z) , dim = 1))



    u_truth = truth[: , 0].reshape(-1 , 1).to(device)
    u_loss = loss_mse(u_pred, u_truth)
    
    v_truth = truth[: , 1].reshape(-1 , 1).to(device)
    v_loss = loss_mse(v_pred, v_truth)
    
    w_truth = truth[: , 2].reshape(-1 , 1).to(device)
    w_loss = loss_mse(w_pred, w_truth)

    p_truth = truth[: , 4].reshape(-1 , 1).to(device)
    p_loss = loss_mse(p_pred, p_truth)

    T_truth = truth[: , 3].reshape(-1 , 1).to(device)
    T_loss = loss_mse(T_pred, T_truth)

    loss = u_loss + v_loss  + w_loss + p_loss + T_loss


    return loss , u_loss  ,v_loss , w_loss , p_loss , T_loss

def impose_boundary_conditions(model_u , model_v, model_w  , model_p , model_T , fileBC):
    # Directly set model predictions to boundary values (hard enforcement)
    with torch.no_grad():

        x_coords , y_coords , z_coords , boundary_values = simple_data_loader(fileBC)

        # Boundary conditions for u, v, T, and p
        model_u.eval()
        model_v.eval()
        model_w.eval()
        model_p.eval()
        model_T.eval()

        model_u(torch.cat((x_coords , y_coords , z_coords) , dim = 1)).copy_(boundary_values[: , 0].reshape(-1 , 1))
        model_v(torch.cat((x_coords , y_coords , z_coords) , dim = 1)).copy_(boundary_values[: , 1].reshape(-1 , 1))
        model_w(torch.cat((x_coords , y_coords , z_coords) , dim = 1)).copy_(boundary_values[: , 2].reshape(-1 , 1))
        model_p(torch.cat((x_coords , y_coords , z_coords) , dim = 1)).copy_(boundary_values[: , 4].reshape(-1 , 1))
        model_T(torch.cat((x_coords , y_coords , z_coords) , dim = 1)).copy_(boundary_values[: , 3].reshape(-1 , 1))
        
        model_u.train()
        model_v.train()
        model_w.train()
        model_p.train()
        model_T.train()

#Generation of collocation point considering location of the buildings    
def collocation_points(x_min , y_min , z_min, x_max , y_max, z_max , cube_x_min, cube_x_max ,
                        cube_y_min, cube_y_max , cube_z_min , cube_z_max ,
                        num_collocation_points ):

    #Stage4-0: Collocation points definition
    # Generate random collocation points within the domain
    np.random.seed(50)

    collocation_points = np.random.rand(num_collocation_points, 3)
    collocation_points[:, 0] = collocation_points[:, 0] * (x_max - x_min) + x_min  # Scale to x bounds
    collocation_points[:, 1] = collocation_points[:, 1] * (y_max - y_min) + y_min  # Scale to y bounds
    collocation_points[:, 2] = collocation_points[:, 2] * (z_max - z_min) + z_min  # Scale to z bounds
    


    # Filter out points that fall within the cube's region
    not_normal_filtered_points = collocation_points[
        ~(
        (collocation_points[:, 0] >= cube_x_min) &
        (collocation_points[:, 0] <= cube_x_max) &
        (collocation_points[:, 1] >= cube_y_min) &
        (collocation_points[:, 1] <= cube_y_max) &
        (collocation_points[:, 2] >= cube_z_min) &
        (collocation_points[:, 2] <= cube_z_max)
        )]

    filtered_points = normal_inputs(not_normal_filtered_points)
    collocation_points_tensor = torch.tensor(filtered_points, dtype=torch.float32 ,  requires_grad=True , device = device)
    X_c = collocation_points_tensor[: , 0].reshape(-1 , 1)
    Y_c = collocation_points_tensor[: , 1].reshape(-1 , 1)
    Z_c = collocation_points_tensor[: , 2].reshape(-1 , 1)

        #plot selected points in domain
    plt.figure(dpi = 100)
    plt.plot(np.zeros(10) , np.linspace(y_min, y_max , 10) , 'r')
    plt.plot(x_max * np.ones(10) , np.linspace(y_min, y_max , 10) , 'r')

    plt.plot(np.linspace(x_min, cube_x_min , 10) , np.zeros(10) , 'r')
    plt.plot(np.linspace(cube_x_max, x_max , 10) , np.zeros(10) , 'r')
    plt.plot(np.linspace(x_min, x_max , 10) , y_max * np.ones(10) , 'r')
    plt.scatter(not_normal_filtered_points[: , 0] , not_normal_filtered_points[: , 1] , s=np.ones(len(not_normal_filtered_points)) * 4 , color = 'k' , label = "collocation points")

    plt.plot([4.5, 4.5 , 5.5 , 5.5], [0.0 , 1.0 , 1.0 , 0.0], 'tab:red',  linewidth=4)
    plt.xlabel("Ground and cube")
    plt.ylabel("Inlet")
    plt.grid()
    plt.legend(loc = "upper right")
    plt.savefig(save_to_path + "colloc_points.png")
    plt.show()

    return X_c , Y_c , Z_c

#calculation od noisy loss (Data Assimilation)
def noisy_data_loss(model_u , model_v, model_w , model_p , model_T , x , y ,z, truth ):

    #x_noisy = add_gaussian_noise(x.reshape(-1 , 1))
    #y_noisy = add_gaussian_noise(y.reshape(-1 , 1))
    x_noisy = x.reshape(-1 , 1)
    y_noisy = y.reshape(-1 , 1)
    z_noisy = z.reshape(-1 , 1)
    
    truth_noisy = add_gaussian_noise(truth.reshape(-1 , 5))
    noisy_int_loss , _ , _ , _ ,_ , _ = data_loss(model_u , model_v, model_w , model_p , model_T, x_noisy , y_noisy , z_noisy , truth_noisy)

    return noisy_int_loss

def total_loss(model_u , model_v, model_w , model_p , model_T , x_c , y_c , z_c ,x_int , y_int , z_int , truth_int , x_noisy , y_noisy , z_noisy , truth_noisy ):
    pde_loss = pde_residuals(model_u , model_v, model_w , model_p  , model_T, x_c , y_c , z_c )
    loss_data , _ , _ , _ , _ , _ = lambda_data * data_loss(model_u , model_v, model_w , model_p , model_T ,x_int , y_int , z_int , truth_int)
    noisy_loss = lambda_noisy * noisy_data_loss(model_u , model_v, model_w , model_p , model_T, x_noisy , y_noisy , z_noisy , truth_noisy )

    loss = pde_loss + loss_data + noisy_loss
    return loss , pde_loss , loss_data , noisy_loss

def plot_solution(x_star,y_star , z_star , u_star , title):

    lb = [min(x_star) , min(y_star) , min(z_star)]
    ub = [max(x_star) , max(y_star) , max(z_star)]
    

    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    Z = np.linspace(lb[2], ub[2], nn)
    X, Y = np.meshgrid(x,y)
    XX , ZZ = np.meshgrid(X , Z)
    points = np.column_stack((x_star.flatten(), y_star.flatten()))
    points_XZ = np.column_stack((x_star.flatten()  , z_star.flatten()))

    U_star = griddata(points, u_star.flatten(), (X, Y), method='cubic')
    UU_star = griddata(points_XZ , u_star.flatten() , (XX , ZZ) , method = 'cubic')

    plt.figure(dpi = 100)
    plt.pcolor(X,Y,U_star,  vmin=-1, vmax=1 , cmap = 'jet')
    plt.title(title)
    plt.colorbar()
    plt.grid()
    plt.show()
    
    plt.figure(dpi = 100)
    plt.pcolor(XX,ZZ,U_star,  vmin=-1, vmax=1 , cmap = 'jet')
    plt.title(title)
    plt.colorbar()
    plt.grid()
    plt.show()

def train(model_u , model_v, model_w, model_p , model_T , fileData , fileBC , nIter):
    LR = 1e-3
    global lambda_1, lambda_2, lambda_3, lambda_4
    lambda_1 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
    lambda_2 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
    lambda_3 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
    lambda_4 = torch.nn.Parameter(torch.ones(1, requires_grad=True))


    #bounds of overall domain
    x_min = -503
    y_min = -502
    z_min = 0

    x_max = 503
    y_max = 502
    z_max = 300
    

    num_collocation_points = 500 #COLLOCATION POINTS NUMBER

    # Define buildings boundaries within the domain (example values)
    cube_x_min, cube_x_max = -201, 203  # x bounds of cubes
    cube_y_min, cube_y_max = -202, 197  # y bounds of cubes
    cube_z_min, cube_z_max = 0, 60.0  # z bounds of cubes
    X_c , Y_c  , Z_c = collocation_points(x_min , y_min , z_min ,  x_max , y_max , z_max ,  cube_x_min, cube_x_max ,
                        cube_y_min, cube_y_max , cube_z_min , cube_z_max , num_collocation_points )
    train_loader , valid_loader = data_loader(fileData ,0.3 )
    x_obj ,y_obj , z_obj , truth_obj = simple_data_loader(fileData)

    x_noisy , y_noisy , z_noisy , truth_noisy = simple_data_loader(fileData)  # simpleDta loded for addig noise and calculation of noisy loss


    x_boundary , y_boundary , z_boundary , truth_boundary = simple_data_loader(fileBC)
    

    optimizer_u = torch.optim.Adam(list(model_u.parameters()) + [lambda_1 , lambda_2 , lambda_3 , lambda_4], lr=LR)
    optimizer_v = torch.optim.Adam(list(model_v.parameters()) + [lambda_1 , lambda_2 , lambda_3 , lambda_4], lr=LR)
    optimizer_w = torch.optim.Adam(list(model_w.parameters()) + [lambda_1 , lambda_2 , lambda_3 , lambda_4], lr=LR)
    optimizer_p = torch.optim.Adam(list(model_p.parameters()) + [lambda_1 , lambda_2 , lambda_3 , lambda_4], lr=LR)
    optimizer_T = torch.optim.Adam(list(model_T.parameters()) + [lambda_1 , lambda_2 , lambda_3 , lambda_4], lr=LR)



    # Hyperparameter tuning with Optuna
    def objective(trial):

        # Hyperparameters for tuning

        #lambda_1 = trial.suggest_float("lambda_1", 1e-3, 10)
        #lambda_2 = trial.suggest_float("lambda_2", 1e-3, 10)
        #lambda_3 = trial.suggest_float("lambda_3", 1e-3, 10)
        #lambda_4 = trial.suggest_float("lambda_4", 1e-3, 10)
        lambda_data = trial.suggest_float("lambda_data", 1e-3, 10)
        lambda_noisy = trial.suggest_float("lambda_noisy", 1e-3, 10)
        LR = trial.suggest_float("LR", 1e-5, 1e-1, log=True)



        optimizer_u = torch.optim.Adam(list(model_u.parameters()) + [lambda_1 , lambda_2 , lambda_3 , lambda_4], lr=LR)
        optimizer_v = torch.optim.Adam(list(model_v.parameters()) + [lambda_1 , lambda_2 , lambda_3 , lambda_4], lr=LR)
        optimizer_w = torch.optim.Adam(list(model_w.parameters()) + [lambda_1 , lambda_2 , lambda_3 , lambda_4], lr=LR)
        optimizer_p = torch.optim.Adam(list(model_p.parameters()) + [lambda_1 , lambda_2 , lambda_3 , lambda_4], lr=LR)
        optimizer_T = torch.optim.Adam(list(model_T.parameters()) + [lambda_1 , lambda_2 , lambda_3 , lambda_4], lr=LR)

        num_epochs = 500  #best value is 500
        for epoch in range(num_epochs):
            optimizer_u.zero_grad()
            optimizer_v.zero_grad()
            optimizer_w.zero_grad()
            optimizer_p.zero_grad()
            optimizer_T.zero_grad()

            loss , pdeLOSS , dataLOSS , noisyLOSS  = total_loss(model_u , model_v , model_w , model_p , model_T , X_c , Y_c , Z_c ,
                                                                 x_obj ,y_obj , z_obj , truth_obj , x_noisy , y_noisy , z_noisy , truth_noisy )


            loss.backward(retain_graph = True)

            optimizer_u.step()
            optimizer_v.step()
            optimizer_w.step()
            optimizer_p.step()
            optimizer_T.step()

            # Return the final loss for this trial
            return loss.item()

    num_trials= 50
    # Run the Optuna hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials = num_trials)  # Adjust n_trials for more thorough search

    # Extract the best lambda values
    best_params = study.best_params


    print("Optimized lambda_data:", best_params["lambda_data"])
    print("Optimized lambda_noisy:", best_params["lambda_noisy"])
    print("Optimized LR:", best_params["LR"])


    lambda_data = torch.tensor(best_params["lambda_data"] , device =device)
    lambda_noisy = torch.tensor(best_params["lambda_noisy"] , device = device)
    LR = best_params["LR"]



    loss_hist = []
    min_valid_loss = np.inf

    for epoch in range(nIter):

        running_loss = 0.0
        for X_inter_train , truth_inter_train in train_loader:
            x_train_inter = X_inter_train[: , 0].reshape(-1 , 1)
            y_train_inter = X_inter_train[: , 1].reshape(-1 , 1)
            z_train_inter = X_inter_train[: , 2].reshape(-1 , 1)

            model_u.train()
            model_v.train()
            model_w.train()
            model_p.train()
            model_T.train()

            optimizer_u.zero_grad()
            optimizer_v.zero_grad()
            optimizer_w.zero_grad()
            optimizer_p.zero_grad()
            optimizer_T.zero_grad()

            loss , pdeLOSS , dataLOSS , noisyLOSS  = total_loss(model_u , model_v , model_w , model_p , model_T , X_c , Y_c , Z_c ,
                                                                x_train_inter, y_train_inter , z_train_inter , truth_inter_train , x_noisy , y_noisy , z_noisy , truth_noisy )
            impose_boundary_conditions(model_u , model_v , model_w , model_p , model_T , fileBC)

            loss.backward(retain_graph = True)

            optimizer_u.step()
            optimizer_v.step()
            optimizer_w.step()
            optimizer_p.step()
            optimizer_T.step()

            loss_hist.append(loss.item())
            running_loss += loss.item() * x_train_inter.size()[0]

        train_loss = running_loss / len(train_loader.dataset)

        model_u.eval()
        model_v.eval()
        model_w.eval()
        model_p.eval()
        model_T.eval()
        valid_loss = 0.0

        for X_inter_val , truth_inter_val in valid_loader:
            x_inter_val = X_inter_val[: , 0].reshape(-1 , 1)
            y_inter_val = X_inter_val[: , 1].reshape(-1 , 1)
            z_inter_val = X_inter_val[: , 2].reshape(-1 , 1)


            loss , pdeLOSS , dataLOSS , noisyLOSS  = total_loss(model_u , model_v , model_w , model_p , model_T , X_c , Y_c , Z_c , 
                                                                x_inter_val , y_inter_val , z_inter_val , truth_inter_val , x_noisy , y_noisy , z_noisy ,truth_noisy )
            impose_boundary_conditions(model_u , model_v , model_w , model_p , model_T , fileBC)

            valid_loss += loss.item() * x_inter_val.size(0)

        valid_loss = valid_loss / len(valid_loader.dataset)
        if epoch % 200 == 0:

            print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss :.3e} \t\t Validation Loss: {valid_loss :.3e}')
            print(f'Epoch {epoch+1} \t\t PDE LOSS: {pdeLOSS.item() :.3e} \t\t DATA LOSS: {dataLOSS.item() :.3e} \t\t NOISE LOSS: {noisyLOSS.item() :.3e}')
            print(f'lambda1: {lambda_1.item():.3e}\t\t lambda2: {lambda_2.item():.3e} \t\t lambda3:{lambda_3.item():.3e} \t\t lambda4: {lambda_4.item():.3e}')

        if min_valid_loss > valid_loss:
            min_valid_loss = valid_loss

            if min_valid_loss < 1e-2 :
              print(f'Validation Loss Decreased({min_valid_loss:.3e}--->{valid_loss:.3e}) \t Saving The Model')
              # Saving State Dict
              torch.save(model_u.state_dict(), save_to_path + 'saved_model_u.pth')
              torch.save(model_v.state_dict(), save_to_path +'saved_model_v.pth')
              torch.save(model_w.state_dict(), save_to_path +'saved_model_w.pth')
              torch.save(model_p.state_dict(), save_to_path +'saved_model_p.pth')
              torch.save(model_T.state_dict(), save_to_path +'saved_model_T.pth')

    plt.plot(loss_hist)
    plt.title("Loss history")
    plt.show()

lambda_1 = 1
lambda_2 = 1
lambda_3 = 1
lambda_4 = 1
lambda_data = 1
lambda_noisy = 1

train(model_u , model_v , model_w , model_p , model_T , file_data , fileBC , 5000)

x , y , z ,truth = simple_data_loader(file_data)

u_star = model_u(torch.cat((x , y , z) , dim = 1))

T_star = model_T(torch.cat((x , y , z) , dim = 1))
p_star = model_p(torch.cat((x , y , z) , dim = 1))
x.detach().numpy()
y.detach().numpy()
u_star.detach().numpy()
T_star.detach().numpy()

plot_solution(x.detach().numpy() , y.detach().numpy() , u_star.detach().numpy() , "PINN")
plot_solution(x.detach().numpy() , y.detach().numpy() , truth[: , 0].detach().numpy() , "Truth")


plt.figure()
plt.plot(T_star.detach().numpy() , label = "PINN")
plt.plot(truth[: , 3].detach().numpy() , label = "Truth")
plt.title("Temperature")
plt.legend()
plt.ylim(-1 , 1)

plt.figure()
plt.plot(u_star.detach().numpy() , label = "PINN")
plt.plot(truth[: , 0].detach().numpy() , label = "Truth")
plt.legend()
plt.title("velocity")
plt.ylim(-1 , 1)

