
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os
import matplotlib




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

save_to_path = 'E:/FOAM_PINN/cavHeat/CaseE/Main_results/'

model_u_name = 'Main_results/saved_model_u.pth'
model_v_name = 'Main_results/saved_model_v.pth'
model_w_name = 'Main_results/saved_model_w.pth'
model_p_name =  'Main_results/saved_model_p.pth'
model_T_name =  'Main_results/saved_model_T.pth'


file_data = r"caseE_Data.csv"
#file_data = r"data_2D_Lamin.csv"
fileBC = r"caseE_BC.csv"
fileTest = r"test_cloud1_U_caseE.csv"



def simple_data_loader(file_name):
    df =  normal_inputs(pd.read_csv(file_name))
    x = torch.tensor(df[['x']].values, dtype=torch.float32 , requires_grad = True).reshape(-1 , 1)
    y = torch.tensor(df[['y']].values, dtype=torch.float32 , requires_grad = True).reshape(-1 , 1)
    z = torch.tensor(df[['z']].values, dtype=torch.float32 , requires_grad = True).reshape(-1 , 1)
    truth = torch.tensor(df[['T' ,'u' ,'v' ,'w' , 'p']].values, dtype=torch.float32 , requires_grad = True).reshape(-1 , 5)

    return x , y , z , truth

def load_model(file_pth , model , layers):
    init_model = model(layers)
    state_dict = torch.load(file_pth , map_location = device)
    #state_dict = torch.load(file_pth
    init_model.load_state_dict(state_dict)
    return init_model #LOADED model



def normal_inputs(df): #df is a dataframe
    normal_df = (2 * (df - df.min()) / (df.max() - df.min() + 1e-10 )) - 1
    return normal_df


import matplotlib.pyplot as plt
import numpy as np

def plot_3d_pinn_results(x, y, values, title, xlabel="X", ylabel="Y", colorbar_label="", cmap="viridis"):
    """
    Plots a 2D pcolor plot of 3D PINN-predicted results (e.g., velocity components, pressure, or temperature).

    Parameters:
        x (array-like): x-coordinates of the scattered data.
        y (array-like): y-coordinates of the scattered data.
        values (array-like): Values to plot (e.g., velocity, pressure, or temperature).
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        colorbar_label (str): Label for the colorbar.
        cmap (str): Colormap for the plot.
    """
    # Define the grid for interpolation
    xi = np.linspace(np.min(x), np.max(x), 210)
    yi = np.linspace(np.min(y), np.max(y), 210)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate data onto the grid
    Z = griddata((x, y), values, (X, Y), method='linear')

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.pcolor(X, Y, Z,  vmin=-1, vmax=1 ,  cmap=cmap, shading='auto')
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_to_path + title +".png")
    plt.show()
    
    #ax.set_facecolor('white')         # Set the plot area background to white





##############################################################################
model_u = load_model(model_u_name , PINN_u , layers)
model_v = load_model(model_v_name , PINN_v , layers)
model_w = load_model(model_w_name , PINN_w , layers)
model_p = load_model(model_p_name , PINN_p , layers)
model_T = load_model(model_T_name , PINN_T , layers)


x , y , z , truth = simple_data_loader(file_data)


u_star = model_u(torch.cat((x , y , z) , dim = 1))
v_star = model_v(torch.cat((x , y , z) , dim = 1))

Tt_star = model_T(torch.cat((x , y ,z ) , dim = 1))
p_star = model_p(torch.cat((x , y ,z ) , dim = 1))


u_star.detach().numpy()
Tt_star.detach().numpy()



x = x.detach().numpy()
y = y.detach().numpy()
z = z.detach().numpy()
truth.detach().numpy()

lb = [min(x) , min(y) , min(z)]
ub = [max(x) , max(y) , max(z)]

nn = 220

X, Y = np.meshgrid(x , y)
points = np.column_stack((x.flatten(), y.flatten()))

U_star = griddata(points, u_star.flatten().detach().numpy(), (X, Y), method='cubic')
P_star = griddata(points, p_star.flatten().detach().numpy(), (X, Y), method='cubic')
T_star = griddata(points, Tt_star.flatten().detach().numpy(), (X, Y), method='cubic')
u_truth = griddata(points, truth[: , 1].flatten().detach().numpy(), (X, Y), method='cubic')
p_truth = griddata(points, truth[: , 4].flatten().detach().numpy(), (X, Y), method='cubic')
T_truth = griddata(points, truth[: , 0].flatten().detach().numpy(), (X, Y), method='cubic')


plt.figure(dpi = 110)
plt.plot(u_star.detach().numpy() , marker = "*" , label= "PINN")
plt.plot(truth[: , 1].detach().numpy() , label= "truth")
plt.title("velocity u")
plt.ylabel("u /Uref")
plt.savefig(save_to_path + "velocity_comp.png")
plt.legend()

plt.figure(dpi = 110)
plt.plot(p_star.detach().numpy()  ,marker = "*", label= "PINN")
plt.plot(truth[: , 4].detach().numpy() , label= "truth")
plt.ylim(-1 , 1)
leg = plt.legend()
plt.setp(leg.get_texts(), color='black')  # Set legend text color to black

plot_3d_pinn_results(x.flatten(), y.flatten(), u_star.flatten().detach().numpy(), "velocity PINN", xlabel="X", ylabel="Y", colorbar_label="", cmap="jet")
plot_3d_pinn_results(x.flatten(), y.flatten(), truth[: ,1].flatten().detach().numpy(), "velocity truth", xlabel="X", ylabel="Y", colorbar_label="", cmap="jet")
