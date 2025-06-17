import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
import time
import os
import sys

from ABM_package import *

# --- Get rp from command-line ---
if len(sys.argv) < 3:
    print("Usage: python script_name.py <rp_value> <n_sims>")
    sys.exit(1)

rp = float(sys.argv[1])
n_sims = int(sys.argv[2])
rd = rp / 2
rm = 1.0
scale_factor = 2
initial_density = 0.3
num_t = 100
T_end = 15.0
#folder_path = "../data"
folder_path = "../../../BINNs/Data"

# Create a label suffix to use consistently
suffix = f"rp_{rp}_rd_{rd}_rm_{rm}_scale_{scale_factor}_dens_{initial_density}"

# --- Run ABM simulations ---
for i in range(n_sims):
    A_out, t_out, plot_list, interp_profiles = BDM_ABM(rp, rd, rm, T_end=T_end, scale=scale_factor, initial_density = initial_density)
    save_data = {'variables': [t_out, A_out, interp_profiles]}
    #filename = f"{folder_path}/modified_logistic_ABM_sim_{suffix}_{i}.npy"
    filename = f"{folder_path}/modified_logistic_ABM_BINNs_sim_{suffix}_{i}.npy"
    print(filename)
    np.save(filename, save_data)

# --- Load and average simulations ---
A_list = []
interp_list = []

for i in range(n_sims):
    #filename = f"{folder_path}/modified_logistic_ABM_sim_{suffix}_{i}.npy"
    filename = f"{folder_path}/modified_logistic_ABM_BINNs_sim_{suffix}_{i}.npy"
    mat = np.load(filename, allow_pickle=True, encoding='latin1').item()
    A_out = mat['variables'][1]
    A_list.append(A_out)
    interp_profiles_out = mat['variables'][2]
    interp_list.append(interp_profiles_out)

A_matrix = np.vstack(A_list)
avg_A = np.mean(A_matrix, axis=0) / (120 * 120)

interp_array = np.stack(interp_list, axis=0)   # shape (n_sims, 100, 120)
avg_interp = np.mean(interp_array, axis=0)     # shape (100, 120)

t_out = mat['variables'][0]
ABM_t = compute_derivative(t_out, avg_A)

# --- Save the averaged results ---
#save_data = {'variables': [t_out, avg_A, ABM_t]}
#filename = f"{folder_path}/modified_logistic_ABM_sim_{suffix}_complete.npy"
#np.save(filename, save_data)

### New saves for work with BINNs
# Create meshgrid
X, T = np.meshgrid(np.arange(avg_interp.shape[1]), t_out)

# Create dictionary for saving data
save_data_BINNs = {
    'dens_' + str(initial_density): {
        'X': X,
        'T': T,
        'U': avg_interp
    }
}

filename = f"{folder_path}/modified_logistic_ABM_BINNs_sim_{suffix}_complete.npy"
np.save(filename, save_data_BINNs)

