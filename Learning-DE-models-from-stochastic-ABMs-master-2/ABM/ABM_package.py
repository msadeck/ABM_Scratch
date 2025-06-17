import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import integrate
import matplotlib as mpl
from scipy import interpolate
import time
from scipy.sparse import lil_matrix
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def compute_derivative(t, y):
    """
    Compute dy/dt using finite differences:
    - Forward difference at the first point
    - Centered differences for interior points
    - Backward difference at the last point

    Parameters:
    - t: time array (1D)
    - y: corresponding y-values (1D), e.g., ABM output

    Returns:
    - dydt: array of derivatives (same length as t)
    """
    t = np.asarray(t).flatten()
    y = np.asarray(y).flatten()
    n = len(t)
    dydt = np.zeros(n)

    # Forward difference for the first point
    dydt[0] = (y[1] - y[0]) / (t[1] - t[0])

    # Centered differences for internal points
    for i in range(1, n - 1):
        dydt[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])

    # Backward difference for the last point
    dydt[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])

    return dydt
    
def SIR_ODE(t,y,q,desc):

    dydt = np.zeros((3,))

    dydt[0] = -q[0]*y[0]*y[1]
    dydt[1] = -q[1]*y[1] + q[0]*y[0]*y[1]
    dydt[2] = q[1]*y[1]
    
    return dydt

def ODE_sim(q,RHS,t,IC,description=None):
    
    #grids for numerical integration
    t_sim = np.linspace(t[0],t[-1],10000)
    
    #Initial condition
    y0 = IC
        
    #indices for integration steps to write to file for
    for tp in t:
        tp_ind = np.abs(tp-t_sim).argmin()
        if tp == t[0]:
            t_sim_write_ind = np.array(tp_ind)
        else:
            t_sim_write_ind = np.hstack((t_sim_write_ind,tp_ind))

    #make RHS a function of t,y
    def RHS_ty(t,y):
         return RHS(t,y,q,description)
            
    #initialize solution
    y = np.zeros((len(y0),len(t)))   
    y[:,0] = IC
    write_count = 1

    #integrate
    r = integrate.ode(RHS_ty).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y0, t[0])   # initial values
    for i in range(1, t_sim.size):
        #write to y during write indices
        if np.any(i==t_sim_write_ind):
            y[:,write_count] = r.integrate(t_sim[i])
            write_count+=1
        else:
            #otherwise just integrate
            r.integrate(t_sim[i]) # get one more value, add it to the array
        if not r.successful():
            print("integration failed for parameter ")
            print(q)
            return 1e6*np.ones(y.shape)

    return y

def local_neighborhood_mask(A_shape, loc, distance=1):
    '''
    Create a sparse matrix with 1s in the neighborhood of a point (loc),
    and 0s elsewhere.

    Parameters:
    - A_shape: tuple (rows, cols) of the matrix
    - loc: tuple (x, y) center of the neighborhood
    - distance: neighborhood distance (default = 1)

    Returns:
    - mask: sparse lil_matrix with 1s in the neighborhood
    '''
    rows, cols = A_shape
    x, y = loc

    # Create an empty sparse matrix
    mask = lil_matrix((rows, cols), dtype=int)

    # Compute neighborhood bounds
    for i in range(max(0, x - distance), min(rows, x + distance + 1)):
        for j in range(max(0, y - distance), min(cols, y + distance + 1)):
            mask[i, j] = 1

    return mask


import numpy as np
from scipy import interpolate
import numpy as np
from scipy import interpolate

def BDM_ABM(rp, rd, rm, scale, T_end, initial_density):
    # Define the size of the lattice (n x n)
    n = 120

    # Initialize the lattice with all zeros (empty cells)
    A = np.zeros((n**2,))

    # Set initial proportion of occupied sites
    A0 = initial_density
    A_num = int(np.ceil(A0 * len(A)))

    # Create a scratch (empty vertical strip in the center of the grid)
    radius = n // 6
    scratch = set(range(n//2 - radius, n//2 + radius))
    start_positions = [(i, j) for i in range(n) for j in range(n) if j not in scratch]

    # Randomly select initial occupied positions avoiding the scratch
    chosen_indices = np.random.choice(len(start_positions), size=A_num, replace=False)
    for idx in chosen_indices:
        i, j = start_positions[idx]
        A[i * n + j] = 1

    # Reshape to 2D lattice and count initial number of agents
    A = A.reshape(n, n)
    A_num = np.sum(A == 1)

    # Calculate final simulation time (non-dimensionalized)
    T_final = T_end / (rp - rd)

    # Initialize simulation time and tracking lists
    t = 0
    t_list = [t]
    A_list = [A_num]
    plot_list = [np.copy(A)]
    density_profiles = [np.sum(A == 1, axis=0) / A.shape[0]]  # initial density profile
    image_count = 1

    # Initialize progress bar for user feedback
    pbar = tqdm(total=50, desc="Running ABM", leave=True)

    # Main simulation loop
    while t_list[-1] < T_final:
        # Find all occupied locations and select a random agent
        agent_loc = np.where(A != 0)
        agent_ind = np.random.permutation(len(agent_loc[0]))[0]
        loc = (agent_loc[0][agent_ind], agent_loc[1][agent_ind])

        # Get the local neighborhood mask and compute number of neighbors
        mask = local_neighborhood_mask((n, n), loc, distance=1)
        neigh_den = mask.multiply(A)
        result = np.sum(neigh_den == 1)

        # Update movement/proliferation rates based on local density
        if result >= 3:
            rmf = scale * rm
            rpf = (1 / scale) * rp
        else:
            rmf = (1 / scale) * rm
            rpf = scale * rp

        # Total rate of all possible events
        a = rmf * A_num + rpf * A_num + rd * A_num

        # Time step
        tau = -np.log(np.random.uniform()) / a
        t += tau

        # Randomly choose which event occurs
        Action = a * np.random.uniform()

        # Movement event
        if Action <= rmf * A_num:
            agent_state = A[loc]
            dir_select = np.random.randint(1, 5)
            if dir_select == 1 and loc[0] < n - 1 and A[loc[0] + 1, loc[1]] == 0:
                A[loc[0] + 1, loc[1]] = agent_state
                A[loc] = 0
            elif dir_select == 2 and loc[0] > 0 and A[loc[0] - 1, loc[1]] == 0:
                A[loc[0] - 1, loc[1]] = agent_state
                A[loc] = 0
            elif dir_select == 3 and loc[1] < n - 1 and A[loc[0], loc[1] + 1] == 0:
                A[loc[0], loc[1] + 1] = agent_state
                A[loc] = 0
            elif dir_select == 4 and loc[1] > 0 and A[loc[0], loc[1] - 1] == 0:
                A[loc[0], loc[1] - 1] = agent_state
                A[loc] = 0

        # Proliferation event
        elif Action <= rmf * A_num + rpf * A_num:
            dir_select = np.random.randint(1, 5)
            if dir_select == 1 and loc[0] < n - 1 and A[loc[0] + 1, loc[1]] == 0:
                A[loc[0] + 1, loc[1]] = 1
            elif dir_select == 2 and loc[0] > 0 and A[loc[0] - 1, loc[1]] == 0:
                A[loc[0] - 1, loc[1]] = 1
            elif dir_select == 3 and loc[1] < n - 1 and A[loc[0], loc[1] + 1] == 0:
                A[loc[0], loc[1] + 1] = 1
            elif dir_select == 4 and loc[1] > 0 and A[loc[0], loc[1] - 1] == 0:
                A[loc[0], loc[1] - 1] = 1

        # Death event
        else:
            A[loc] = 0

        # Update counts and tracking
        A_num = np.sum(A == 1)
        density_profile = np.sum(A == 1, axis=0) / A.shape[0]  # average across rows
        t_list.append(t)
        A_list.append(A_num)
        density_profiles.append(density_profile)

        # Save snapshot if time passed threshold
        if len(t_list) == 2:
            plot_list.append(np.copy(A))
            image_count += 1
        elif t_list[-2] < image_count * T_final / 50 and t_list[-1] >= image_count * T_final / 50:
            plot_list.append(np.copy(A))
            image_count += 1
            pbar.update(1)  # progress bar

    pbar.close()

    # Interpolate agent count over uniform time grid
    t_out = np.linspace(0, T_final, 100)
    f = interpolate.interp1d(t_list, A_list)
    A_out = f(t_out)

    # Convert list of density profiles to NumPy array
    density_profiles = np.array(density_profiles)

    # Interpolate each column of the density profile
    interp_profiles = np.array([
        np.interp(t_out, t_list, density_profiles[:, j])
        for j in range(density_profiles.shape[1])
    ]).T  # shape: (len(t_out), width)

    # Return interpolated agent count, time vector, plot snapshots, and interpolated density
    return A_out, t_out, plot_list, interp_profiles

def plot_density_3d(t_out, interp_profiles):
    """
    Plot a 3D surface of density over time and position.
    X-axis: Position (space)
    Y-axis: Time
    Z-axis: Density
    """
    # Create meshgrid for surface plot
    time_grid, pos_grid = np.meshgrid(t_out, np.arange(interp_profiles.shape[1]), indexing='ij')

    # Set up 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(pos_grid, time_grid, interp_profiles, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Position')
    ax.set_ylabel('Time')
    ax.set_zlabel('Density')
    ax.set_title('Density Over Time and Position')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Density')

    plt.tight_layout()
    plt.show()

def SIR_ABM(ri,rr,rm,T_end=5.0):

    #number of lattice sites
    n = 40

    A = np.zeros((n**2,))

    #initial proportions of susceptible, infected, and recovered agents
    s0 = 0.49
    i0 = 0.01
    r0 = 0.0

    #randomly place susceptible (1), infected (2), and recovered (3) agents
    s_num = np.int(np.ceil(s0*len(A)))
    i_num = np.int(np.ceil(i0*len(A)))
    r_num = np.int(np.ceil(r0*len(A)))
    A[:s_num] = 1
    A[s_num:s_num+i_num] = 2
    A[s_num+i_num:s_num+i_num+r_num] = 3
    #shuffle up
    A = A[np.random.permutation(n**2)]
    #make square
    A = A.reshape(n,n)

    #count number of susceptible, infected, and recovered agents.
    S_num = np.sum(A==1)
    I_num = np.sum(A==2)
    R_num = np.sum(A==3)
    total_num = S_num + I_num + R_num

    #Convert agent counts to proportions
    S = np.float(S_num)/np.float(total_num)
    I = np.float(I_num)/np.float(total_num)
    R = np.float(R_num)/np.float(total_num)

    #nondimensionalized time
    T_final = T_end/rr

    #initialize time
    t = 0

    #track time, agent proportions, and snapshots of ABM in these lists
    t_list = [t]
    S_list = [S]
    I_list = [I]
    R_list = [R]
    A_list = [A]
    #number of snapshots saved
    image_count = 1


    while t_list[-1] < T_final:

        a = rm*(S_num+I_num+R_num) + ri*I_num + rr*I_num
        tau = -np.log(np.random.uniform())/a
        t += tau

        Action = a*np.random.uniform()

        if Action <= rm*(S_num+I_num+R_num):
            #any agent movement
            
            # Select Random agent
            agent_loc = np.where(A!=0)
            agent_ind = np.random.permutation(len(agent_loc[0]))[0]
            loc = (agent_loc[0][agent_ind],agent_loc[1][agent_ind])
            
            #determine status
            agent_state = A[loc]

            ### Determine direction
            dir_select = np.ceil(np.random.uniform(high=4.0))

            #move right
            if dir_select == 1 and loc[0]<n-1:
                if A[(loc[0]+1,loc[1])] == 0:
                    A[(loc[0]+1,loc[1])] = agent_state
                    A[loc] = 0
            #move left
            elif dir_select == 2 and loc[0]>0:
                if A[(loc[0]-1,loc[1])] == 0:
                    A[(loc[0]-1,loc[1])] = agent_state
                    A[loc] = 0
            #move up
            elif dir_select == 3 and loc[1]<n-1:
                if A[(loc[0],loc[1]+1)] == 0:
                    A[(loc[0],loc[1]+1)] = agent_state
                    A[loc] = 0

            #move down                    
            elif dir_select == 4 and loc[1]>0:
                if A[(loc[0],loc[1]-1)] == 0:
                    A[(loc[0],loc[1]-1)] = agent_state
                    A[loc] = 0

        elif (rm*(S_num+I_num+R_num) < Action) and (Action <= rm*(S_num+I_num+R_num) + ri*I_num):
            #infection event
            
            ### Select Random infected agent
            I_ind = np.random.permutation(I_num)[0]
            loc = (np.where(A==2)[0][I_ind],np.where(A==2)[1][I_ind])

            ### Determine direction
            dir_select = np.ceil(np.random.uniform(high=4.0))

            #infect right
            if dir_select == 1 and loc[0]<n-1:
                if A[(loc[0]+1,loc[1])] == 1:
                    A[(loc[0]+1,loc[1])] = 2

            #infect left
            elif dir_select == 2 and loc[0]>0:
                if A[(loc[0]-1,loc[1])] == 1:
                    A[(loc[0]-1,loc[1])] = 2

            #infect up        
            elif dir_select == 3 and loc[1]<n-1:
                if A[(loc[0],loc[1]+1)] == 1:
                    A[(loc[0],loc[1]+1)] = 2

            #infect down
            elif dir_select == 4 and loc[1]>0:
                if A[(loc[0],loc[1]-1)] == 1:
                    A[(loc[0],loc[1]-1)] = 2

        elif (rm*(S_num+I_num+R_num) + ri*I_num < Action) and (Action <= rm*(S_num+I_num+R_num) + ri*I_num + rr*I_num):
            #Recovery event
            
            ### Select Random I
            I_ind = np.random.permutation(I_num)[0]
            loc = (np.where(A==2)[0][I_ind],np.where(A==2)[1][I_ind])
            A[loc] = 3

        #count number of susceptible, infected, recovered agents
        S_num = np.sum(A==1)
        I_num = np.sum(A==2)
        R_num = np.sum(A==3)
        #convert counts to proportions
        S = np.float(S_num)/np.float(total_num)
        I = np.float(I_num)/np.float(total_num)
        R = np.float(R_num)/np.float(total_num)

        #append information to lists
        t_list.append(t)
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)

        #sometimes save ABM snapshot
        if t_list[-2] < image_count*T_final/20 and t_list[-1] >= image_count*T_final/20:
            A_list.append(np.copy(A))
            image_count+=1

    #interpolation to equispace grid
    t_out = np.linspace(0,T_final,100)

    f = interpolate.interp1d(t_list,S_list)
    S_out = f(t_out)

    f = interpolate.interp1d(t_list,I_list)
    I_out = f(t_out)

    f = interpolate.interp1d(t_list,R_list)
    R_out = f(t_out)


    return S_out,I_out,R_out,t_out,A_list,total_num


def ABM_depict(A_list):
    cmaplist = [(1.0,1.0,1.0,1.0),(0.0,0.0,1.0,1.0),(0.0,1.0,0.0,1.0),(1.0,0.0,0.0,1.0)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, N = 4)

    fig = plt.figure()
    ax = fig.add_subplot(1,3,1)
    ax.matshow(A_list[6],cmap=cmap)
    ax = fig.add_subplot(1,3,2)
    ax.matshow(A_list[13],cmap=cmap)
    ax = fig.add_subplot(1,3,3)
    im = ax.matshow(A_list[-1],cmap=cmap)
    fig.colorbar(im,ax=ax)