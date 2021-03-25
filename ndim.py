import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
from itertools import product
from copy import copy, deepcopy

# Setting plotting defaults

# Make legend text small
plt.rcParams['legend.fontsize'] = 'small' 
# Shrink axes labels a bit 
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['xtick.labelsize'] = 'small'
plt.rcParams['ytick.labelsize'] = 'small'
# Set limits on when scientific notation is used
plt.rcParams['axes.formatter.limits'] = [-2, 3] 
# Use LaTeX to format axes labels and numbers
plt.rcParams['axes.formatter.use_mathtext'] = True
# Get rid of spines on top and bottom
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
# Ticks point in  
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# Change DPI of figure images
plt.rcParams['figure.dpi'] = 150

# Simulation functions 
DT = 0.1

# Function implimenting the midpoint method aka RK2
def find_next_point_midpoint_nD(s, dsdt, dt=DT):
    s_test = s + dsdt(*s)*dt
    s_next = s + dsdt(*((s + s_test)/2.))*dt
    return(s_next)

# Simulation function
def simulate_nD(s_0, t_f, dsdt, simulation_function=find_next_point_midpoint_nD, dt=DT, verbose=False):
    S_solution = [s_0]
    T = np.arange(s_0[-1], t_f + dt, dt)
    for t in T[1:]:
        s_previous = S_solution[-1]
        s_next = simulation_function(s_previous, dsdt, dt=dt)
        if verbose:
            print(s_next)
        S_solution.append(s_next)
    S_solution = np.array(S_solution)
    return(S_solution)

# Function to find peaks
from scipy.signal import find_peaks

def AP_times(V, T, dt=DT):
    return(T[find_peaks(V, height=0, width=int(1./dt))[0]])

# Functions specific to 2D systems (can depend on time)

## Function to find the critical points, (do not change)
def find_2D_critical_points(dxdt, dydt, t=0, epsilon=0.0001, res_c=0.1, xlim=(-3, 3), ylim=(-3, 3)):
    x_min, x_max = xlim
    y_min, y_max = ylim
    
    ## Helper function needed for fsolve
    def DDT(arg):
        x, y = arg
        return(dxdt(x, y, t), dydt(x, y, t))

    ## Creating a grid to search for critical points
    x = np.arange(*xlim, res_c)
    y = np.arange(*ylim, res_c)
    X, Y = np.meshgrid(x, y)

    ## Finding the critical points
    cp_x = []
    cp_y = []
    for x_0, y_0 in product(x, y):
        x_c, y_c = fsolve(DDT, (x_0, y_0), factor=1.0)
        
        # Checking if its in range
        if (x_c >= x_min) & (x_c <= x_max) & (y_c >= y_min) & (y_c <= y_max):
            if len(cp_x) > 0:
                # Check to make sure the point is not already included
                is_included = np.any(np.abs(np.array(cp_x) - x_c) < epsilon) & np.any(np.abs(np.array(cp_y) - y_c) < epsilon)
                if not is_included:
                    cp_x.append(x_c)
                    cp_y.append(y_c)
            else:
                # first point
                cp_x.append(x_c)
                cp_y.append(y_c)
    
    cp_x = np.array(cp_x)
    cp_y = np.array(cp_y)
    
    # Checking which ones actually worked
    actually_worked = np.isclose(dxdt(cp_x, cp_y, t), np.zeros(len(cp_x))) * np.isclose(dydt(cp_x, cp_y, t), np.zeros(len(cp_x)))
    cp_x = cp_x[actually_worked]
    cp_y = cp_y[actually_worked]

    ## Sorting the lists of critical points
    sort_x = np.argsort(cp_x)
    cp_x = cp_x[sort_x]
    cp_y = cp_y[sort_x]
    sort_y = np.argsort(cp_y)
    cp_x = cp_x[sort_y]
    cp_y = cp_y[sort_y]

    print("Critical points found at:")            
    print([i for i in zip(cp_x, cp_y)])
    print()
    return cp_x, cp_y

# Function for plotting the phase diagram (do not change)
def plot_2D_phase_space(dxdt, dydt, t=0, xlim=(-3, 3), ylim=(-3, 3), res=0.1, q_scale=1.0, x_label="x", y_label="y", ax=None, figsize=(3,3)):
    if ax == None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1,aspect='equal')

    ## Creating the Grid for plotting
    x = np.arange(*xlim, res)
    y = np.arange(*ylim, res)
    X, Y = np.meshgrid(x, y)

    ## Caculating the change vectors
    dxdt_q = dxdt(X, Y, t)
    dydt_q = dydt(X, Y, t)
    max_ddt = np.max(np.sqrt(dxdt_q**2 + dydt_q**2))

    ## Finding critical points 
    cp_x, cp_y = find_2D_critical_points(dxdt, dydt, xlim=xlim, ylim=ylim)

    ## Plotting quivers and critical points
    ax.axhline(0, color='.8')
    ax.axvline(0, color='.8')
    ax.set_xlim(xlim)
    ax.scatter(cp_x, cp_y, s=100, color='blue', marker='o', facecolor='none')
    ax.quiver(x, y, dxdt_q, dydt_q, scale_units='width', scale=max_ddt/(.1 * q_scale), color='blue')
    ax.set_title("2D phase portrait")
    ax.set_ylim(ylim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return(ax)
    
def plot_2D_solutions(T_f, X_0s, Y_0s, dxdt, dydt, simulation_functions, colors=['.k'], dt=DT, xlim=(-3, 3), ylim=(-3, 3), res=0.1, q_scale=1.0, x_label="x", y_label="y", figsize=(9,3)):
    fig, (ax, ax_x, ax_y) = plt.subplots(figsize=figsize, ncols=3)
    
    plot_2D_phase_space(dxdt, dydt, xlim=xlim, ylim=ylim, res=res, q_scale=q_scale, x_label=x_label, y_label=y_label, ax=ax)
    ax.set_aspect(aspect='equal')
    
    # Wrapping dynamics into one function
    def dSdt(x, y, t):
        return(np.array([dxdt(x,y, t), dydt(x, y, t), 1.]))
    
    X_solutions = []
    Y_solutions = []
    for simulation_function, (X_0, Y_0) in product(simulation_functions, zip(X_0s, Y_0s)):
        S = simulate_nD((X_0, Y_0, 0.), T_f, dSdt, simulation_function, dt=dt)
        X_solutions.append(S[:,0])
        Y_solutions.append(S[:,1])
        T = S[:,2]
        
    if len(colors) != len(X_solutions):
        colors = len(X_solutions) * colors

    for X_solution, Y_solution, color in zip(X_solutions, Y_solutions, colors):
        ax.plot(X_solution, Y_solution, color=color)
        ax.scatter(X_solution[0], Y_solution[0], color=color, zorder=10, s=10.0)

        ax_x.set_title("{0} vs. time".format(x_label))
        ax_x.plot(T, X_solution, color=color)
        ax_x.axhline(0, color='.8', zorder=-1)
        ax_x.scatter(T[0], X_solution[0], color=color, zorder=10, s=10.0)
        ax_x.set_ylabel(x_label)
        ax_x.set_xlabel("Time")
        ax_x.set_ylim(xlim)

        ax_y.set_title("{0} vs. time".format(y_label))
        ax_y.plot(T, Y_solution, color=color)
        ax_y.axhline(0, color='.8', zorder=-1)
        ax_y.scatter(T[0], Y_solution[0], color=color, zorder=10, s=10.0)
        ax_y.set_ylabel(y_label)
        ax_y.set_xlabel("Time")
        ax_y.set_ylim(ylim)
        
    return(ax, ax_x, ax_y)
    
# Some functions to "linearize" a system near crtical points and find eigenvalues and vectors there 

# Function to "linearize" non-linear systems near a point (do not change)
def linearize_2D(x, y, dxdt, dydt, D=0.000001, t=0):
    dx_func = lambda x, y, t=t: dxdt(x, y, t)
    dy_func = lambda x, y, t=t: dydt(x, y, t)
    L = np.array(
        [[(dx_func(x+D,y) - dx_func(x-D,y, t=0)) / (2*D), (dx_func(x,y+D, t) - dx_func(x,y-D)) / (2*D)], 
        [(dy_func(x+D,y) - dy_func(x-D,y)) / (2*D), (dy_func(x,y+D) - dy_func(x,y-D)) / (2*D)]]
    )
    return(L)

# Function to "linearize" non-linear system near all critical points
def linearize_all_2D_critical_points(dxdt, dydt, t=0, xlim=(-3, 3), ylim=(-3, 3)):
    cp_x, cp_y = find_2D_critical_points(dxdt, dydt, t=t, xlim=xlim, ylim=ylim)
    Ls = []
    for i, (x, y) in enumerate(zip(cp_x, cp_y)):
        L = linearize_2D(x, y, dxdt, dydt)
        Ls.append(L)
        print(f" At critical point #{i}, ({x}, {y}), the functions are approximated by \n {L}\n")
    return Ls, cp_x, cp_y

# Function to find and print eigenvalues and eigenvectors of a matrix (do not change)
def find_eigenvalues_and_eigenvectors(M):
    vals, mat = eig(M)
    for i in range(len(vals)):
        if np.imag(vals[i]) == 0.0:
            vals = np.array(vals, dtype='float')
        print(f"\tEigenvalue {i+1} is {vals[i]:.3}, with vector ({mat[0,i]:.3}, {mat[1,i]:.3})")
    return vals, mat

# A function to find and print eigenvalues and eigenvectors at all crtiical points (do not change)
def find_all_2D_critcal_eigenvalues_and_eigenvectors(dxdt, dydt, t=0, xlim=(-3, 3), ylim=(-3, 3)):
    Ls, cp_x, cp_y = linearize_all_2D_critical_points(dxdt, dydt, t=t)
    vals = []
    mats = []
    for i, (L, x, y) in enumerate(zip(Ls, cp_x, cp_y)):
        print(f"At critical point #{i}, ({x}, {y}),")
        val, mat = find_eigenvalues_and_eigenvectors(Ls[i])
        print("")
        vals.append(val)
        mats.append(mat)
    return vals, mats, Ls, cp_x, cp_y
        