
import numpy as np
from scipy.optimize import fsolve
from scipy.misc import derivative
import matplotlib as mpl
import matplotlib.pyplot as plt

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

X_MIN = -5
X_MAX = 5
X_LABEL = "Voltage"
DT = .01

# function to find critical points
def find_1D_critical_points(dXdT,  xlim=(X_MIN, X_MAX), res_c=0.001, epsilon=0.0001):
    x_min, x_max = xlim
    
    # Define starting positions
    X_0s = np.arange(x_min, x_max, res_c)
    critical_points = []
    # Loop over possible starting positions
    for X_0 in X_0s:
        # Find critical point from starting position
        X_c = fsolve(dXdT, X_0, factor=0.1)[0]
        
        # Checking if its in range
        if (X_c >= x_min) & (X_c <= x_max):
            if len(critical_points) > 0:
                # Check to make sure the point is not already included
                is_included = np.any(np.abs(np.array(critical_points) - X_c) < epsilon)
                if not is_included:
                    critical_points.append(X_c)
            else:
                # first point
                critical_points.append(X_c)
    critical_points = np.array(critical_points)
    
    # Checking which ones actually worked
    
    actually_worked = np.isclose(dXdT(critical_points), np.zeros(len(critical_points)))
    critical_points = critical_points[actually_worked]
    critical_points 
    
    return(critical_points)

# function to find sign of function at points
def find_sign_of_derivative(func, Xs):
    return(np.sign([derivative(func, X) for X in Xs]))

# function to plot the phase portrait
def plot_1D_phase_space(dXdT, xlim=(X_MIN, X_MAX), x_label=X_LABEL, res_c=0.001, epsilon=0.0001, ax=None):
    if ax == None:
        fig, ax = plt.subplots()
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['left'].set_position(('data',0))

    
    # Plotting derivative
    X = np.arange(xlim[0], xlim[1], res_c)
    Y = dXdT(X)
    ax.plot(X, dXdT(X), color='blue')

    # Setting labels and moving them
    #ax.set_xlabel(x_label)
    #ax.set_ylabel(f"d({x_label})/dT")
    X_max = np.max(X)
    Y_max = np.max(Y)
    ax.text(X_max, 0, x_label, va="bottom", ha="right")
    ax.text(0, Y_max, f"d({x_label})/dT", rotation="vertical", ha="left", va="top")
    
    # Finding critical points
    X_c = find_1D_critical_points(dXdT,  xlim=xlim, res_c=res_c, epsilon=epsilon)
    
    # Finding stability
    X_c_sign = find_sign_of_derivative(dXdT, X_c)
    facecolors = ["blue" if sign < 0 else "white" for sign in X_c_sign]
    
    # Plotting critical points
    ax.scatter(X_c, np.zeros(len(X_c)), s=100, color='blue', marker='o', facecolor=facecolors)
    ax.set_xlim(xlim)
    
# Functions implimenting Euler's method
def find_next_point_Euler_1D(x, dxdt, dt=DT):
    x_next = x + dxdt(x)*dt
    return(x_next)

# Function implimenting the midpoint method aka RK2
def find_next_point_midpoint_1D(x, dxdt, dt=DT):
    x_test = x + dxdt(x)*dt
    x_next = x + dxdt((x + x_test)/2.)*dt
    return(x_next)

# Simulation functions
def simulate_1D(x_0, t_f, dxdt, simulation_function, dt=DT):
    X_solution = [x_0]
    T = np.arange(0, t_f + dt, dt)
    for t in T[1:]:
        x_previous = X_solution[-1]
        x_next = simulation_function(x_previous, dxdt, dt=dt)
        X_solution.append(x_next)
    return (T, X_solution)

def simulate_1D_collection(X_0s, t_f, dxdt, simulation_function, dt=DT):
    X = []
    for X_0, in X_0s:
        T, X_sol = simulate_1D(X_0, t_f, dxdt, simulation_function, dt=dt)
        X.append(X_sol)
    return T, X

# Function to plot phase space diagram w/ critical points 

from itertools import product

def plot_1D_solutions(T_f, X_0s, dxdt, simulation_functions, colors=['k'], dt=DT, xlim=(X_MIN, X_MAX), x_label=X_LABEL):
    fig, (ax, ax_t) = plt.subplots(ncols=2, figsize=(9, 4))
    plot_1D_phase_space(dxdt, xlim=xlim, x_label=x_label, ax=ax)
    
    X_solutions = []
    try:
        iter(simulation_functions)
    except TypeError:
        simulation_functions = [simulation_functions]
        
    for simulation_function, X_0 in product(simulation_functions, X_0s):
        T, X_solution = simulate_1D(X_0, T_f, dxdt, simulation_function, dt=dt)
        X_solutions.append(X_solution)

    if len(colors) != len(X_solutions):
        colors = len(X_solutions) * colors

    for X_solution, color in zip(X_solutions, colors):
        ax_t.set_title("{0} vs. time".format(x_label))
        ax_t.spines['bottom'].set_position(('data',0))
        ax_t.plot(T, X_solution, color=color)
        ax_t.scatter(T[0], X_solution[0], color=color, zorder=10, s=10.0)
        ax_t.set_ylabel(x_label)
        ax_t.set_xlabel("Time")
        ax_t.set_ylim(xlim)