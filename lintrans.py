import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
from matplotlib import cm

# Creating a set of vectors arranged on a circle to vizualize
R = 1.0 # radius the points will sit at. 
N_POINTS = 60 # the number of points to draw
theta = np.linspace(0, 2*np.pi, N_POINTS, endpoint=False) # angles for the points
V_sel = [0, 15] # Special points to fully draw the vectors  

# Defining the set of vectors, one at each theta
V = np.array([(R*np.cos(i), R*np.sin(i)) for i in theta]).T

# Picking some pretty colors to use for the points
hsv = cm.get_cmap('hsv', N_POINTS)
colors = list(hsv(range(N_POINTS)))

# Defining some functions to plot the points and special points in the transformation

def plot_points(points, sel=None, colors='blue', marker='o', size=10.0, alpha=1.0, zorder=0, figsize=(5,5), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    if not isinstance(colors, list):
        colors = len(points[0]) * [colors]   
        
    out = ax.scatter(points[0,:], points[1,:], c=colors, marker=marker, s=size, alpha=alpha, zorder=zorder)
    
    if sel is not None:
        for i in V_sel:
            ax.plot([0,points[0][i]], [0, points[1][i]], color=colors[i], linewidth=3, alpha=alpha)
    
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['left'].set_position(('data',0))
    ax.set_aspect('equal')
    return(ax, out)

def plot_trans(V, M, sel=V_sel, colors=colors, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    
    # Plotting the point
    plot_points(V, sel=V_sel, colors=colors, zorder=1, alpha=.2, ax=ax)
    
    # Matrix multiplicaton (@) of points V by M to get new points V2
    V2 = M @ V
    
    # Plotting the new points
    plot_points(V2, sel=V_sel, colors=colors, zorder=3, ax=ax)
    
    return(ax)


def plot_eigs(ax, M, zorder=3):
    w, v = np.linalg.eig(M)
    T = np.linspace(-100, 100, 10)
    v1 = v[:,0]
    v2 = v[:,1]
    
    X1 = v1[0]*T
    Y1 = v1[1]*T
    
    X2 = v2[0]*T
    Y2 = v2[1]*T

    ax.arrow(0, 0, w[0]*v1[0], w[0]*v1[1], width=.05, length_includes_head=True, color='k')
    ax.arrow(0, 0, v1[0], v1[1], width=.05, length_includes_head=True, color='.5')

    ax.arrow(0, 0, w[1]*v2[0], w[1]*v2[1], width=.05, length_includes_head=True, color='k')
    ax.arrow(0, 0, v2[0], v2[1], width=.05, length_includes_head=True, color='.5')
    pass

def prettyprint_eig(M):
    vals, mat = eig(M)
    for i in range(len(vals)):
        if np.imag(vals[i]) == 0.0:
            vals = np.array(vals, dtype='float')
        print(f"\tEigenvalue {i+1} is {vals[i]:.3}, with vector ({mat[0,i]:.3}, {mat[1,i]:.3})")
        
          
def plot_trans_w_eigs(V, M, sel=V_sel, colors=colors, ax=None):
    prettyprint_eig(M)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    
    # Plotting the points
    plot_points(V, colors=colors, zorder=1, alpha=.2, ax=ax)

    V2 = M @ V
    
    # Plotting the newpoints
    plot_points(V2, colors=colors, zorder=3, ax=ax)

    try:
        plot_eigs(ax, M)
    except:
        pass
    
    return(ax) 
