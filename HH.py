# Functions defining external current

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn 

## Possible currents
def make_static_i(t, i_0=0):
    def i_e(t):
        return(i_0) 
    return(i_e) 

def make_sinusoidal_i(t_s, A, T):
    def i_e(t):
        if (t>=t_s):
            return A*np.sin(2*np.pi*t/T)
        else:
            return 0
    return(i_e)

def make_ramping_i(t_s, t_f, i_max):
    def i_e(t):
        if (t>=t_s):
            return i_max*(t - t_s)/ (t_f - t_s)
        elif (t>=t_f):
            return i_max
        else:
            return 0
    return(i_e)

def make_noise_i(t_s, i_avg, A):
    def i_e(t):
        if (t>=t_s):
            return A*randn() + i_avg
        else:
            return 0
    return(i_e)

## A step current
def make_step_i(t_s, t_f, i_max=0.05):
    def i_e(t):
        if (t>=t_s) & (t<t_f):
            return i_max
        else:
            return 0
    return(i_e)

# Setting paramaters

## External current function (in micro-A/mm^2)
i = make_step_i(10, 200, 0.03)

## Capacitance (in micro-F/mm^2)
c = 0.010

## Conductances (in mS/mm^2)
gL = 0.003
gK = 0.036
gNa = 1.2

## Reversal potentials (in mV)
eL = -70
eK = -77
eNa = 50

# Membrane voltage dynamics

def dVdt(V, n, m, h, t, i=lambda t: 0):
    i_m = gL*(V-eL) + gK*(n**4)*(V-eK) + gNa*(m**3)*h*(V-eNa) 
    return((-i_m + i(t)) / c)

# Voltage-gated channel subunit dynamics

def alpha_n(V):
    return(0.01*(V + 55) / (1 - np.exp(-0.1*(V + 55))))

def beta_n(V):
    return(0.125*np.exp(-0.0125*(V + 65)))

def dndt(V, n):
    delta = alpha_n(V)*(1.0-n) - beta_n(V)*n
    if (delta > 1.-n) or (n>1.):
        delta = 1.-n
    elif (delta < -n) or (n<0):
        delta = -n
    return(delta) 

def alpha_m(V):
    return(0.1*(V + 40) / (1 - np.exp(-0.1*(V + 40))))

def beta_m(V):
    return(4*np.exp(-0.0556*(V + 65)))

def dmdt(V, m):
    delta = alpha_m(V)*(1.0-m) - beta_m(V)*m
    if (delta > 1.-m) or (m>1.):
        delta = 1. - m
    elif (delta < -m) or (m<0.):
        delta = -m
    return(delta) 

def alpha_h(V):
    return(0.07*np.exp(-0.05*(V + 65)))

def beta_h(V):
    return(1 / (1 + np.exp(-0.1*(V + 35))))

def dhdt(V, h):
    delta = alpha_h(V)*(1.0-h) - beta_h(V)*h
    if (delta > 1.-h) or (h>1.):
        delta = 1.-h
    elif (delta < -h) or (h<0.):
        delta = -h
    return(delta) 

# Creating functions for asymptotic values

def n_inf(V):
    return(alpha_n(V) / (alpha_n(V) + beta_n(V)))
           
def n_tau(V):
    return(1. / (alpha_n(V) + beta_n(V)))

def m_inf(V):
    return(alpha_m(V) / (alpha_m(V) + beta_m(V)))
           
# Creating functions for time constants

def m_tau(V):
    return(1. / (alpha_m(V) + beta_m(V)))
    
def h_inf(V):
    return(alpha_h(V) / (alpha_h(V) + beta_h(V)))
           
def h_tau(V):
    return(1. / (alpha_h(V) + beta_h(V)))

# Putting all the dynamics together into one function
def dSdt_HH(V, n, m, h, t):
    return(np.array([dVdt(V, n, m, h, t), dndt(V, n), dmdt(V, m), dhdt(V, h), 1.]))

def make_dSdt_HH(i):
    def dSdt_HH(V, n, m, h, t):
        return(np.array([dVdt(V, n, m, h, t, i=i), dndt(V, n), dmdt(V, m), dhdt(V, h), 1.]))
    return dSdt_HH

# A function to plot 
def plot_HH_dynamics(V, n, m, h, T):
    fig, (ax_i, ax_v, ax_c, ax_c2) = plt.subplots(nrows=4, sharex=True)

    ## Plotting current
    ax_i.plot(T, [i(t) for t in T], c='orange', label="$i_e$")
    ax_i.legend()
    ax_i.set_ylabel("$\\mathrm{\mu A}/\\mathrm{mm^2}$")

    ## Plotting membrane voltage
    ax_v.plot(T, V, c='k', label="$V$")
    ax_v.legend()
    ax_v.set_ylabel("$\\mathrm{mV}$")

    ## Plotting activation and inactivation gates
    ax_c.plot(T, n, c='blue', label="$n$")
    ax_c.plot(T, m, c='green', label="$m$")
    ax_c.plot(T, h, c='green', ls=':', label="$h$")
    ax_c.legend()
    ax_c.set_ylabel("Prob.")

    ## Plotting conductances
    ax_c2.plot(T, gK*n**4, c='blue', label="$\\bar{g}_{K} n^4$")
    ax_c2.plot(T, gNa*m**3*h, c='green', label="$\\bar{g}_{Na} m^3h$")
    ax_c2.legend()
    ax_c2.set_ylabel("$\\mathrm{mS}/\\mathrm{mm}^2$")

    ax_c2.set_xlabel("Time (ms)")
