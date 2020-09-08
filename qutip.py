import numpy as np
import matplotlib.pyplot as plt
import math
from qutip import *
from scipy import *

N = 4 
q = np.linspace(-0.5, 0.5, N)
cosp = np.zeros((N,N)) + (1j) * np.zeros((N,N))
cosp_init = np.zeros((N,N)) + (1j) * np.zeros((N,N))
omega = 1.0 * 2 * np.pi
h_vec = np.linspace(0, 10, 100)/omega
h0 = 0.1
T = (2*np.pi)/omega
tlist  = np.linspace(0.0, 10 * T, 101)
args = {'w': omega}

cospinit = 0.0
for k in np.arange(N):
    for m in np.arange(N):
        for l in np.arange(N+1):
            for j in np.arange(l+1):
                cospinit = cospinit + np.array([pow((-1),(l+j))/math.factorial(2 * l)* math.comb(l,j)])
        cosp[k,m] = cospinit
        cospinit = 0.0
                                                
psi0   = np.zeros(N)
psi0[0] = 1.0
psi0 = Qobj(psi0)
H0 = Qobj(-2*q*q)
H1 = Qobj(- np.sqrt(1.0 - 4.0 * q * q) * cosp)
for idw, wv in enumerate(h_vec):
    H = [H0,[H1,lambda t,h0: h0+h*np.cos(args[w] * t)]]
    f_modes, f_energies = floquet_modes(H, T, args, True)