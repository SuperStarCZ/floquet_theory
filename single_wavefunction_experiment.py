#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 05:55:46 2020

@author: mahi
"""

import numpy as np
from odeintw import odeintw
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot, norm, eig
import math
import time
from array import *

start = time.time()

cosp_init = 0.0    
N = 10
h = 25.0
h0 = 0.1
nn =0   #this is the th column of the floquet evolution matrix for which the path we shall follow

q = np.linspace(-0.5, 0.5, N)
omega_range = np.linspace(6.25,6.255,50)
omegas = omega_range


psi = np.eye(N) + (1j) * np.zeros((N,N))
cosp = np.zeros((N,N)) + (1j) * np.zeros((N,N))
cosp_init = np.zeros((N,N)) + (1j) * np.zeros((N,N))
floqEvolution_mat = np.zeros((N,N)) + (1j) * np.zeros((N,N))
phasefunc_path = np.zeros(len(omegas))

title = "mf floquet dynamics: n " + str(N)
photoname = "n_" + str(N) + "_mfd_bessel.jpeg"
filename = "n_" + str(N) + "_mfd.txt"

class Periodic_Lattice(np.ndarray):
    def __new__(cls, input_array, lattice_spacing=None):
        obj = np.asarray(input_array).view(cls)
        obj.lattice_shape = input_array.shape
        obj.lattice_dim = len(input_array.shape)
        obj.lattice_spacing = lattice_spacing
        return obj
    
    def __getitem__(self, index):
        index = self.latticeWrapIdx(index)
        return super(Periodic_Lattice, self).__getitem__(index)
    
    def __setitem__(self, index, item):
        index = self.latticeWrapIdx(index)
        return super(Periodic_Lattice, self).__setitem__(index, item)
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.lattice_shape = getattr(obj, 'lattice_shape', obj.shape)
        self.lattice_dim = getattr(obj, 'lattice_dim', len(obj.shape))
        self.lattice_spacing = getattr(obj, 'lattice_spacing', None)
        pass
    
    def latticeWrapIdx(self, index):
        if not hasattr(index, '__iter__'): return index         # handle integer slices
        if len(index) != len(self.lattice_shape): return index  # must reference a scalar
        if any(type(i) == slice for i in index): return index   # slices not supported
        if len(index) == len(self.lattice_shape):               # periodic indexing of scalars
            mod_index = tuple(( (i%s + s)%s for i,s in zip(index, self.lattice_shape)))
            return mod_index
        raise ValueError('Unexpected index: {}'.format(index))


def floq_jac(periodic_psi,t, h, h0, w, cosp):
    drive = h0 + h * np.cos(w * t)
    jac = (1j) * N * (-2.0 * q * q - drive * np.sqrt(1.0 - 4.0 * q *q) * cosp)
    return jac

def floq_func(periodic_psi,t,h,h0,w,cosp):
    return np.dot(floq_jac(periodic_psi,t, h0, h, w, cosp), periodic_psi)

    
if __name__ == '__main__':
    periodic_psi  = Periodic_Lattice(psi)       
    cospinit = 0.0
    
    for k in np.arange(N):
        for m in np.arange(N):
            for l in np.arange(N+1):
                for j in np.arange(l+1):
                    cospinit = cospinit + np.array([pow((-1),(l+j))/math.factorial(2 * l)* \
                                          math.comb(l,j)* periodic_psi[(k+(l-j),(m))]])
            cosp[k,m] = cospinit
            cospinit = 0.0
    
    # calculate for first frequency
    w = omegas[0]
    print('w',w)
    T = 2 * np.pi/w                            
    t = np.linspace(0,2 * np.pi/w,N)           
    floqEvolution_mat = np.zeros((N,N)) + (1j) * np.zeros((N,N))
    for mm in np.arange(N):
        psi0 = periodic_psi[mm]       
        psi_t = odeintw(floq_func,psi0,t,args=(h,h0,w,cosp), Dfun=floq_jac)
        floqEvolution_mat[mm] = psi_t[N-1]
    
    #print('flomat',floqEvolution_mat)
    evals, evecs = eig(floqEvolution_mat)
    phasefunc = (1j * np.log(evals[nn]))/T
    evecs_path = evecs[nn]/(np.linalg.norm(evecs[nn]))     # here the eigevector is normalised
    print('evecs_path1',evecs_path)
    #print('dot 1 =',np.dot(evecs_path,evecs_path))
    phasefunc_path[0] = phasefunc
    
    # calculate for rest of the frequencies
    for j, w in enumerate(omegas[1:len(omegas)]):
        #print('evecs_path',evecs_path)
        #print('j,w=',j+1,w)
        T = 2 * np.pi/w                      # time periode
        t = np.linspace(0,2 * np.pi/w,N)     # time range
        floqEvolution_mat = np.zeros((N,N)) + (1j) * np.zeros((N,N))        
        for mm in np.arange(N):
            psi0 = periodic_psi[mm]       
            psi_t = odeintw(floq_func,psi0,t,args=(h,h0,w,cosp), Dfun=floq_jac)
            floqEvolution_mat[mm] = psi_t[N-1] 
        evals, evecs = eig(floqEvolution_mat)
        #print('flomat1',floqEvolution_mat)
        for xx in np.arange(N):
            print('dot product',np.dot(np.conjugate(evecs_path),\
                                       evecs[xx]/(np.linalg.norm(evecs[xx]))).real)
            if (np.abs(1-np.dot(np.conjugate(evecs_path),\
                                evecs[xx]/(np.linalg.norm(evecs[xx]))).real) <= 0.00001):
                evecs_path = evecs[xx]/(np.linalg.norm(evecs[xx]))
                pp = xx
                print('pp',pp)
                break
        
        #hasefunc = (1j * np.log(evals[pp]))/T
        #print('phasefunct',phasefunc)
        #phasefunc_path[j+1] = phasefunc