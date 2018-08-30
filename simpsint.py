# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:33:01 2018

Python equivalent of Krawczynski et al.(2004) simpsint.c numerical integral 
calculator. 

@author: cnrmc

Integrates dg from a to b, 
dg(1)... dg(n) corrosponds 
dg(a)... dg(b) equally spaced

N has to be odd
in di(x) int_a^x is stored
in dx(i) the abcissa(the distance from a point to the vertical or y -axis, 
measured parallel to the horizontal or x -axis; the x -coordinate) is stored

Uses Simpson's Rule: 
    Integral from a to b f(x)dx
    is roughly equal to
    ((deltaX)/3)*(f(x0)+4*f(x1)+2*f(x2)+...)
    
    with deltaX = (b-a)/n and xi = a+i*deltaX
"""
from numba import jit
import sys
MAXDIMS = 10001
@jit
def simpsint(N, dg, di, dx, a, b):
    
    delta, dresult = 0.0, 0.0
    
    f = [0.0]*MAXDIMS

    i, ind = 0,0
    
    delta = ((b-a)/(N-1))
    
    """
    Exception if number of steps is greater than number of bins
    """
    if (N > MAXDIMS):
        sys.exit("MAXDIMS too small " + str(N) + "\n")
        
    """
    Corrects incase of even N value inputed
    """    
    if ((N%2)!=1):
        b -= delta
        N -=1
        delta = ((b-a)/(N-1))
    
    """
    Copies array dg to f
    """
    for i in range(0, N):
        f[i] = dg[i]

    dresult = 0.0
    di[0] = 0.0
    dx[0] = a
    """
    Simpson's Integration method
    """
    for j in range(0, int(((N-1)/2))):
        ind = ((2*j)+1)
        dresult += delta/3.0*(f[ind]+4.0*f[ind]+f[ind+1])
        
        "Storing dresults in di"
        di[ind] = (di[ind-1]+dresult)/2.0
        di[ind+1] = dresult

        "Storing dresults in dx"
        dx[ind] = dx[ind-1]+delta
        dx[ind+1] = dx[ind]+delta

    return dresult 