#!/usr/bin/env python
# coding: utf-8

# In[59]:


# import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.special as sp
from scipy.optimize import fsolve

#Parameters
N = 20
L0 = 1
SIGMA_r = 1
A0 = 0.0015
r0 = 2
px = 10


##COLLOCATION POINTS
#Collocation points for r
# k_values = np.arange(0, 2 * N + 4)
# x_col = np.cos(np.pi * k_values / (2 * N + 3))
# r_col_pre = L0 * x_col / np.sqrt(1 - x_col**2)
# r_col = np.array([r_col_pre[N + 2 - k] for k in range(1, N + 2)])
# print('r_col=', r_col)  #confere

######### no division for zero
# k_values = np.arange(0, 2*N + 4)
# x_col = np.cos(np.pi * k_values / (2*N + 3))
# epsilon = 1e-15  #para evitar divisão exata por zero
# r_col_pre = L0 * x_col / np.sqrt(1 - x_col**2 + epsilon)
# r_col = np.array([r_col_pre[N + 2 - k] for k in range(1, N + 2)])
####


#new r collocation points with linspace
k_values = np.linspace(0, 2*N + 3, 2*N + 4, dtype=np.float64)
x_col = np.cos(np.pi * k_values / (2*N + 3))
epsilon = 1e-15  # Valor pequeno para evitar divisão por zero
r_col_pre = L0 * x_col / np.sqrt(1 - x_col**2 + epsilon)
r_col = np.array([r_col_pre[N + 2 - k] for k in range(1, N + 2)])
# print(r_col)
# np.savetxt('r_col.txt', r_col, fmt='%.20f', header= 'r_col values')

#collocation points for x
P_prime = sp.legendre(2 * px + 1).deriv()
x_roots = fsolve(P_prime, np.cos(np.pi * (np.arange(1, 2 * px + 2) / (2 * px + 2))))
x_col_prel = np.concatenate(([-1], np.sort(x_roots), [1]))
x__col = -np.flip(x_col_prel[:px + 1])
# print('x=',x__col)
# np.savetxt('x__col.txt', x__col, fmt='%.20f', header='x_col')



######### 2n+1 #########
##BASES
#r basis
def SB(n, r):
    return np.sin((n+1)*np.arctan(L0/r))
# print(SB(1,2))

def rSB(n, r):
    theta = np.arctan(L0/r)
    return -np.cos((n+1)*theta)*(n+1)*L0/(r**2*(1+L0**2/r**2))

def rrSB(n, r):
    theta = np.arctan(L0/r)
    term1 = -np.sin((n+1)*theta)*(n+1)**2*L0**2/(r**4*(1+L0**2/r**2)**2)
    term2 = 2*np.cos((n+1)*theta)*(n+1)*L0/(r**3*(1+L0**2/r**2))
    term3 = -2*np.cos((n+1)*theta)*(n+1)*L0**3/(r**5*(1+L0**2/r**2)**2)
    return term1 + term2 + term3


#x basis
def P(i, x):
    return sp.legendre(i)(x)
# print(P(12,5))

def xP(i, x):
    return sp.legendre(i).deriv()(x)

def xxP(i, x):
    return sp.legendre(i).deriv().deriv()(x)

# ##COLLOCATION POINS ON THE BASES
# #r
# psi = np.array([[SB(i, r_val) for r_val in r_col] for i in range(N+1)])
# rpsi = np.array([[rSB(i, r_val) for r_val in r_col] for i in range(N+1)])
# rrpsi = np.array([[rrSB(i, r_val) for r_val in r_col] for i in range(N+1)])

# #x
# P = np.array([[legendre_poly(i, x_val) for x_val in x__col] for i in range(px+1)])
# P_x = np.array([[legendre_poly_x(i, x_val) for x_val in x__col] for i in range(px+1)])
# P_xx = np.array([[legendre_poly_xx(i, x_val) for x_val in x__col] for i in range(px+1)])


#complete bases
def Psi(n, r, x):
    list = [SB(2 * i, r) * P(j, x)
            for i in range(N + 1) for j in range(px + 1)]
    return list[n]

def rPsi(n,r,x):
    list = [rSB(2 * i,r) * P(j, x)
           for i in range (N + 1) for j in range(px + 1)]
    return list[n]

def rrPsi(n,r,x):
    list = [rrSB(2 * i,r) * P(j, x)
           for i in range (N + 1) for j in range(px + 1)]
    return list[n]

def xPsi(n,r,x):
    list = [SB(2 * i,r) * xP(j, x)
           for i in range (N + 1) for j in range(px + 1)]
    return list[n]

def xxPsi(n,r,x):
    list = [SB(2 * i,r) * xxP(j, x)
           for i in range (N + 1) for j in range(px + 1)]
    return list[n]


#collocation points on the bases
Psim = [[SB(2 * i, r_col[k]) * P(j, x__col[n])
            for i in range(N + 1)
            for j in range(px + 1)]
            for k in range(N + 1)
            for n in range(px + 1)]
# print(Psi)
# np.savetxt('Psi.txt', Psi, fmt='%.9f', header='Psi')

rPsim = [[rSB(2 * i, r_col[k]) * P(j, x__col[n])
            for i in range(N + 1)
            for j in range(px + 1)]
            for k in range(N + 1)
            for n in range(px + 1)]
# np.savetxt('rPsi.txt', rPsi, fmt='%.9f', header='rPsi')

rrPsim = [[rrSB(2 * i, r_col[k]) * P(j, x__col[n])
            for i in range(N + 1)
            for j in range(px + 1)]
            for k in range(N + 1)
            for n in range(px + 1)]
# np.savetxt('rrPsi.txt', rrPsi, fmt='%.9f', header='rrPsi')

xPsim = [[SB(2 * i, r_col[k]) * xP(j, x__col[n])
            for i in range(N + 1)
            for j in range(px + 1)]
            for k in range(N + 1)
            for n in range(px + 1)]
# np.savetxt('xPsi.txt', xPsi, fmt='%.9f', header='xPsi')

xxPsim = [[SB(2 * i, r_col[k]) * xxP(j, x__col[n])
            for i in range(N + 1)
            for j in range(px + 1)]
            for k in range(N + 1)
            for n in range(px + 1)]
# np.savetxt('xxPsi.txt', xxPsi, fmt='%.9f', header='xxPsi')


# #inverse matrix
inv_psi = np.linalg.inv(Psim)
# print(inv_psi.shape)



# In[67]:


#initial data

R, X = np.meshgrid(r_col, x__col)

def gaussian(r,x):
    return np.exp(-(r)**2) * (-x**2 + 1)

def gauss_approx(r,x,a0):
    res = sum(a0[k] * Psi(k, R, X) for k in range((N+1) * (px+1)))
    return res

gaussian_approx = gaussian(R,X)

Phi_0 = gaussian_approx.reshape(-1, 1)
# print(Phi_0.shape) #(231,1)

a0 = np.dot(inv_psi,Phi_0)
# print(a0.shape) #(231,1)

Phi0 = np.dot(Psim, a0)

Phi0plot = gauss_approx(R,X,a0)


ax = plt.axes(projection = '3d')
ax.plot_surface(R, X, Phi0plot, cmap='viridis')
plt.savefig('initial_phi.png', dpi=300, bbox_inches='tight')  # Pode usar .jpg, .tiff, etc.
plt.show()


# In[68]:


#grid plot

M = 100

rplot = np.linspace(0.0001,1,M)
xplot = np.linspace(0.0001,1,M)

Psiplot = [[SB(2 * i, rplot[k]) * P(j, xplot[n])
            for i in range(N + 1)
            for j in range(px + 1)]
            for k in range(N + 1)
            for n in range(px + 1)]

# rPsiplot = [[rSB(2 * i, rplot[k]) * P(j, xplot[n])
#             for i in range(N + 1)
#             for j in range(px + 1)]
#             for k in range(N + 1)
#             for n in range(px + 1)]

# rrPsiplot = [[rrSB(2 * i, rplot[k]) * P(j, xplot[n])
#             for i in range(N + 1)
#             for j in range(px + 1)]
#             for k in range(N + 1)
#             for n in range(px + 1)]

# xPsiplot = [[SB(2 * i, rplot[k]) * xP(j, xplot[n])
#             for i in range(N + 1)
#             for j in range(px + 1)]
#             for k in range(N + 1)
#             for n in range(px + 1)]

# xxPsiplot = [[SB(2 * i, rplot[k]) * xxP(j, xplot[n])
#             for i in range(N + 1)
#             for j in range(px + 1)]
#             for k in range(N + 1)
#             for n in range(px + 1)]

    
# Rplot, Xplot = np.meshgrid(rplot, xplot)


# In[ ]:


Rm = np.repeat(r_col,11)
Rr = Rm.reshape(-1,1)
r = Rr

Xm = np.repeat(x__col,21)
Xr = Xm.reshape(-1,1)
x = Xr

da = np.zeros((N+1)*(px+1))

h = 0.001
tf = 5

It = int(tf/h)

t = np.linspace(0, tf, It)

phi_set = np.zeros([It,M])
pi_set = np.zeros([It,M])
drphi_set = np.zeros([It,M])


for i in range(It):  # Runge Kutta 4th order

    phi = np.dot(a0.T, Psim)
    dda = np.dot(np.dot(a0.T, rrPsim + 2/r*rPsim + (-x**2 + 1)*xxPsim/r**2 - 2*x*xPsim), inv_psi)
    L1 = h*(da)
    K1 = h*(dda)

    phi = np.dot(a0.T + L1/2, Psim)
    dda = np.dot(np.dot(a0.T + L1/2, rrPsim + 2/r*rPsim + (-x**2 + 1)*xxPsim/r**2 - 2*x*xPsim), inv_psi)
    L2 = h*(da + K1/2)
    K2 = h*(dda)

    phi = np.dot(a0.T + L2/2, Psim)
    dda = np.dot(np.dot(a0.T + L2/2, rrPsim + 2/r*rPsim + (-x**2 + 1)*xxPsim/r**2 - 2*x*xPsim), inv_psi)
    L3 = h*(da + K2/2)
    K3 = h*(dda)

    phi = np.dot(a0.T + L3, Psim)
    dda = np.dot(np.dot(a0.T + L3, rrPsim + 2/r*rPsim + (-x**2 + 1)*xxPsim/r**2 - 2*x*xPsim), inv_psi)
    L4 = h*(da + K3)
    K4 = h*(dda)

    da = da + 1/6 * (K1 + 2*K2 + 2*K3 + K4)
    a0 = a0 + 1/6 * (L1 + 2*L2 + 2*L3 + L4)
    phi_set[i,:] = np.dot(a0, Psiplot)
#     pi_set[i,:] = np.dot(da, psiplot)
#     drphi_set[i,:] = np.dot(a0, rpsiplot)


