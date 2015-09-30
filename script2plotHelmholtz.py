# a script to plot exterior solutions to Helmhotz eqn.
# Last modified: May 25, 2015

import numpy
import matplotlib.pyplot as plt
import smoothStar
import scipy.special as spec
import scipy.linalg as linalg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#parameters
a=.3
w=5
Nqd = 200
omega=4

#quadrature and gemoetry
quad = numpy.zeros([2,Nqd])
quad[0,:] = 2*numpy.pi*numpy.linspace(1,Nqd,Nqd)/Nqd
quad[1,:] = 2*numpy.pi*numpy.ones([1,Nqd])/Nqd
g=smoothStar.makeStar(a,w,quad)

# Set up the BIEs
DX = g.nodes[0,numpy.newaxis].T - g.nodes[0,numpy.newaxis]
DY = g.nodes[1,numpy.newaxis].T - g.nodes[1,numpy.newaxis]
R = numpy.sqrt(DX**2 + DY**2)
weights = quad[1,:]*g.speed

phi1=1j/4*spec.hankel1(0,omega*R)
phi2=1j*omega/4*spec.hankel1(1,omega*R)*(DX*g.normal[0,:].T+DY*g.normal[1,:].T)/R
numpy.fill_diagonal(phi1,0)
numpy.fill_diagonal(phi2,0)
phi = (phi2-phi1)*weights.T

# kapur-rokhlin weights of order 2
gamma=numpy.array([1.825748064736159,-1.325748064736159])
r = numpy.zeros((1,Nqd))
r[0,1] = gamma[0]
r[0,2] = gamma[1]
W=linalg.toeplitz(r)
Phi=phi*W

# Solve
## RHS = numpy.sin(2*g.nodes[0,:])*numpy.exp(numpy.cos(4*g.nodes[1,:]))
RHS = 1j/4*spec.hankel1(0,omega*numpy.sqrt((g.nodes[0,:]-5)**2+g.nodes[1,:]**2))
sigma = numpy.linalg.solve( 0.5*numpy.identity(Nqd)+Phi,RHS)

#plotting parameters
Nobs=200
xv=numpy.linspace(-4,4,Nobs)
yv=numpy.linspace(-4,4,Nobs)
Xobs,Yobs = numpy.meshgrid(xv,yv)
obs = numpy.array([numpy.reshape(Xobs,Nobs**2),numpy.reshape(Yobs,Nobs**2)])
DXobs = obs[numpy.newaxis,0].T - g.nodes[0,numpy.newaxis]
DYobs = obs[numpy.newaxis,1].T - g.nodes[1,numpy.newaxis]
Robs = numpy.sqrt(DXobs**2 + DYobs**2)

# Postprocess
phiobs1=1j/4*spec.hankel1(0,omega*Robs)
phiobs2=1j*omega/4*spec.hankel1(1,omega*Robs)*(DXobs*g.normal[0,:].T+DYobs*g.normal[1,:].T)/Robs
phiobs = (phiobs2-phiobs1)*weights.T
sol =numpy.dot(phiobs,sigma)

# plot
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(Xobs, Yobs, numpy.real(sol.reshape(Nobs,Nobs)),cmap=cm.coolwarm)
plt.show()

