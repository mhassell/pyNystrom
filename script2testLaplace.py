# script to test the laplace functions
# January 24, 2015

import numpy
import matplotlib.pyplot as plt
import smoothStar
import laplace
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#parameters
a=.3
w=5
Nqd = 100

#quadrature and gemoetry
quad = numpy.zeros([2,Nqd])
quad[0,:] = 2*numpy.pi*numpy.linspace(1,Nqd,Nqd)/Nqd
quad[1,:] = 2*numpy.pi*numpy.ones([1,Nqd])/Nqd
g=smoothStar.makeStar(a,w,quad)

#observation and source points
sc = numpy.array([2,3])
obs = numpy.array([[0,0],[-.5,.5],[1,0]])

# Solve and postprocess
K = laplace.laplaceOperators(g,quad)
Op = K-0.5*numpy.eye(Nqd)
RHS = numpy.log(numpy.sqrt((g.nodes[0,:]-sc[0])**2+(g.nodes[1,:]-sc[1])**2))
sigma = numpy.linalg.solve(Op,RHS)
D = laplace.laplacePotentials(g,obs,quad)
sol = numpy.dot(D,sigma)

# errors
exact =  numpy.log(numpy.sqrt((obs[:,0]-sc[0])**2+(obs[:,1]-sc[1])**2))
err = numpy.max(numpy.abs(exact-sol))
print(err)

#plotting
N=500

xv=numpy.linspace(-1.5,1.5,N)
yv=numpy.linspace(-1.5,1.5,N)
X,Y = numpy.meshgrid(xv,yv)

obs2 = numpy.array([numpy.reshape(X,N**2),numpy.reshape(Y,N**2)]).T
D2 = laplace.laplacePotentials(g,obs2,quad)
sol2 = numpy.dot(D2,sigma)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, sol2.reshape(N,N),cmap=cm.coolwarm)
plt.show()
