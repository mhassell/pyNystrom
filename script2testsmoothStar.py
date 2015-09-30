import numpy
import matplotlib.pyplot as plt
import smoothStar

a=.3
w=5
Nqd = 5

quad = numpy.zeros([2,Nqd])
quad[0,:] = 2*numpy.pi*numpy.linspace(0,Nqd,Nqd)/Nqd
quad[1,:] = 2*numpy.pi*numpy.ones([1,Nqd])/Nqd

g=smoothStar.makeStar(a,w,quad)

plt.plot(g.nodes[0],g.nodes[1])
plt.show()
