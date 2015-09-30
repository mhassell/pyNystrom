# laplace Module
# last modified: January 21, 2015

import numpy

def laplaceOperators(g,quad):
    DX = g.nodes[0,numpy.newaxis].T - g.nodes[0,numpy.newaxis]
    DY = g.nodes[1,numpy.newaxis].T - g.nodes[1,numpy.newaxis]
    R = numpy.sqrt(DX**2 + DY**2)
    weights = quad[1,:]*g.speed
    #V = -numpy.log(R)*weights.T/(2*numpy.pi) #Needs special care

    K = (DX*g.normal[0,:].T+DY*g.normal[1,:].T)/(2*numpy.pi*R**2)
    numpy.fill_diagonal(K,0)
    K = K-numpy.diag(g.curvature)/(4*numpy.pi)
    K = K*weights.T
    return K


def laplacePotentials(g,obs,quad):
    obs=obs.T
    DX = obs[numpy.newaxis,0].T - g.nodes[0,numpy.newaxis]
    DY = obs[numpy.newaxis,1].T - g.nodes[1,numpy.newaxis]
    R = numpy.sqrt(DX**2 + DY**2)
    weights = quad[1,:]*g.speed

    #S = -numpy.log(R)*weights.T/(2*numpy.pi)
    D = (DX*g.normal[0,:].T+DY*g.normal[1,:].T)*weights
    D = D/(2*numpy.pi*R**2)
    return D

