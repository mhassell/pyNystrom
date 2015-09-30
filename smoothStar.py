# smoothStar module
# last modified 1/19/2015

import math
import numpy
import collections

def makeStar(a,w,quad):
    t=quad[0,:]
    R = lambda x: 1 + a*math.cos(w*x)
    Rp = lambda x: -w*a*math.sin(w*x)
    Rpp = lambda x: -w**2*a*math.cos(w*x)
    
    nodesx = numpy.array([R(s)*math.cos(s) for s in t])
    nodesy = numpy.array([R(s)*math.sin(s) for s in t])
    nodespx = numpy.array([-1*R(s)*math.sin(s)+Rp(s)*math.cos(s) for s in t])
    nodespy = numpy.array([R(s)*math.cos(s)+Rp(s)*math.sin(s) for s in t])
    nodesppx = numpy.array([-R(s)*math.cos(s) - Rp(s)*math.sin(s) - Rp(s)*math.cos(s) + Rpp(s)*math.cos(s) for s in t])
    nodesppy = numpy.array([-R(s)*math.sin(s) + Rp(s)*math.cos(s) + Rp(s)*math.cos(s) + Rpp(s)*math.sin(s) for s in t])

    speed = numpy.sqrt(nodespx**2 + nodespy**2)
    normal = numpy.array([nodespy/speed, -nodespx/speed])
    curvature = (nodespx*nodesppy - nodesppx*nodespy)/speed**3

    nodes = numpy.array([nodesx, nodesy])
    nodesp = numpy.array([nodespx, nodespy])
    nodespp = numpy.array([nodesppx, nodesppy])

    geometry = collections.namedtuple('geometry',['nodes','pnodes','ppnodes','speed','normal','curvature'])

    g=geometry(nodes,nodesp,nodespp,speed,normal,curvature)
    
    return g
