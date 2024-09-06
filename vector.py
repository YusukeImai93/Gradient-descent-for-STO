import math
import numpy as np
from numpy import linalg

import torch

import common

def sphericalAngle(state1, state2):
    if state1.ndim == 2:
        return np.arccos(1- np.square(np.linalg.norm(state1-state2, axis = 1))/2.0)

def scaleAngle(state1, state2, k):
    angle = sphericalAngle(state1, state2).reshape(-1,1)
    #return (state1*np.sin((1-k)*angle)+state2*np.sin(k*angle))/np.sin(angle)
    return np.where(np.abs(angle) < common.THRESHOLDE, (1-k)*state1+k*state2 , (state1*np.sin((1-k)*angle)+state2*np.sin(k*angle))/np.sin(angle))

def sphericalCoordinateToCartesianCoordinate(sphiricalCoordinate):
    return sphiricalCoordinate[0]*torch.tensor([math.sin(sphiricalCoordinate[1])*math.cos(sphiricalCoordinate[2])
                                          , math.sin(sphiricalCoordinate[1])*math.sin(sphiricalCoordinate[2])
                                          , math.cos(sphiricalCoordinate[1])])

def cartesianToPolar(cartesian):
    if cartesian.ndim == 2:
        polar = torch.zeros([cartesian.shape[0], 2])
        polar[:,0] = torch.arccos(cartesian[:,2]/torch.linalg.norm(cartesian, axis = 1))
        polar[:,1] = torch.sign(cartesian[:,1]) * torch.arccos(cartesian[:,0]/torch.linalg.norm(cartesian[:,:1], axis = 1))
        return polar
    elif cartesian.ndim == 1:
        return torch.array([ torch.arccos(cartesian[2]/torch.linalg.norm(cartesian) ) \
                        , torch.sign(cartesian[1]) * torch.arccos(cartesian[0]/torch.linalg.norm(cartesian[:,:1])) \
                        ])

def arcToRad(arc):
    return arc*math.pi/180.0

def vectorizeOuterProd (a, b):
    
    #if a.ndim == 1:
    #    newA = a.reshape(1, -1)
    #else: newA = a
    newA = a

    ans = torch.zeros(b.shape)
    ans[:,:,0] = newA[:, :, 1] * b[:, :, 2] - newA[:, :, 2] * b[:, :, 1]
    ans[:,:,1] = newA[:, :, 2] * b[:, :, 0] - newA[:, :, 0] * b[:, :, 2]
    ans[:,:,2] = newA[:, :, 0] * b[:, :, 1] - newA[:, :, 1] * b[:, :, 0]
    return ans

def vectorizeInnerProd (a, b):
    
    #if a.ndim == 1:
    #    newA = a.reshape(1, -1)
    #else: newA = a
    newA = a

    ans = newA[:, :, 0] * b[:, :, 0] + newA[:, :, 1] * b[:, :, 1] + newA[:, :, 2] * b[:, :, 2]
    return ans

