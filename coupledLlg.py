import math

import numpy as np

import torch
import torch.nn as nn

import common
import vector
import random

# Do you use a GPU?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

class CoupledLlg(nn.Module):
#system parameter
    NAME = "coupledLlg"

    
    def __init__(self, configs, inputLlgConfigs, coupledLlgConfigs,appliedField_change,current_change,alpha_change,num_batch):
        super(CoupledLlg, self).__init__()
        
        self.num_batch = num_batch

        self.magnetization = configs["magnetization"]
        self.anisotropyField = configs["magneticField"] - 4.0 * self.magnetization * math.pi * configs["demagCoef"]
        #print("self.anisotropyField",self.anisotropyField)
        #self.appliedField = configs["appliedField"]
        self.appliedField = appliedField_change
        self.current = current_change
        self.alpha = alpha_change
        #print("field",self.appliedField)
        self.dip = configs["dip"]
        #self.current = configs["current"]
        self.spinPolarization = configs["spinPolarization"]
        self.torqueAsymmetry = configs["torqueAsymmetry"]
        self.beta = configs["beta"]
        self.pinned = vector.sphericalCoordinateToCartesianCoordinate(torch.tensor([1, configs["theta_p"], configs["phi_p"]])).reshape(1,1,-1)
        self.volume = configs["thickness"] * math.pi * configs["radius"][0] * configs["radius"][1]
        self.gyro = configs["gyro"]
        #self.alpha = configs["alpha"]
        self.temperature = configs["temperature"]

    #coupling configurations
        #self.inputWeight = 2*np.random.rand(self.stoCount, len(self.stoCount))-1
        self.stoCount = coupledLlgConfigs["stoCount"]
        np.random.seed(coupledLlgConfigs["seed"])
        self.seed = coupledLlgConfigs["seed"]
        self.dimension = self.stoCount*3

        self.appliedUnitVector = vector.sphericalCoordinateToCartesianCoordinate(torch.tensor([1, configs["theta_H"], configs["phi_H"]]))
        self.couplingAppliedUnitVector = vector.sphericalCoordinateToCartesianCoordinate(torch.tensor([1, coupledLlgConfigs["theta_coupling"], coupledLlgConfigs["phi_coupling"]]))
        
        if self.stoCount == 1:
            self.internalWeight = torch.zeros([1,1])
        else:
            torch_fix_seed()
            self.internalWeight = 2*torch.rand(self.stoCount, self.stoCount)-1
        #no self coupling
            for i in range(self.stoCount):
                self.internalWeight[i, i] = 0
        #set spector radius
            eigenValues,__ = torch.linalg.eig(self.internalWeight)
            self.internalWeight *= coupledLlgConfigs["spectorRadius"]/torch.amax(torch.abs(eigenValues))
            self.thetaCoupling = coupledLlgConfigs["theta_coupling"]
            self.phiCoupling = coupledLlgConfigs["phi_coupling"]      
                          

    def normalization(self, stepSize):
        self.fieldNormalizationFactor = torch.amax(torch.abs(self.anisotropyField))
        self.dt = (self.gyro * self.fieldNormalizationFactor * stepSize)

        self.pinnedMultipleddipoleField = self.dip / self.fieldNormalizationFactor * torch.tensor([-1, -1, 2]) * self.pinned

        self.anisotropy = self.anisotropyField / self.fieldNormalizationFactor

    def forward(self, time, states):
        applied = torch.zeros(self.num_batch,self.stoCount,3)
        applied[:,:,2] = self.appliedField[:,:,0]
        applied = (applied )/ self.fieldNormalizationFactor

        self.h_sst = common.H_BAR * self.spinPolarization * self.current  / (2.0 * common.CHARGE * self.magnetization * self.volume * self.fieldNormalizationFactor)

        asymmetricFactor = 1.0 / (1.0 + torch.sum(self.torqueAsymmetry * states * self.pinned, axis=2)).reshape(states.shape[0], states.shape[1], 1)
        temp = self.anisotropy * states

        b = applied + self.pinnedMultipleddipoleField.reshape(1, 1, -1)  + temp + (asymmetricFactor * self.h_sst) * (vector.vectorizeOuterProd(self.pinned, states) + self.beta * self.pinned.reshape(1, 1, -1))
        c = vector.vectorizeOuterProd(states, b)
        return (-1 / (1 + self.alpha*self.alpha)) * (c + self.alpha * vector.vectorizeOuterProd(states, c))
