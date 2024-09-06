import math
import numpy as np

import torch

import common

# Do you use a GPU?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

def loadLlgConfigs(llgConfigFileName, separator):
    tempLlgConfigs = np.genfromtxt(llgConfigFileName, dtype = "unicode")
    llgConfigs = {"magnetization": float(tempLlgConfigs[0])
                  , "demagCoef": torch.tensor([float(tempLlgConfigs[1]), float(tempLlgConfigs[2]), float(tempLlgConfigs[3])]).to(device)
                  , "magneticField": torch.tensor([float(tempLlgConfigs[4]), float(tempLlgConfigs[5]), float(tempLlgConfigs[6])]).to(device)
                  , "appliedField": float(tempLlgConfigs[7])
                  , "theta_H": common.arcToRad(float(tempLlgConfigs[8]))
                  , "phi_H": common.arcToRad(float(tempLlgConfigs[9]))
                  , "dip": float(tempLlgConfigs[10]) 
                  , "current": float(tempLlgConfigs[11]) 
                  , "spinPolarization": float(tempLlgConfigs[12])
                  , "torqueAsymmetry": float(tempLlgConfigs[13])
                  , "beta": float(tempLlgConfigs[14])
                  , "theta_p": common.arcToRad(float(tempLlgConfigs[15]))
                  , "phi_p": common.arcToRad(float(tempLlgConfigs[16]))
                  , "thickness": float(tempLlgConfigs[17])
                  , "radius": torch.tensor([float(tempLlgConfigs[18]), float(tempLlgConfigs[19])]).to(device)
                  , "gyro": float(tempLlgConfigs[20])
                  , "alpha": float(tempLlgConfigs[21])
                  , "temperature": float(tempLlgConfigs[22])
                  , "theta": common.arcToRad(float(tempLlgConfigs[23]))
                  , "phi": common.arcToRad(float(tempLlgConfigs[24]))
                }
    return llgConfigs

def loadOdeConfigs(odeConfigFileName, separator):
    tempOdeConfigs = np.genfromtxt(odeConfigFileName, dtype = "unicode")
    llgConfigs = {"solverName": tempOdeConfigs[0]
                  ,"totalTime": float(tempOdeConfigs[1])
                  , "stepSize": float(tempOdeConfigs[2])
                  , "samplingInterval": float(tempOdeConfigs[3])
                }
    return llgConfigs

def loadLyapunovConfigs(configFileName, separator):
    tempConfigs = np.genfromtxt(configFileName, dtype = "unicode")
    return {"solverName": tempConfigs[0]
           ,"washoutTime": float(tempConfigs[1])
           ,"evaluationTime": float(tempConfigs[2])
           , "stepSize": float(tempConfigs[3])
           , "samplingInterval": float(tempConfigs[4])
           , "samplingTime": float(tempConfigs[5])
           , "perturbationNorm": float(tempConfigs[6])
           , "seed": int(tempConfigs[7])
           }

def loadInputLlgConfigs(configFileName, separator):
    tempConfigs = np.genfromtxt(configFileName, dtype = "unicode")
    configs = {"inputCount": int(tempConfigs[0])
              , "inputScaleFactors": tempConfigs[1].split(',')
              , "theta_input": common.arcToRad(float(tempConfigs[2]))
              , "phi_input": common.arcToRad(float(tempConfigs[3]))
              , "seed": int(tempConfigs[4])
              }

    return configs
    


def loadCoupledLlgConfigs(configFileName, separator):
    tempConfigs = np.genfromtxt(configFileName, dtype = "unicode")
    configs = {"stoCount": int(tempConfigs[0])
              , "spectorRadius": float(tempConfigs[1])
              , "theta_coupling": common.arcToRad(float(tempConfigs[2]))
              , "phi_coupling": common.arcToRad(float(tempConfigs[3]))
              , "seed": int(tempConfigs[4])
              }
    return configs

def loadCommandLineConfigs(configs, commandLineConfigs):
    configNames = list(configs)
    for configName in configNames:
        if configName in commandLineConfigs:
            configs[configName] = commandLineConfigs[configName]