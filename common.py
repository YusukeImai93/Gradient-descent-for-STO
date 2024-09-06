import math
import numpy as np
from numpy import linalg as LA

import torch

#probram configs
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s :%(message)s"

#physical constant value
H_BAR = 1.05457266e-27
CHARGE = 1.60217733e-19
K_B = 1.3806504e-16

#dir name
RESULT_DIR_NAME = "result"
FIGURE_DIR_NAME = "figure"

#file name
ODE_CONFIG_FILE_NAME = "data/odeConfig.txt"
LLG_CONFIG_FILE_NAME = "data/llgConfig.txt"
COUPLED_LLG_CONFIG_FILE_NAME = "data/coupledLlgConfig.txt"
INPUT_LLG_CONFIG_FILE_NAME = "data/inputLlgConfig.txt"
LYAPUNOV_CONFIG_FILE_NAME = "data/lyapunovConfig.txt"

#thresholde
THRESHOLDE = 0.000001

def arcToRad(arc):
    return arc*math.pi/180.0
