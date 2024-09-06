

import glob
import os
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.multiprocessing as multiprocessing
from multiprocessing import Pool

import Spin_neuralODE

PROCESSE_COUNT = 8
JOB_COUNT = 32

scaleFactor = 1.33352143

def iterateRun(n):
    #commandLineConfigs = {"spectorRadius": pow(scaleFactor, n)*0.1}
    with torch.cuda.device(n % torch.cuda.device_count()):
        Spin_neuralODE.spinNeuralOde(commandLineConfigs)
    os._exit(n)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    commandLineConfigs = {"spectorRadius": 100.0}
    Spin_neuralODE.spinNeuralOde(commandLineConfigs)