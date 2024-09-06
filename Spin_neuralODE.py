import os
import math
import copy
import time
import datetime
import logging
from pickle import FALSE

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import common
import file
import plot
import coupledLlg
import random
from torch.optim.lr_scheduler import LambdaLR


from torchdiffeq import odeint_adjoint as odeint


### general parameter ###
num_epochs = 20
num_batch = 100 

### prameter range ###
applied_min = 1500.0e0
applied_max = 2000.0e0
applied_range = applied_max- applied_min
current_min = 2.5e-3
current_max = 7.5e-3
current_range = current_max- current_min
alpha_min = 1.0e-3
alpha_max = 10.0e-3
alpha_range = alpha_max- alpha_min

### paramter values for teacher dynamics ###
applied_answer_tmp = random.uniform(applied_min, applied_max) 
current_answer_tmp = random.uniform(current_min, current_max)
alpha_answer_tmp = 0.009e0 #0.002e0


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

if torch.cuda.is_available():
   device = 'cuda'
   torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
   device = 'cpu'

image_size = 28*28 
transform = transforms.Compose([
    transforms.ToTensor()
    ])

train_dataset = datasets.MNIST(
    './data',               # directory of data
    train = True,           # obtain training data
    download = True,        # if you don't have data, download
    transform = transform   # trainsform to tensor
    )
# evaluation
test_dataset = datasets.MNIST(
    './data', 
    train = False,
    transform = transform
    )

# data loader
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = num_batch,
    shuffle = True,
    #num_workers = 2,
    generator=torch.Generator(device='cuda')
    )
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,     
    batch_size = num_batch,
    shuffle = True,
    #num_workers = 2,
    generator=torch.Generator(device='cuda')
    )

#----------------------------------------------------------
# Definition of neural net
class ODEBlock(nn.Module):
    def __init__(self, input_size, output_size, iterationTime, commandLineConfigs, appliedField_change,current_change,alpha_change):
        super(ODEBlock, self).__init__()

        #initialize
        self.odeConfigs = file.loadOdeConfigs(common.ODE_CONFIG_FILE_NAME, " ")
        llgConfigs = file.loadLlgConfigs(common.LLG_CONFIG_FILE_NAME, " ")
        inputLlgConfigs = file.loadInputLlgConfigs(common.INPUT_LLG_CONFIG_FILE_NAME, " ")
        coupledLlgConfigs = file.loadCoupledLlgConfigs(common.COUPLED_LLG_CONFIG_FILE_NAME, " ")

        #commandLineConfigs
        file.loadCommandLineConfigs(coupledLlgConfigs, commandLineConfigs)
        
        applied_answer = torch.zeros(num_batch,1,1)
        current_answer = torch.zeros(num_batch,1,1)
        alpha_answer = torch.zeros(num_batch,1,1)
        applied_answer[:,:,0] = applied_answer_tmp
        current_answer[:,:,0] = current_answer_tmp
        alpha_answer[:,:,0] = alpha_answer_tmp

        self.appliedField_change = appliedField_change
        self.current_change = current_change
        self.alpha_change = alpha_change
        self.system = coupledLlg.CoupledLlg(llgConfigs, inputLlgConfigs, coupledLlgConfigs,applied_answer,current_answer,alpha_change,num_batch)
        # defining coupled LLG in coupledLLG.py
        self.system.normalization(self.odeConfigs["stepSize"])
        self.system_test = coupledLlg.CoupledLlg(llgConfigs, inputLlgConfigs, coupledLlgConfigs,applied_answer,current_answer,alpha_answer,num_batch)
        self.system_test.normalization(self.odeConfigs["stepSize"])        

        self.stepCount = math.ceil(iterationTime/self.odeConfigs["stepSize"])
        self.integration_time = torch.tensor([0, iterationTime])
        self.integration_time2 = torch.tensor(torch.arange(0.0, iterationTime + self.odeConfigs["stepSize"], self.odeConfigs["stepSize"]))
    #normalize
        self.integration_time_dynamics = torch.tensor([0, self.odeConfigs["stepSize"]])
        self.integration_time_dynamics_initial = torch.tensor([0, 100*iterationTime])
        self.integration_time *= self.system.gyro * self.system.fieldNormalizationFactor 
        self.integration_time2 *= self.system.gyro * self.system.fieldNormalizationFactor 
        self.integration_time_dynamics *= self.system.gyro * self.system.fieldNormalizationFactor 
        self.integration_time_dynamics_initial *= self.system.gyro * self.system.fieldNormalizationFactor 
        self.averageNorm = torch.empty(1)

        # Instances of class
        self.fc1 = nn.Linear(input_size, self.system.stoCount*3)
        #self.fc1.weight.requires_grad = False
        self.fc2 = nn.Linear(self.system.stoCount*3, output_size)

    def answer(self, x):
        if self.integration_time[-1] > 0:
            x = odeint(self.system_test, x,self.integration_time_dynamics, method='rk4', options=dict(step_size=self.odeConfigs["stepSize"]*self.system.gyro * self.system.fieldNormalizationFactor))
            x = x[-1].squeeze()
            x = x.unsqueeze(0)
            x = torch.transpose(x, 0, 1)
        return x
    
    def forward_train(self, x):
        if self.integration_time[-1] > 0:
            x = odeint(self.system, x,self.integration_time_dynamics, method='rk4', options=dict(step_size=self.odeConfigs["stepSize"]*self.system.gyro * self.system.fieldNormalizationFactor))
            x = x[-1].squeeze()
            x = x.unsqueeze(0)
            x = torch.transpose(x, 0, 1)
        return x
    
    def forward_initial(self, x):
        if self.integration_time[-1] > 0:
            x = odeint(self.system_test, x,self.integration_time_dynamics_initial, method='rk4', options=dict(step_size=self.odeConfigs["stepSize"]*self.system.gyro * self.system.fieldNormalizationFactor))
            x = x[-1].squeeze()
            x = x.unsqueeze(0)
            x = torch.transpose(x, 0, 1)
        return x
    

    def forward(self, x):
        if self.integration_time[-1] > 0:
            #NeuralODE(STO)
            x = odeint(self.system, x, self.integration_time, method='rk4', options=dict(step_size=self.odeConfigs["stepSize"]*self.system.gyro * self.system.fieldNormalizationFactor))
            x = x[-1].squeeze()
            x = x.unsqueeze(0)
            x = torch.transpose(x, 0, 1)
        return x




def spinNeuralOde(commandLineConfigs):
    
    os.makedirs("result", exist_ok=True)
    os.makedirs("result/weight", exist_ok=True)
    os.makedirs("result/state", exist_ok=True)
    os.makedirs("figure", exist_ok=True)
    os.makedirs("figure/timeSeries", exist_ok=True)
    os.makedirs("figure/table", exist_ok=True)

    iteration = [1.0e-9]
    losses = np.zeros(len(iteration))

    #logging
    logFileName = common.RESULT_DIR_NAME + '/status'
    for key, value in commandLineConfigs.items():
        logFileName += "_" + str(key) + "=" + str(value)
    logFileName += ".log"
        
    logging.basicConfig(filename=logFileName, level=logging.INFO, format=common.LOG_FORMAT)

    for _ in range(1): # for loop
        #iterate through the iteration time
        for t in range(len(iteration)):
            trainingFileName = common.RESULT_DIR_NAME + '/training'
            for key, value in commandLineConfigs.items():
                trainingFileName += "_" + str(key) + "=" + str(value)        
            trainingFileName += "_iteration=" + str(iteration[t])
            trainingFileName += ".txt"

            losses[t] = 10.0         

            initialTime = time.time()
            logging.info("Start cluculation: " + str(datetime.datetime.now()) + " Iteration: " + str(iteration[t]) + ", device: " + device)

            for epoch_in_epoch in range(1): 
                if (epoch_in_epoch == 0):              
                    appliedField_change_numpy = applied_min+0.5*applied_range*np.ones((num_batch,1,1))
                    appliedField_change = torch.nn.Parameter(torch.from_numpy(appliedField_change_numpy.astype(np.float64)).clone())
                    current_change_numpy = current_min+0.5*current_range*np.ones((num_batch,1,1))
                    current_change = torch.nn.Parameter(torch.from_numpy(current_change_numpy.astype(np.float64)).clone())           
                    alpha_change_numpy = alpha_min+0.5*alpha_range*np.ones((num_batch,1,1))
                    alpha_change = torch.nn.Parameter(torch.from_numpy(alpha_change_numpy.astype(np.float64)).clone())


                model = ODEBlock(image_size, 10, iteration[t], commandLineConfigs, appliedField_change,current_change,alpha_change).to(device)


                if (epoch_in_epoch == 0):
                    optimizer = torch.optim.Adam(params=[
                        {"params": model.appliedField_change, "lr": (applied_range/(10))},
                        #{"params": model.current_change, "lr": (current_range/(10))},
                        #"params": model.alpha_change, "lr": (alpha_range/(10))},
                    ])


                appliedField_change_list = [appliedField_change_numpy[0]]
                current_change_list = [current_change_numpy[0]]
                alpha_change_list = [alpha_change_numpy[0]]
                accuracy_x_list = []
                answer_applied_list=[]
                answer_current_list=[]
                answer_alpha_list=[]
            
                for epoch in range(num_epochs): 
                    loss_batch=torch.zeros(num_batch)

                    loss_sum = 0    

                    loss=0
                    correct=0
                    int_loss=0
                    torch_fix_seed()
                    x_train = 2*torch.rand(num_batch, model.system.stoCount, 3)-1
                    x_train[:,:,2] = torch.abs(x_train[:,:,2])
                    x_train = x_train / torch.norm(x_train,dim=2).reshape(num_batch, model.system.stoCount,1) #normalize

                    if ((epoch_in_epoch ==0 )&(epoch==0)):
                        x_answer_tmp = x_train 
                    if ((epoch_in_epoch ==0 )&(epoch==0)):
                        with torch.no_grad():
                            x_answer_tmp = model.forward_initial(x_answer_tmp)
     
                    optimizer.zero_grad()
                    model.train()  

                    x_train_list_0= []
                    x_train_list_1= []
                    x_train_list_2= []
                    x_train_list_norm= []
                    if ((epoch_in_epoch ==0 )&(epoch==0)):
                        x_answer_list_0 = []
                        x_answer_list_1 = []
                        x_answer_list_2 = []

                    x_answer = x_answer_tmp 
                    x_train = x_answer_tmp
                    
                    if ((epoch_in_epoch ==0 )&(epoch==0)):
                        x_answer_t = torch.zeros((model.stepCount,x_answer.to('cpu').detach().numpy().copy().shape[0],x_answer.to('cpu').detach().numpy().copy().shape[1],x_answer.to('cpu').detach().numpy().copy().shape[2]))
                    

                    for t_in in range(model.stepCount): 
                        x_train = model.forward_train(x_train)
                        if ((epoch_in_epoch ==0 )&(epoch==0)):
                            x_answer = model.answer(x_answer)
                            x_answer_t[t_in,:] = x_answer
                        if (t_in > int(model.stepCount/2)):
                            if (np.mod(t_in, 100)==0):
                                loss_batch[:] = loss_batch[:] + torch.abs(x_train[:,0,0]-x_answer_t[t_in,:,0,0])
                                int_loss = int_loss + 1

                        x_train_list_0.append(x_train[0,0,0].to('cpu').detach().numpy().copy().tolist())
                        x_train_list_1.append(x_train[0,0,1].to('cpu').detach().numpy().copy().tolist())
                        x_train_list_2.append(x_train[0,0,2].to('cpu').detach().numpy().copy().tolist())
                        x_train_list_norm.append(np.sqrt(x_train[0,0,0].item()**2+x_train[0,0,1].item()**2+x_train[0,0,2].item()**2))
                        if ((epoch_in_epoch ==0 )&(epoch==0)):
                            x_answer_list_0.append(x_answer[0,0,0].item())
                            x_answer_list_1.append(x_answer[0,0,1].item())
                            x_answer_list_2.append(x_answer[0,0,2].item())    
                        
                    loss_batch = loss_batch/model.system.stoCount
                    loss_batch = loss_batch/int_loss
                    loss = torch.mean(loss_batch)

                    correct = correct/model.system.stoCount
                    correct = correct/int_loss

                    i_min = torch.argmin(loss_batch)

                    loss_sum += loss

                    if (epoch == 0):
                        loss.backward()
                        optimizer.step()
                    if (epoch > 0): 
                        if (loss_before < loss):
                            #print("Q",epoch_in_epoch,epoch)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = 0.5*param_group['lr']
                        else:
                            loss.backward()
                            optimizer.step()

                    appliedField_change_list.append(appliedField_change[i_min].to('cpu').detach().numpy().copy())
                    current_change_list.append(current_change[i_min].to('cpu').detach().numpy().copy())
                    alpha_change_list.append(alpha_change[i_min].to('cpu').detach().numpy().copy())

                    loss_before = loss
                    i_min = torch.argmin(loss_batch)
                    
                    accuracy_x = loss.item()
                    accuracy_x_list.append(accuracy_x)
                    answer_applied_list.append(applied_answer_tmp)
                    answer_current_list.append(current_answer_tmp)
                    answer_alpha_list.append(alpha_answer_tmp)