#ZZY 02/June/2021
import torch
import os
import numpy as np
from model import CNN
import numpy as np
from utils.parse_util import Record_Tensor_txt
from utils.parse_util_bin import Record_Tensor_bin
import struct

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=7)

#output bin
out_bin=1

np.set_printoptions(threshold=np.inf)
model = CNN()

param={} #print size of each layer
for name,parameters in model.state_dict().items():
    print(name,':',parameters.size())
    param[name]=parameters.cpu().detach().numpy()
#%%
#print all parameters
pth=r'your_model.pth'
model.load_state_dict(torch.load(pth))
for i in model.named_parameters():
    print(i)
#%% this module for Weights and bias 
filename_bin=r'./para_BN_W&b_bin/'
filename_txt=r'./para_BN_W&b_txt/'
def Params_Extractor(name,tensor):
  tensor_tmp = param[tensor]
  if out_bin==1:
      Record_Tensor_bin(filename_bin,tensor_tmp,name)
  else:
      Record_Tensor_txt(filename_txt,tensor_tmp,name)
#%% this module for BN parameters
filename_txt=r'./para_BN_W&b_txt/'
filename_bin=r'./para_BN_W&b_bin/'
def Scale_Shift (mean,var,gamma,beta,name_scale,name_shift,eps=0.00001):
    tensor_mean = param[mean]
    tensor_var = param[var]
    tensor_gamma = param[gamma]
    tensor_beta = param[beta]

    coeff_A=tensor_gamma/np.sqrt(tensor_var+eps)
    coeff_B=tensor_beta-(tensor_mean*tensor_gamma)/np.sqrt(tensor_var+eps)
    #-------Scale-------
    coeff_A = coeff_A.reshape(-1,1) #make tensor to vector
    if out_bin==1:
        f = open(filename_bin+name_scale+'.bin', 'wb')
        for i in range(np.shape(coeff_A)[0]):#Only one dimension
            a=struct.pack('f',coeff_A[i]) #convert list to string representation
            f.write(a)
    #Write to txt anyway
    coeff_A = np.array2string(coeff_A,separator='  ',suppress_small=True)
    coeff_A=coeff_A.replace('[','').replace(']',',')
    with open(filename_txt+name_scale+'.txt', 'w') as f:  
        f.write(coeff_A) 
    #-------Shift-------
    coeff_B = coeff_B.reshape(-1,1) #make tensor to vector
    if out_bin==1:
        f = open(filename_bin+name_shift+'.bin', 'wb')
        for i in range(np.shape(coeff_B)[0]): #Only one dimension
            a=struct.pack('f',coeff_B[i])
            f.write(a)
    #Write to txt anyway
    coeff_B = np.array2string(coeff_B,separator='  ',suppress_small=True)
    coeff_B=coeff_B.replace('[','').replace(']',',')
    with open(filename_txt+name_shift+'.txt', 'w') as f:  
        f.write(coeff_B) 

#%% Extract parameters in batch normalization
Params_Extractor("conv1_weight"        ,"conv1.weight")
Params_Extractor("conv1_bias"        ,"conv1.bias")

Params_Extractor("conv2_weight"        ,"conv2.weight")
Params_Extractor("conv2_bias"        ,"conv2.bias")

Params_Extractor("conv3_weight"        ,"conv3.weight")
Params_Extractor("conv3_bias"        ,"conv3.bias")

Params_Extractor("conv4_weight"        ,"conv4.weight")
Params_Extractor("conv4_bias"        ,"conv4.bias")


#%% ZZY 17/Jan/2022 
#Convert decimal to binary, and write to files
#get floating-point number first (in_folder) and convert them to fixed-point (out_folder)

in_folder=r'./para_BN_W&b_txt/'
out_folder=r'./para_W_extracted_txt/'#final results

# -*- coding: utf-8 -*-

