#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
# import fwi
import fwiLBFGS as fwi
import torch
import time 
import matplotlib.pylab  as plt
from util import * 
import deepwave
from scipy import signal
from skimage.transform import resize, rescale 
import horovod.torch as hvd
from torch.utils.data import TensorDataset, DataLoader



    
# =================================================================== #


# ========================== Main  ============================== #
hvd.init()  # initilaize
torch.cuda.set_device(hvd.local_rank())
device = torch.device(f'cuda:{hvd.local_rank()}')
# ============================ setting parameters =============================#

# Define the model and achuisition parameters
par = {     'nx':1685,   'dx':0.02, 'ox':0,
            'nz':201,   'dz':0.02, 'oz':0,
            # 'ns':200,   'ds':0.165,   'osou':0,  'sz':0.06,
            # 'ns':400,   'ds':0.082675,   'osou':0,  'sz':0.06,
             'ns':60,   'ds':0.33,   'osou':7,  'sz':0.06,
            'nr':674,   'dr':0.04,  'orec':0,    'rz':0.06,
            'nt':2500,  'dt':0.002,  'ot':0,
#             'nt':1250,  'dt':0.004,  'ot':0,
#             'nt':625,  'dt':0.008,  'ot':0,
            'freq':10,
            'FWI_itr': 50,
            'num_dims':2
      }


par['mxoffset']=6
par['nr'] = int((2 * par['mxoffset'])//(par['dr'])) +1  
# par['osou'] = 0
par['orec'] = par['osou'] - par['mxoffset']  
fs = 1/par['dt'] # sampling frequency
par ['batch_size'] = 4
par ['num_batches'] = par['ns']//par ['batch_size'] 
# Don't change the below two lines 
num_sources_per_shot=1


# ============================ I/O =============================#
path = './'
velocity_path = './velocity/'
# i/o files
vel_true =velocity_path+'bp_full_fixed.npy' # true model 

fwi_pass= 1
minF = 3
maxF = 7
TV_ALPHA = 0.5
# TV_ALPHA = 0.0
smth1 = 1e-17
smth2 = 1e-17
# smth1 = 10
# smth2 = 15
# opt='SGD'
# opt='Adam'
opt='LBFGS'

# inv_file=f"BPfull_1stinv_TV{TV_ALPHA}_offs{par['mxoffset']}_DomFreq{par['freq']}_MinFreq{minF}_MaxFre{maxF}_fwi{fwi_pass}_smth{smth1}-{smth2}"


inv_file=f"BPfull_1stinv_TV{TV_ALPHA}_offs{par['mxoffset']}_DomFreq{par['freq']}_MinFreq{minF}_MaxFre{maxF}_fwi{fwi_pass}_smth{smth1}-{smth2}_nt{par['nt']}-ns{par['ns']}-opt{opt}-1Dtemp"


# inv_file=f"BPfull_1stinv_TV{TV_ALPHA}_offs{par['mxoffset']}_DomFreq{par['freq']}_MinFreq{minF}_MaxFre{maxF}_fwi{fwi_pass}_smth{smth1}-{smth2}_nt{par['nt']}-ns{par['ns']}-opt{opt}-BP"

output_file = velocity_path+inv_file
    
mtrue = np.load(vel_true)
# mtrue = mtrue.T



# Mapping the par dictionary to variables 
for k in par:
    locals()[k] = par[k]

  

    


mtrue = np.load(vel_true)

# In[3]:


# ============================ Forward modelling =============================#
# convert to tensor

mtrue = torch.tensor(mtrue,dtype=torch.float32)
mtrue = np.repeat(mtrue[:,150].reshape(nz,-1),axis=1,repeats=nx) 

# initiate the fwi class
inversion = fwi.fwi(par,2)


wavel = inversion.Ricker(freq)  
data = torch.zeros((nt,ns,nr),dtype=torch.float32)
data = inversion.forward_modelling(mtrue,wavel.repeat(1,ns,1),device).cpu()


torch.cuda.empty_cache()


# In[4]:

# filter frequencies 
if maxF==100:
    wavel_f = freq_filter(freq=minF,wavelet=wavel,btype='hp',fs=fs)
    data_f = freq_filter(freq=minF,wavelet=data,btype='hp',fs=fs)   
else:    
    wavel_f = freq_filter(freq=[minF,maxF],wavelet=wavel,btype='bp',fs=fs)
    data_f = freq_filter(freq=[minF,maxF],wavelet=data,btype='bp',fs=fs)





# ========================= Cretae initial model =================== # 
# mask 


msk = mask(mtrue.numpy(),1.5)

# bp_mean = np.nanmean(mtrue,axis=1)
# bp_mean = bp_mean.reshape(-1,1)
# minit =  np.repeat(bp_mean,nx,axis=1)
# minit = minit * msk
# minit[minit==0] = 1.5


# constant init 
minit = msk.copy()
minit = minit.astype(np.float32())

for ix in range (nx):
    iz = np.where(minit[:,ix] > 0)[0][0]
    minit[iz:,ix]  = mtrue[iz,ix]
minit [minit == 0] = 1.5
#if hvd.local_rank()==0:
    #Plot_model(mtrue,par,title='Mtrue',name='1D-2D-profile')
    #Plot_model(minit,par,title='Minit',name='1D-2D-Initprofile')


# Convert to torch
minit = torch.tensor(minit,dtype=torch.float32)
# data_f = torch.tensor(data_f,dtype=torch.float32)
# wavel_f = torch.tensor(wavel_f,dtype=torch.float32)


# distributed sampler
data_f = data_f.permute(1,0,2) # change the shape for the sampler  [ns,nt,nr]
dataset = TensorDataset(data_f,inversion.s_cor,inversion.r_cor)

train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank(),shuffle=True)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=par['batch_size'], sampler=train_sampler) # batch size is the number of shots simulated simltonuously 



#  inversion 
minv,loss = inversion.run_inversion(
    minit,train_loader,wavel_f.repeat(1,ns,1),msk,FWI_itr,device,
    smth_flag=True,smth=[smth1,smth2],vmin=1.5,vmax=4.5,hvd_flag=True,
    alphatv=TV_ALPHA,plot_flag=False)

# In[13]:

if hvd.local_rank()==0:
    # save_2drsf(minv[-1,].T,par,f"{output_file}.rsf")
    np.save(output_file,minv)
    print(f"modeled saved in:  {output_file}")



# Plot loss 
if hvd.local_rank()==0:
    plt.plot(loss)
    if not os.path.exists('./Fig'):
        os.makedirs('./Fig')
        if not os.path.exists('./Fig/losses'):
            os.makedirs('./Fig/losses')
    plt.savefig(f'./Fig/losses/loss-{inv_file}.png')
    plt.show()
    plt.pause





