# %%
import random
import numpy as np
# import matplotlib.pylab  as plt
# import matplotlib as mpl
import fwi
import torch
import time 
# from skimage.transform import resize
# import m8r as sf
import sys 
import os
from scipy import signal

# ========================== Functions  ============================== #


def freq_filter(freq, wavelet,btype,fs):
    """
    Filter out low frequency

    Parameters
    ----------
    freq : :obj:`int` or `array in case of bandpass `
    Cut-off frequency
    wavelet : :obj:`torch.Tensor`
    Tensor of wavelet
    btype : obj: 'str'
    Filter type  
    dt : :obj:`float32`
    Time sampling
    Returns
    -------
    : :obj:`torch.Tensor`
    Tensor of highpass frequency wavelet
    """


    ''' AA: Added 6/fs to prevent frequency leak. 
    The argument (2 * freq /fs) is from the definition of the filter signal.butter, 
    chek the value of Wn in the definition of signal.butter. 
    I manually added/subtract 6/fs to prevent leak. The number was selected based on trial and error and plotting the spectrum
    '''
    if btype == 'hp': sos = signal.butter(4,  6/fs + 2 * freq /fs, 'hp', output='sos') 
    if btype == 'lp': sos = signal.butter(4,   2 * freq /fs - 6/fs , 'lp', output='sos') 
    if btype == 'bp': sos = signal.butter(4,  [2/fs + 2 * freq[0] /fs,  2 * freq[1] /fs - 2/fs ], 
                            'bp', output='sos') 
    return torch.tensor( signal.sosfiltfilt(sos, wavelet,axis=0).copy(),dtype=torch.float32)


# ============================ setting global parameters =============================#

# Define the model and achuisition parameters
par = {'nx':801,   'dx':0.02, 'ox':0,
       'nz':200,   'dz':0.02, 'oz':0,
       'ns':1,   'ds':0.08,   'osou':8,  'sz':0.06,
       'nr':400,   'dr':0.04,  'orec':0, 'rz':0.06,
#        'nt':2500,  'dt':0.002,  'ot':0,
       'nt':1250,  'dt':0.004,  'ot':0,
       'freq':5,
        'FWI_itr': 300,
       'num_dims':2
      }



par['mxoffset']= 6 
par['nr'] = int((2 * par['mxoffset'])//(par['dr'])) + 1  
par['ds'] = np.round((par['nx']*par['dx'] - 2 * par['mxoffset']  )/par['ns'],3)

par['orec'] = par['osou'] - par['mxoffset']
   
fs = 1/par['dt'] # sampling frequency
par ['num_batches'] = 1
 
# Don't change the below two lines 
num_sources_per_shot=1

# Mapping the par dictionary to variables 
for k in par:
    locals()[k] = par[k]

alphatv=0.01

num_models=20
device = torch.device('cuda:0')

path = './output/samples/'
if not os.path.exists(path): os.makedirs(path) 

# command line Arg 
# istart=int(sys.argv[1]) * int(sys.argv[2]) 
# num_models=int (sys.argv[2])
istart= 0
num_models=5



device = torch.device('cuda:0')
path = './output_ibx_copy/'

if not os.path.exists(path):
    os.makedirs(path) 



# %% 
# ====================================== Inversion ======================================= #
start = time.time()
inversion = fwi.fwi(par,1)

   
# ==========wavelet ========#
wavel = inversion.Ricker(freq)  #source will be repeated as many shots
# filter frequency
wavel_f = freq_filter(freq=[3,7],wavelet=wavel,btype='bp',fs=fs)




# Forward modeling 
data = torch.zeros((nt,ns,nr),dtype=torch.float32)

inverted_models = torch.zeros((nz,nx),dtype=torch.float32)
count=1


save_inv = []
save_init = []
save_true = []
for i in  range(num_models): 
     #if os.path.isfile(path+'pred1_m'+str(istart+count)+'.npy'): continue 
     print("MODEL NUMBER: ", i)
     # load the models 
     true_m = np.load(path+'true_m'+str(istart+count)+'.npy')
     init_m = np.load(path+'pred2_m'+str(istart+count)+'.npy').ravel()
     wb = np.array(true_m[true_m==1.5].shape[0])
     # convert to 2D models 

     true_m = np.repeat(true_m,nx,axis=0)
     true_m = true_m.reshape(nz,nx)

     init_m = np.repeat(init_m,nx,axis=0)
     init_m = init_m.reshape(nz,nx)
     # convert model to tensors
     true_m = torch.tensor(true_m,dtype=torch.float32)
     init_m = torch.tensor(init_m,dtype=torch.float32) 


     data  = inversion.forward_modelling(true_m,wavel_f.repeat(1,ns,num_sources_per_shot),device) 


     alphatv = np.random.choice(np.array([0.4]))
     inverted_models  = inversion.run_inversion (init_m,
                        data,wavel_f,wb,FWI_itr,device,tv_flag=True,alphatv=alphatv) 

     save_inv.append(inverted_models[:,:,0])
     save_init.append(init_m[:,0])
     save_true.append(true_m[:,0])
    #  np.save(path+'inv_2nd_m'+str(istart+count),inverted_models[:,:,0])

     count += 1 


end = time.time()
print ('Running time  for all the inversino is :', (end-start)/60, ' minutes')
print (f"Number of generated file is {count-1}")

print(' ---------- Done yaaaay')


# %% 
import matplotlib.pylab as plt
for i in range(len(save_inv)):
    plt.figure()
    plt.plot(save_inv[i][0,:],label='inversion')
    plt.plot(save_init[i][:],label='init')
    plt.plot(save_true[i][:],label='init')
    plt.legend()
# %%
