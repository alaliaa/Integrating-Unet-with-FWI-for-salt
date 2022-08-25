# %%
import random
# import deepwave
import numpy as np
import numpy.random as random2
import matplotlib.pylab  as plt
import matplotlib as mpl
from  scipy.ndimage import gaussian_filter
from scipy import signal
# import fwi
import fwiLBFGS as fwi 
# import fwi2 as fwi
import torch
import time 
from model_generator import   create_random_model2, create_random_model3
from skimage.transform import resize
from util import load_2drsf_data
import os 

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
    Time samplingF
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


def Plot_model(m,par,cmap='jet',**kwargs):
    font = {
        'weight' : 'bold',
        'size'   : 14}
    mpl.rc('font', **font)
    vmax = kwargs.pop('vmax', None)
    vmin = kwargs.pop('vmin', None)
    name = kwargs.pop('name', None)
    figsize = kwargs.pop('figsize', (10,3))
    title =  kwargs.pop('title', None)
    if 'vmin'==None: vmin, _ = np.percentile(m,[2,98])
    if 'vmax'==None: _, vmax = np.percentile(m,[2,98])
    plt.figure(figsize=figsize)
    plt.imshow(m,cmap=cmap,vmin=vmin,vmax=vmax,extent=[par['ox'],par['dx']*par['nx'],par['nz']*par['dz'],par['oz']])
    plt.axis('tight')
    plt.xlabel('Distance (km)',fontsize=14,weight='heavy')
    plt.ylabel('Depth (km)',fontsize=14,weight='heavy')
    plt.colorbar(label='km/s',shrink=0.8,pad=0.01)
    if title != None: plt.title(title)
    if name!=None:
        if not os.path.isdir('./Fig'): os.mkdir('./Fig')
        plt.savefig('./Fig/'+name,bbox_inches='tight')
        print('Figure is saved ')
    plt.show(block=False)


def plot_models1D(model,init,i,j):
    n=i*j
    font = {
        'weight' : 'bold',
        'size'   : 12}
    mpl.rc('font', **font)
    indx = []
    f = plt.figure(figsize=(12,25))
    for k in range(1,n+1):    
        ax = f.add_subplot(i,j,k)
        m  = random.randint(0,num_models-1)
        im =ax.plot(model[m,:],label='True model',linewidth=5)
        im =ax.plot(init[m,:],label='initial fwi',linewidth=5)
        plt.axis('tight')
        plt.legend(prop={'size':12, 'weight':'bold'},loc='upper left') 
    plt.savefig('models')
    # plt.show()
    plt.close()

def mask(m,value):
    """
    Return a mask for the model (m) using the (value)
    """
#     msk = m > value
#     msk = msk.astype(int)
    
    msk = np.ones_like(m)
    for im in range (m.shape[0]):
        for ix in range(m.shape[2]):
            for iz in range (m.shape[1]):
                if m[im,iz,ix]<=value : 
                    msk[im,iz,ix] = 0
                else: break

    return msk
    

# %% 
# setting parameters

mpl.rcParams['image.cmap'] ='seismic'



# Define the model and achuisition parameters
par = {     'nx':1685,   'dx':0.02, 'ox':0,
            'nz':201,   'dz':0.02, 'oz':0,
            'ns':400,   'ds':0.0825,   'osou':0,  'sz':0.06,
            # 'ns':60,   'ds':0.33,   'osou':7,  'sz':0.06,
            # 'ns':1,   'ds':.4,   'osou':6,  'sz':0.06,
            'ns':1,   'ds':1,   'osou':16,  'sz':0.06,
            'nr':842,'dr':0.04,  'orec':0,    'rz':0.06,
            # 'nt':4000,  'dt':0.002,  'ot':0,
            'nt':1250,  'dt':0.004,  'ot':0,
#             'nt':1250,  'dt':0.004,  'ot':0,
#             'nt':625,  'dt':0.008,  'ot':0,
            'freq':5,
            'FWI_itr': 1,
            'num_dims':2
      }

# par['mxoffset']= 6 
# par['nr'] = int((2 * par['mxoffset'])//(par['dr'])) + 1  
par['osou'] = (par['nx']*par['dx'])/2
# par['orec'] = par['osou'] - par['mxoffset']



fs = 1/par['dt'] # sampling frequency
par ['num_batches'] = 1
 
# Don't change the below two lines 
num_sources_per_shot=1

# Mapping the par dictionary to variables 
for k in par:
    locals()[k] = par[k]

alphatv= 0
# alphatv= 0.0
par['alphatv']=alphatv
smth1 = 1e-7
smth2 = 1e-7
# smth1 = 0.0001
# smth2 = 0.0001
num_models=1
device = torch.device('cuda:0')

path = './output/samples/'
if not os.path.exists(path): os.makedirs(path) 

# path = './output/test/'
#istart=3181  # start naming the output files 
istart=0

# %%

# get 1D models 
Fvel = 'bp_full_fixed.npy'
vel =  np.load(Fvel)
vel =  vel.T
bp =  resize(vel,(vel.shape[0],nz))



# mtrue_bp = bp[900,:].copy() # profile from BP   
mtrue_bp = bp[500,:].copy() # profile from BP   
bp[ bp >= 4.4] = np.nan
# Compute mean and std
bp_mean = np.nanmean(bp,axis=0)
bp_std = np.nanstd(bp,axis=0)

models1D = np.zeros((num_models,nz))
initials1D = np.zeros((num_models,nz))
vflood = np.zeros((num_models,nz))
wb = np.zeros((num_models,1))
msk = np.zeros((num_models,nz,nx))

# for i in range(num_models):
#     layer = int(np.random.rand()*nz//4)
#     models1D[i,:], initials1D[i,:],vflood[i,:],wb[i,:] = create_random_model3(nz,bp_mean,bp_std)

# # # ============= bp profiles 
iz = np.where(mtrue_bp[:] > 1.5)[0][0]
initials1D[0,iz:]  = mtrue_bp[iz]
initials1D[0,:iz]  = 1.5
wb[0,] = iz
models1D[0,] = mtrue_bp
bp_mean = bp_mean.reshape(1,-1)
bp_mean = np.repeat(bp_mean,initials1D.shape[0],axis=0)
# # # ==============

# plot_models1D(models1D,initials1D,5,5)

# convert to 2D models 
models2D = np.repeat(models1D,nx,axis=1)
models2D = np.reshape(models2D,(num_models,nz,nx))

initials2D = np.repeat(initials1D,nx,axis=1)
initials2D = np.reshape(initials2D,(num_models,nz,nx))


msk = mask(models2D,1.5)

#plot_models2D(models2D,initials2D,5,5)



# convert model to tensors
true_m = torch.tensor(models2D,dtype=torch.float32)
init_m = torch.tensor(initials2D,dtype=torch.float32) 



# %%
# inversion block
inversion = fwi.fwi(par,1)


wavel = inversion.Ricker(freq)  #source will be repeated as many shots

# plt.magnitude_spectrum(wavel[:,0,0],fs)
# plt.show()
# plt.figure(figsize=(10,5))
# plt.magnitude_spectrum(wavel_f[:,0,0],fs)
# plt.hlines(0,0,15,color='k')
# plt.vlines(3,0,0.0006,color='k')
# plt.vlines(7,0,0.0006,color='k')
# plt.xlim([0,15])
# plt.xticks(np.arange(0, 15, 1))
# plt.show()

# forward modelling



# %% 
data = torch.zeros((nt,ns,nr),dtype=torch.float32)
inverted_models = torch.zeros((nz,nx),dtype=torch.float32)
count=1
inv_m= []
start = time.time()
for i in  range(num_models): 
# for i in  range(4,5): 
     print("MODEL NUMBER: ", i)

     data  = inversion.forward_modelling(true_m[i,:],wavel.repeat(1,ns,num_sources_per_shot),device) 

    #  # filter frequency
     wavel_f = freq_filter(freq=[3,7],wavelet=wavel,btype='bp',fs=fs)
     data_f =  freq_filter(freq=[3,7],wavelet=data,btype='bp',fs=fs)

     inverted_models, losses  = inversion.run_inversion (init_m[i,:],
                        data_f,wavel_f.repeat(1,ns,1),msk[i,:],FWI_itr,device,
                        tv_flag=True,alphatv=alphatv,
                        smth_flag=False,smth=[smth1,smth2],method='1Dss') 



     np.save(path+'true_m'+str(istart+count),true_m[i,:,0].cpu().numpy())
     np.save(path+'init_m'+str(istart+count),init_m[i,:,0].cpu().numpy())
     np.save(path+'inv_m'+str(istart+count),inverted_models[:,:,0])
     count += 1 
     inv_m.append(inverted_models)
end = time.time()
print ('Running time  for all the inversino is :', (end-start)/60, ' minutes')
print (f"Number of generated file is {count-1}")

print(' ---------- Done yaaaay')



# %% 
sample_idx = 11
inv_m_sample = np.array(inv_m[sample_idx])[-1,:,:] 
def plot_models1D(model,init,inv,vflood,i,j):
    n=i*j
    font = {
        'weight' : 'bold',
        'size'   : 15}
    mpl.rc('font', **font)
    indx = []
    f = plt.figure(figsize=(40,25))
    for k in range(1,n+1):    
        ax = f.add_subplot(i,j,k)
        # m  = random.randint(0,num_models-1)
        im =ax.plot(model,label='True model',linewidth=5)
        im =ax.plot(init,label='initial fwi',linewidth=5)
        im =ax.plot(inv,label='inverted fwi',linewidth=5)
        im =ax.plot(vflood,'--',label='flood fwi',linewidth=5)
        plt.axis('tight')
        plt.legend(prop={'size':15, 'weight':'bold'},loc='upper left') 
    # plt.savefig('models')
    plt.show()
    plt.close()
plot_models1D(true_m[sample_idx,:,5],init_m[sample_idx,:,5],inv_m_sample[:,5],vflood[sample_idx,],1,1)


# inverted_models = inverted_models.numpy()
# for i in range (num_models):
#     plt.figure()
#     plt.plot(true_m[i,:,5].numpy(),label='true')     
#     plt.plot(init_m[i,:,5].numpy(),label='initial')
#     plt.plot(save_models[i,:,5],label='inv')
#     plt.legend()
#     plt.show()

# %% 
inv_m_sample = np.array(inv_m[-1])[-1,:,:] 

par['nz'] = 150 - 20
plt.figure(figsize=(10,10))
# plt.imshow(inv_m[0]- init_m[0].numpy()
#           ,extent=[par['ox'],par['dx']*par['nx'],par['nz']*par['dz'],par['oz']])
plt.imshow(inv_m_sample[20:150,]- init_m[0,20:150,].numpy()
          ,extent=[par['ox'],par['dx']*par['nx'],par['nz']*par['dz'],par['oz']],
          vmin=-1e-6,vmax=1e-6)
plt.xlabel("Position (km)",fontsize=13)
plt.ylabel("Depth (km)",fontsize=13)
plt.vlines(700*0.02,0,par['nz']*0.02,color='k')
plt.scatter(inversion.r_cor[0,:,1],inversion.r_cor[0,:,0],marker='.',s=0.3,c='r')
plt.scatter(inversion.s_cor[:,0,1],inversion.s_cor[:,0,0],marker='x',s=50,c='b')
plt.axis('tight')
plt.savefig('1-shot.eps',format='eps')

plt.figure(figsize=(5,10))
plt.plot(inv_m_sample[20:150,700]- init_m[0,20:150,700].numpy(),np.arange(0,init_m[0,20:150,700].shape[0])*0.02,linewidth=3)
plt.gca().invert_yaxis()# plt.plot(true_m[0,:,800].numpy())
plt.savefig('1-shot-profile.eps',format='eps')

# %%
Plot_model(inv_m[-1,],par,vmin=1.5,vmax=4.5)

plt.figure()
plt.plot(inv_m[-1,:,900])
plt.plot(true_m[0,:,900])
# %%
