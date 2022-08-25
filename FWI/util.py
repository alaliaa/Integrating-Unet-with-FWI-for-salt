import numpy as np 
import matplotlib.pyplot as plt
import os 
import subprocess
import matplotlib as mpl
from scipy import signal
import torch


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




# Plot one shot gather
def Plot_shot(data,idx,par):
    
    vmin, vmax = np.percentile(data[:,idx].cpu().numpy(), [2,98])
    plt.figure()
    plt.imshow(data[:,idx].cpu().numpy(), aspect='auto',
           vmin=vmin, vmax=vmax,cmap='gray',extent=[par['orec']+idx*par['ds'],par['orec']+idx*par['ds']+par['dr']*par['nr'],
                                                    par['nt']*par['dt'],par['ot']])
    plt.ylabel('Time (s)')
    plt.xlabel('Distance (km)')


def cmd(command):
    """
    Run command and pipe what you would see in terminal into the output cell
    """
    print(command)
    process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    while True:
        output = process.stderr.readline().decode('utf-8')
        if output == '' and process.poll() is not None:
            # this prints the stdout in the end
            output2 = process.stdout.read().decode('utf-8')
            print(output2.strip())
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

   
def save_2drsf(model,par,path):
    
    binarf=path+'@'
    # save binary 
    model.astype('float32').tofile(binarf)
    cmd('''echo  ' # This is a Madagascar like header \n \n\t n1=%d \n \t n2=%d \n\t d1=%g \n \t d2=%g \n\t o1=%g \n\t o2=%g \n\t data_format=native_float \n\t in=%s' > %s  
        '''%(par['nz'],par['nx'],par['dz'],par['dx'],par['oz'],par['ox'],binarf,path))



def save_1drsf(model,par,path):

    if len(model.shape) == 1: model = model.reshape(1,-1)
    elif len(model.shape) == 2 and model.shape[0]>1: model = model.T

    binarf=path+'@'
    # save binary 
    model.astype(np.float32).tofile(binarf)
    cmd('''echo  ' # This is a Madagascar like header \n \n\t n1=%d \n\t d1=%g \n\t o1=%g \n\t data_format=native_float \n\t in=%s \n\t n2=1 \n\t n3=1' > %s  
        '''%(par['nt'],par['dt'],par['ot'],binarf,path))


    


def save_3ddata(model,par,path):
    
    assert model.shape == (par['ns'],par['nr'],par['nt']), "fix the shape of the data to be [ns,nr,nt]"
    binarf=path+'@'
    # save binary 
    model.astype(np.float32).tofile(binarf)
    cmd('''echo  ' # This is a Madagascar like header \n \n\t n1=%d \n\t n2=%d \n\t d1=%g \n\t d2=%g \n\t o1=%g \n\t o2=%g \n\t data_format=native_float \n\t
    \n\t n3=%d \n\t d3=%g \n\t o3=%g \n\t in=%s' > %s  
        '''%(par['nt'],par['nr'],par['dt'],par['dr'],par['ot'],par['orec'],par['ns'],par['ds'],par['os'],binarf,path))


    


def freq_filter(freq, wavelet,btype,fs):
    """
    Filter frequency

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


def mask(m,value):
    """
    Return a mask for the model (m) using the (value)
    """
#     msk = m > value
#     msk = msk.astype(int)
    
    msk = np.ones_like(m)
    for ix in range(m.shape[1]):
        for iz in range (m.shape[0]):
            if m[iz,ix]<=value : 
                msk[iz,ix] = 0
            else: break
    
    return msk
    