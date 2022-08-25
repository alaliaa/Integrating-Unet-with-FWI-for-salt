import time
import torch
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import deepwave
from  scipy.ndimage import gaussian_filter, gaussian_filter1d
import horovod.torch as hvd


class fwi():
    def __init__(self,par,acquisition):
       self.nx=par['nx']
       self.nz=par['nz']
       self.dx=par['dx']
       self.nt=par['nt']
       self.dt=par['dt']
       self.num_dims=par['num_dims']
       self.num_shots=par['ns']
       self.num_batches=par['num_batches']
       self.num_sources_per_shot=1
       self.num_receivers_per_shot = par['nr']
       self.ds= par['ds']
       self.dr= par['dr']
       self.sz = par['sz']
       self.rz = par['rz']
       self.os = par['osou']
       self.orec = par['orec']
       self.ox = par['ox']
       self.num_sources_per_shot=1

        
       self.s_cor, self.r_cor =self.get_coordinate(acquisition)


    def get_coordinate(self,mode):
       """ 
       Create arrays containing the source and receiver locations
       ------------------------------------------------------------
       Argumunts :
       ----------
       mode:  int: 1 or 2
           1: Offset is not specified and the recievers are the same for all shots 
           2: Recivers are designed to have specific offset (p.s. this should be done from the parameters) 
       
       output: 
       -------
        x_s: Source locations [num_shots, num_sources_per_shot, num_dimensions]
        x_r: Receiver locations [num_shots, num_receivers_per_shot, num_dimensions]
        Note: the depth is set to zero , to change it change the first element in the last dimensino 
        """
       x_s = torch.zeros(self.num_shots, self.num_sources_per_shot, self.num_dims)
       x_r = torch.zeros(self.num_shots, self.num_receivers_per_shot, self.num_dims)
       # x direction 
       x_s[:, 0, 1] = torch.arange(0,self.num_shots).float() * self.ds + self.os  
       # z direction  
       x_s[:, 0, 0] = self.sz
                        
       if mode ==1:
         # x direction 
         x_r[0, :, 1] = torch.arange(0,self.num_receivers_per_shot).float() * self.dr + self.orec
         x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1)
         # z direction 
         x_r[:, :, 0] = self.rz
            
            
       elif mode ==2: # fixed spread !! 
         # x direction 
         # for i in range (self.num_shots):
         #    orec =  i * self.ds
         #    x_r[i, :, 1] = torch.arange(0,self.num_receivers_per_shot* self.dr,self.dr).float() + orec 
         x_r[0,:,1] = torch.arange(self.num_receivers_per_shot).float() * self.dr   + self.orec
         x_r[:,:,1] = x_r[0,:,1].repeat(self.num_shots,1) + \
                  torch.arange(0,self.num_shots).repeat(self.num_receivers_per_shot,1).T * self.ds
         # z direction 
         x_r[:, :, 0] = self.rz

         # Avoid out-of-bound error   
         # xr_corr = x_r[:,:,1]
         # xr_corr [xr_corr < 0 ] = 0
         # xr_corr [xr_corr > (self.nx-2)*self.dx ] = (self.nx-2)*self.dx # ~last point in the model 
         # x_r[:, :, 1] =  xr_corr
        
        
 
       return x_s,x_r


    def Ricker(self,freq):
        wavelet = (deepwave.wavelets.ricker(freq, self.nt, self.dt, 1/freq)
                                 .reshape(-1, 1, 1))

                        
        return wavelet
    
    def forward_modelling(self,model,wavelet,device):
       # pml_width parameter control the boundry, for free surface first argument should be 0 
       prop = self.propagator(model,device)
       data = prop(wavelet.to(device), self.s_cor.to(device), self.r_cor.to(device), self.dt).cpu()
       return data

    def batch_forward_modelling(self,model,wavl,device,batch_size):
        
        batch_wavl = wavl[:,::batch_size,:].to(device)
        batch_x_s = self.s_cor[::batch_size].to(device)
        batch_x_r = self.r_cor[::batch_size].to(device)  
        prop = self.propagator(model,device)
        data = prop(batch_wavl, batch_x_s, batch_x_r, self.dt).cpu()
        return data
    
    
    def propagator(self,model,device):
        return deepwave.scalar.Propagator({'vp': model.to(device)}, self.dx,pml_width=[0,20,20,20,0,0]
                                         , survey_pad=[None,None,None,None])
    

    def run_inversion(self,model,data_t,wavelet,msk,niter,device=None,**kwargs): 
       """ 
      This run the FWI inversion,  
      ===================================
      Arguments: 
         model: torch.Tensor [nz.nx]: 
            Initial model for FWI 
         data_t: torch.Tensor [nt,ns,nr]: 
            Observed data
         wavelet: torch.Tensor [nt,1,1] or [nt,ns,1]
            wavelet 
         msk: torch.Tensor [nz,nx]:
            Mask for water layer
         niter: int: 
            Number of iteration 
         device: gpu or cpu  
       ==================================
      Optional: 
         vmin: int:
            upper bound for the update 
         vmax: int: 
            lower bound for the update 
         smth_flag: bool: 
            smoothin the gradient flag 
         smth: sequence of tuble or list: 
            each element define the amount of smoothing for different axes
         hvd: bool: 
            Use horovod for multi-GPU implimintation 
         tv_flag: bool: (Removed )
             Flag for adding TV reg
         alphatv:  float:
             Tv coefficient 
         plot_flag: bool 
             Excute plotting command for gradients and model updates
         Method: string ("1D" or None)         
       """

       # Defining parameters 
       model = model.to(device)
       wavelet = wavelet.to(device)
       msk = torch.from_numpy(msk).int().to(device)
       model.requires_grad=True 
       m_max = kwargs.pop('vmax', 4.5)
       m_min = kwargs.pop('vmin', 1.5)
       smth_flag = kwargs.pop('smth_flag', False)
       hvd_flag = kwargs.pop('hvd_flag', False)
       alphatv = kwargs.pop('alphatv', 0)
       tv_flag =  alphatv !=0 
       plot_flag = kwargs.pop('plot_flag', False)           
       Method = kwargs.pop('Method', None)           
       if smth_flag: 
          smth = kwargs.pop('smth', '')
          assert smth != '', " 'smth' is not specified "
        
        
        
       # Defining objective and step-length  
       criterion = torch.nn.MSELoss()
       LR = 0.5
#        alpha = torch.nn.Parameter(torch.tensor([0.01]))
       optimizer = torch.optim.Adam([{'params':[model],'lr':LR}])
#        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=20,threshold=1e-3,verbose=False)		
       # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,niter, verbose=True)

       num_batches = self.num_batches
       num_shots_per_batch = int(self.num_shots / num_batches)
       prop = self.propagator(model,device)
       t_start = time.time()
       loss_iter=[]
       increase = 0
       min_loss = 0
       first = True
       tol = 1e-5
   
       # updates is the output file
       updates=[]
       relax = 10
        # main inversion loop 
        
       if hvd_flag: 
           for itr in range(niter):
              running_loss = 0 
              optimizer.zero_grad()
              for ibatch,(batch_data_t,batch_x_s,batch_x_r) in enumerate(data_t): 
                 shot_indices =  ( torch.round ( (batch_x_s[:, 0, 1]  - self.os  + self.ox )/ self.ds )).int().numpy()
                 if shot_indices.shape[0] > 1:  
                    batch_wavl = wavelet[:, shot_indices, 0].view(wavelet.shape[0],shot_indices.shape[0],1)
                 else: 
                       batch_wavl = wavelet[:, shot_indices, 0].view(wavelet.shape[0],1,1)  
                 batch_data_t = batch_data_t.cuda()
                 batch_data_t = batch_data_t.permute(1,0,2)
                 batch_x_s    = batch_x_s.cuda()
                 batch_x_r    = batch_x_r.cuda()
                 batch_data_pred = prop(batch_wavl, batch_x_s, batch_x_r, self.dt)
                 loss = criterion(batch_data_pred, batch_data_t) 
                 loss.backward()
                 running_loss += loss.item()  
              running_loss = self.metric_average(running_loss, 'avg_loss') 
              model.grad = hvd.allreduce(model.grad.contiguous())      
              if smth_flag: model.grad = self.grad_smooth(model,smth[1],smth[0]).to(device)            
              if itr == 0 : gmax0 = (torch.abs(model.grad)).max() # get max of first itr 
          # model.grad = model.grad / gmax0   # normalize                 
              model.grad =  self.grad_reg(model,mask=msk)        
              if tv_flag: 
                   lossTV = tv_loss(model)
                   tv_grad = torch.autograd.grad(lossTV,model)[0]                 
                   tv_grad = tv_grad/(torch.abs(tv_grad)).max() * msk # normalize 
                   model.grad = (model.grad + alphatv * tv_grad)   # combine the gradient of the two 
              model.grad =  self.grad_reg(model,mask=msk)
            
#               LR = self.line_search(running_loss,model,wavelet,data_t,criterion,True,m_min,m_max,device)
# #               # update the learning rate in opt 
#               for param_group in optimizer.param_groups:
#                     param_group['lr'] = LR            
            
              optimizer.step()   

              # model.add(model.grad,alpha= -LR)

    
            # scheduler.step()
              model.data[model.data < m_min] = m_min
              model.data[model.data > m_max] = m_max
              loss_iter.append(running_loss)
              if hvd.local_rank()==0: 
                    print('Iteration: ', itr, 'Objective: ', running_loss)
                    if itr%1==0: 
                        updates.append(model.detach().clone().cpu().numpy())       
                        np.save('update_progress',updates)
                        print("save update")                    
              if plot_flag and itr%1==0 and hvd.local_rank()==0:
                   plt.figure(figsize=(10,5))
                   plt.imshow(model.grad.cpu().numpy(),cmap='seismic',vmin=-0.3,vmax=0.3)
                   plt.colorbar(shrink=0.3)
                   plt.show()

    #                plt.close()
                   plt.figure(figsize=(10,5))
                   plt.imshow(model.detach().clone().cpu().numpy(),cmap='jet',vmax=4.5,vmin=1.5)
                   plt.colorbar(shrink=0.3)
                   plt.show()
    #                plt.close()
                   
              # stopping criteria or relax condition smoothing ()
              if np.abs(loss_iter[itr] - loss_iter[itr-1])/max(loss_iter[itr],loss_iter[itr-1]) < tol and itr>20: 
#                   if smth_flag or tv_flag: 
                  if relax < 5:
                        if smth_flag: 
                            smth[0]=smth[0]/2
                            smth[1]=smth[1]/2
                            increase =5
                        if tv_flag :
                            alphatv = alphatv/2
                            increase =5
                        relax += 1     
                        if hvd.local_rank()==0: print(f"Reduce the smoothing from {smth[0]*2},{smth[1]*2} to {smth[0]},{smth[1]}")
                  else:
                      t_end = time.time()
                      if hvd.local_rank()==0: print('Runtime in min :',(t_end-t_start)/60)  
                      return np.array(updates),loss_iter
              # early stopping       
              elif min_loss < loss_iter[itr] and itr > 20: 
                 increase +=1
              else: 
                 increase = 0
                 min_loss = loss_iter[itr] 
              if  increase == 10: 
#                   if smth_flag or tv_flag: relax +=1 
                  if relax < 5:
                        if smth_flag: 
                            smth[0]=smth[0]/2
                            smth[1]=smth[1]/2
                            increase =5
                        if tv_flag :
                            alphatv = alphatv/2
                            increase =5
                        relax += 1     
                        if hvd.local_rank()==0: print(f"Reduce the smoothing from {smth[0]*2},{smth[1]*2} to {smth[0]},{smth[1]}")
                  else:
                      t_end = time.time()
                      if hvd.local_rank()==0:  print('Runtime in min :',(t_end-t_start)/60)  
                      return np.array(updates),loss_iter
                        
       # If not hvd
       else: 
           for itr in range(niter):
              running_loss = 0 
              optimizer.zero_grad()
              for it in range(num_batches): # loop over shots 
                 # batch_wavl = wavelet.repeat(1, num_shots_per_batch, 1)
                 batch_wavl = wavelet[:,it::num_batches]
                 batch_data_t = data_t[:,it::num_batches].to(device)
                 batch_x_s = self.s_cor[it::num_batches].to(device)
                 batch_x_r = self.r_cor[it::num_batches].to(device)
                 batch_data_pred = prop(batch_wavl, batch_x_s, batch_x_r, self.dt)        
                 loss = criterion(batch_data_pred, batch_data_t)
                 if loss.item() == 0.0: 
                    updates.append(model.detach().cpu().numpy())
                    return np.array(updates)
                 loss.backward()            
                 running_loss += loss.item()         
              
              if Method == "1D": model.grad = self.grad_sum_spread(model.grad)               
              
              if smth_flag: model.grad = self.grad_smooth(model,smth[1],smth[0]).to(device)            
              if itr == 0 : gmax0 = (torch.abs(model.grad)).max() # get max of first itr 
#           model.grad = model.grad / gmax0   # normalize                 
              model.grad =  self.grad_reg(model,mask=msk)
        
              if tv_flag: 
                   lossTV = tv_loss(model)
                   tv_grad = torch.autograd.grad(lossTV,model)[0]                 
                   tv_grad = tv_grad/(torch.abs(tv_grad)).max() * msk # normalize 
                   model.grad = (model.grad + alphatv * tv_grad)   # combine the gradient of the two 
              model.grad =  self.grad_reg(model,mask=msk)

            

              optimizer.step() 
              # scheduler.step()
        
              model.data[model.data < m_min] = m_min
              model.data[model.data > m_max] = m_max
              loss_iter.append(running_loss)
              print('Iteration: ', itr, 'Objective: ', running_loss) 
              if itr%1==0 :
                    updates.append(model.detach().clone().cpu().numpy())  
                
              if plot_flag and itr%1==0:
                   plt.figure(figsize=(10,3))
                   plt.imshow(model.grad.cpu().numpy(),cmap='seismic',vmin=-0.3,vmax=0.3)
                   plt.axis('tight')
                   plt.colorbar(shrink=0.5)
                   plt.show()
    
                   if itr>1: 
                       plt.figure(figsize=(10,3))
                       plt.imshow(updates[itr]-updates[itr-1],cmap='seismic')
                       plt.axis('tight')
                       plt.colorbar(shrink=0.5)
                       plt.show()  
                                          
                       plt.figure(figsize=(10,3))
                       plt.imshow(updates[itr],cmap='jet')
                       plt.axis('tight')
                       plt.colorbar(shrink=0.5)
                       plt.show()   
            

                    
              # stopping criteria or relax condition smoothing ()
              if np.abs(loss_iter[itr] - loss_iter[itr-1])/max(loss_iter[itr],loss_iter[itr-1]) < tol and itr>20: 
#                   if smth_flag or tv_flag: 
                  if relax < 5:
                        if smth_flag: 
                            smth[0]=smth[0]/2
                            smth[1]=smth[1]/2
                            increase =5
                        if tv_flag :
                            alphatv = alphatv/2
                            increase =5
                        relax += 1     
                        print(f"Reduce the smoothing from {smth[0]*2},{smth[1]*2} to {smth[0]},{smth[1]}")
                  else:
                      t_end = time.time()
                      print('Runtime in min :',(t_end-t_start)/60)  
                      return np.array(updates),loss_iter
              # early stopping       
              elif min_loss < loss_iter[itr] and itr > 20: 
                 increase +=1
              else: 
                 increase = 0
                 min_loss = loss_iter[itr] 
              if  increase == 10: 
#                   if smth_flag or tv_flag: relax +=1 
                  if relax < 5:
                        if smth_flag: 
                            smth[0]=smth[0]/2
                            smth[1]=smth[1]/2
                            increase =5
                        if tv_flag :
                            alphatv = alphatv/2
                            increase =5
                        relax += 1     
                        print(f"Reduce the smoothing from {smth[0]*2},{smth[1]*2} to {smth[0]},{smth[1]}")
                  else:
                      t_end = time.time()
                      print('Runtime in min :',(t_end-t_start)/60)  
                      return np.array(updates),loss_iter
       # End of FWI iteration
       t_end = time.time()
       if hvd_flag==True and hvd.local_rank()==0:
            print('Runtime in min :',(t_end-t_start)/60)
       elif hvd_flag==False:
           print('Runtime in min :',(t_end-t_start)/60)         
       return np.array(updates),loss_iter

# ---------------------------------------
 # Functions used in the maing inversion 
    
    def grad_smooth(self,model,sigmax,sigmaz):
               gradient = model.grad.cpu().numpy() 
               gradient = gaussian_filter1d(gradient,sigma=sigmaz,axis=0) # z
               gradient = gaussian_filter1d(gradient,sigma=sigmax,axis=1) # x 
               gradient = torch.tensor(gradient)
               return gradient

    def grad_sum_spread(self,gradient):
           return gradient.sum(dim=1).view(gradient.shape[0],1).expand(-1,gradient.shape[1])


    
    def grad_reg(self,model,mask):
               gradient = model.grad            
               gmax     = (torch.abs(gradient)).max() 
               gradient = gradient / gmax  # normalize the gradient                
               gradient = gradient * mask
               return gradient
            
    def metric_average(self,val, name):
               tensor = torch.tensor(val)
               avg_tensor = hvd.allreduce(tensor, name=name)
               return avg_tensor.item()


    def line_search(self,res0,model,wavelet,data,criterion,verb,m_min,m_max,device): 
        """
        Back tracking algorithm 
        """
        
        alpha = 0.5
        decay_factor = 0.5
        model_cp = model.clone().detach()
        grad = model.grad.clone().detach()
        batch_size = 10
        

        
        for j in range (10):
                    
            model_cp.data[model_cp.data < m_min] = m_min
            model_cp.data[model_cp.data > m_max] = m_max
            prop = self.propagator(model_cp.add(grad,alpha=-alpha),device)
            running_loss=0
            for ibatch,(batch_data_t,batch_x_s,batch_x_r) in enumerate(data): 
                     shot_indices =  ( torch.round ( (batch_x_s[:, 0, 1]  - self.os  + self.ox )/ self.ds )).int().numpy()
                     if shot_indices.shape[0] > 1:  
                        batch_wavl = wavelet[:, shot_indices, 0].view(wavelet.shape[0],shot_indices.shape[0],1)
                     else: 
                           batch_wavl = wavelet[:, shot_indices, 0].view(wavelet.shape[0],1,1)  
                     batch_data_t = batch_data_t.cuda()
                     batch_data_t = batch_data_t.permute(1,0,2)
                     batch_x_s    = batch_x_s.cuda()
                     batch_x_r    = batch_x_r.cuda()
                     batch_data_pred = prop(batch_wavl, batch_x_s, batch_x_r, self.dt)

                     loss = criterion(batch_data_pred, batch_data_t)             
                     running_loss += loss.item()
            res1 =  self.metric_average(running_loss, 'avg_loss')   
            
            if res0 < res1:  
                alpha *= decay_factor
            else: break 
             
        if verb: print(f" Alpha == {alpha}")
        return alpha
        
#     def line_search(self,prop,model,wavl,data,opt,alpha,verb,m_min,m_max,device):
#         """ 
#         Back tracking line search with quadratic interpolation, ref (https://csim.kaust.edu.sa/files/ErSE328.2013/Lecture_PPTs/Lec3a.LineSearch.Gaurav.pdf) 
#         """
#         x1 = 0 
#         x2 = alpha/2
#         x3 = alpha*2
#         model1 =  model.clone().detach()
#         model2 =  model.clone().detach()
#         model3 =  model.clone().detach()
        
#         opt2 = torch.optim.Adam([{'params':[model2],'lr':x2}])
#         opt3 = torch.optim.Adam([{'params':[model3],'lr':x3}])
        

        
#         model1.grad = model.grad.clone()
#         model2.grad = model.grad.clone()
#         model3.grad = model.grad.clone()
        
#         opt2.step()
#         opt2.zero_grad() 
#         opt3.step()
#         opt3.zero_grad() 
        
#         model2.data[model2.data < m_min] = m_min
#         model2.data[model2.data > m_max] = m_max
                
#         model3.data[model3.data < m_min] = m_min
#         model3.data[model3.data > m_max] = m_max        
    

#         # Compute on subset of the data         
#         data1 = self.batch_forward_modelling (model1,wavl,device,10).cpu()
#         data2 = self.batch_forward_modelling (model2,wavl,device,10).cpu()
#         data3 = self.batch_forward_modelling (model3,wavl,device,10).cpu() 
#         batch_data = data[:,::10,:].cpu()
        
#         res1 =  torch.nn.MSELoss()(data1,batch_data)
#         res2 =  torch.nn.MSELoss()(data2,batch_data)
#         res3 =  torch.nn.MSELoss()(data3,batch_data)

#         print(f"res {res1,res2,res3}")
        
#         if res1 < res2 and res2 < res3:
#             alpha = 0.5 * ( (res1-res3)* x2**2 + (res2-res1)* x3**2) / ( (res1-res3)*x2 + (res2-res1)*x3 )
#         elif res3 < res2 and res2 < res1:
#             alpha = 0.5 * ( (res3-res2)* x2**2 + (res2-res3)* x1**2) / ( (res3-res1)*x2 + (res2-res3)*x1 )
#         elif res2< res3 and res3<res1:
#             alpha = 0.5 * ( (res2-res1)* x3**2 + (res3-res2)* x1**2) / ( (res2-res1)*x3 + (res3-res2)*x1 )
#         elif res2< res1 and res1<res3:
#             alpha = 0.5 * ( (res2-res3)* x1**2 + (res1-res2)* x3**2) / ( (res2-res3)*x1 + (res1-res2)*x3 )
#         else:
#             print("None of the cases")
        
#         if verb: print(f"Line search| alpah {alpha}")

      
#         torch.cuda.empty_cache()
        
        
#         return alpha 

        
        
def tv_loss(model):
#         h, w = model.shape 
#         a = torch.square(model[:h - 1, :w - 1] - model[1:, :w - 1])
#         b = torch.square(model[:h - 1, :w - 1] - model[:h - 1, 1:])
        tvx = torch.square(torch.gradient(model)[0])
        tvz = torch.square(torch.gradient(model)[1])
         
        return torch.sum(torch.pow( tvx +  tvz + 1e-15 , 0.5))
    

    
    
    
