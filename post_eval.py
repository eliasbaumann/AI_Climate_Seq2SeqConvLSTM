import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import h5py

from train import BATCH_SIZE, SEQUENCE_LENGTH, path

hf = h5py.File(path+'predictions.h5', 'r')
preds = hf.get('preds').value
futures = hf.get('preds_future').value

# print(elev_eu.shape)

futures_t2m = futures[:,:,:,:,0,:,:] 
futures_sd = futures[:,:,:,:,1,:,:]
futures_tp = futures[:,:,:,:,2,:,:] 
#(:, :, :, :, 2, :, :)
def create_distributions(data):
    mean = np.mean(data, axis=1)
    var = np.var(data, axis=1)
    mean = np.reshape(mean, (BATCH_SIZE*SEQUENCE_LENGTH,*mean.shape[2:]))
    var = np.reshape(var, (BATCH_SIZE*SEQUENCE_LENGTH,*var.shape[2:]))
    return mean, var

t2m_m, t2m_var = create_distributions(futures_t2m)
sd_m, sd_var = create_distributions(futures_sd)
tp_m, tp_var = create_distributions(futures_tp)

for i in range(13):
    plt.imshow(sd_m[-1,i,:,:], vmax=.2)
    plt.savefig(path+'sd_mean_pred_2020_'+str(i)+'.png')