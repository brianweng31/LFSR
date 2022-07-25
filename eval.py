import os
import torch
import cv2
import numpy as np
from helper import remove_img_margin, refocus_pixel_focal_stack_batch
from prepare_data import get_dataloaders
from model import BaselineMethod, FilterBankMethod, LinearFilter

######
model_name = "BaselineMethod" #FilterBankMethod, LinearFilter, BaselineMethod
model_idx = "T1"
dataset_name = "HCI"
batch_size = 16
######

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"

train_dataloader, test_dataloader = get_dataloaders(dataset_name, batch_size=batch_size)

if model_name == "FilterBankMethod":
    model = FilterBankMethod(device, 3, 3, in_channels=9, out_channels=9, kernel_size=(1, 7, 7), stride=(1, 3, 3), model_idx=model_idx)
elif model_name == "LinearFilter":
    if dataset_name == "HCI":
        h, w = 512, 512
    if dataset_name == "INRIA_Lytro":
        h, w = 379, 379
    model = LinearFilter(device, h, w, s=3, t=3, model_idx=model_idx)
elif model_name == "BaselineMethod":
    model = BaselineMethod(3,3)

if model != "BaselineMethod":
    model.load_model(os.path.join('model',model.name,'best_model'))
    model.eval_mode()


with torch.no_grad():
    for i_batch, sample_batched in enumerate(test_dataloader):
        reshape_ = (sample_batched.shape[0], -1, sample_batched.shape[3], sample_batched.shape[4], sample_batched.shape[5])
        b, s, t, c, h, w = sample_batched.shape
        sample_batched_reshaped = torch.reshape(sample_batched, reshape_).permute(0,1,4,2,3).to(device)
        
        hr_refocused = refocus_pixel_focal_stack_batch(sample_batched_reshaped, test_dataloader.dataset.disparity_range, s, t)
        hr_refocused = remove_img_margin(hr_refocused)

        lr = model.downsampling(sample_batched_reshaped)
        sr = model.enhance_LR_lightfield(lr)
        sr_refocused = refocus_pixel_focal_stack_batch(sr, test_dataloader.dataset.disparity_range, s, t)
        sr_refocused = remove_img_margin(sr_refocused)

        if i_batch == 0:
            down_lf = np.array(lr.detach().cpu())
            light_field = np.array(sr_refocused.detach().cpu())
        else:
            down_lf = np.concatenate((down_lf, lr.detach().cpu()), 0)
            light_field = np.concatenate((light_field, sr_refocused.detach().cpu()), 0)

#print(light_field.shape)
print(down_lf.shape)
light_field = np.moveaxis(light_field, 2, -1)
down_lf = np.moveaxis(down_lf, 2, -1)
if not os.path.isdir('npy'):
    os.mkdir('npy')
np.save(f'npy/down_{model_name}_{model_idx}',down_lf)
np.save(f'npy/{model_name}_{model_idx}',light_field)
#print(light_field.shape)
print(down_lf.shape)
'''
for i in range(light_field.shape[0]):
    print(light_field[i,5].shape)
    cv2.imshow(f"{i}",light_field[i,5])
    cv2.waitKey(0)
'''