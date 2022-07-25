import os
import torch
from helper import remove_img_margin, refocus_pixel_focal_stack
from prepare_data import get_dataloaders
from model import BaselineMethod, FilterBankMethod, LinearFilter

######
model_name = "FilterBankMethod" #FilterBankMethod, LinearFilter, BaselineMethod
model_idx = "F1"
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

model.load_model(os.path.join('model',model.name,'best_model'), torch.device('cpu'))
model.eval_mode()

with torch.no_grad():
    for i_batch, sample_batched in enumerate(dataloader):
        reshape_ = (sample_batched.shape[0], -1, sample_batched.shape[3], sample_batched.shape[4], sample_batched.shape[5])
        b, s, t, c, h, w = sample_batched.shape
        sample_batched_reshaped = torch.reshape(sample_batched, reshape_).permute(0,1,4,2,3).to(device)
        
        hr_refocused = refocus_pixel_focal_stack_batch(sample_batched_reshaped, dataloader.dataset.disparity_range, s, t)
        hr_refocused = remove_img_margin(hr_refocused)

        lr = model.downsampling(sample_batched_reshaped)
        sr = model.enhance_LR_lightfield(lr)
        sr_refocused = refocus_pixel_focal_stack_batch(sr, dataloader.dataset.disparity_range, s, t)
        sr_refocused = remove_img_margin(sr_refocused)

        print(sr_refocused.shape)