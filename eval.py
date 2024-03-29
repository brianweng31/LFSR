import os
import torch
import cv2
import numpy as np
import torch.nn as nn
import piq

from helper import remove_img_margin, refocus_pixel_focal_stack_batch
from prepare_data import get_dataloaders
from model import BaselineMethod, FilterBankMethod, LinearFilter
from test import testing

######
model_name = "LinearFilter" #FilterBankMethod, LinearFilter, BaselineMethod
model_idx = "delta_ecr_-2"
dataset_name = "HCI_single" #HCI, HCI_single, RandomTraining, SR_test_dataset
batch_size = 8

optimized_losses = [nn.L1Loss()]
#optimized_losses = [nn.MSELoss()]
estimate_clear_region = True

refocused_img_metrics = [piq.psnr, piq.ssim, piq.gmsd, ""]
refocused_img_metrics_name = ["PSNR", "SSIM", "GMSD", "LPIPS"]
######

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"

print(device)
## prepare dataloader
train_dataloader, test_dataloader = get_dataloaders(dataset_name, batch_size=batch_size)

## set up model
if model_name == "FilterBankMethod":
    model = FilterBankMethod(device, 3, 3, in_channels=9, out_channels=9, kernel_size=(1, 7, 7), stride=(1, 3, 3), model_idx=model_idx)
    #model = FilterBankMethod(device, 3, 3, in_channels=9, out_channels=9, kernel_size=(1, 7, 7), stride=(1, 1, 1), model_idx=model_idx)
    #model = FilterBankMethod(device, 3, 3, in_channels=9, out_channels=9, kernel_size=(1, 3, 3), stride=(1, 3, 3), model_idx=model_idx)    

elif model_name == "LinearFilter":
    if dataset_name == "HCI":
        h, w = 512, 512
    if dataset_name == "HCI_single":
            h, w = 512, 512
    if dataset_name == "INRIA_Lytro":
        h, w = 379, 379
    model = LinearFilter(device, h, w, s=3, t=3, model_idx=model_idx)
elif model_name == "BaselineMethod":
    model = BaselineMethod(3,3)

if model != "BaselineMethod":
    model.load_model(os.path.join('model',model.name,'best_model'))
    model.eval_mode()
    kernels = 0
    i = 0
    for params in model.net.parameters():
        print(params.size())
        if i == 0:
            kernels = params
            i += 1

with torch.no_grad():
    ## generate down_lf and sr
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

    ## print metrics
    if estimate_clear_region:
        losses, metrics, sr_refocused_reshaped, hr_refocused_reshaped, estimate_clear_regions = testing(test_dataloader, device, model, 0, estimate_clear_region)
    else:
        losses, metrics, sr_refocused_reshaped, hr_refocused_reshaped = testing(test_dataloader, device, model, 0, estimate_clear_region)

    loss_history = []
    metric_history = []
    log_str = ""
    for optimized_losses_idx in range(len(optimized_losses)):
        loss_history.append(losses[optimized_losses_idx])
        log_str += "Loss %d: %.6f " % (optimized_losses_idx, loss_history[-1])
    for refocused_img_metrics_idx in range(len(refocused_img_metrics)):
        metric_history.append(metrics[refocused_img_metrics_idx])
        log_str += "Metric %s: %.6f " % (refocused_img_metrics_name[refocused_img_metrics_idx], metric_history[-1])

    print(log_str)

light_field = np.moveaxis(light_field, 2, -1)
down_lf = np.moveaxis(down_lf, 2, -1)

if not os.path.isdir('npy'):
    os.mkdir('npy')
np.save(f'npy/down_{model_idx}',down_lf)
np.save(f'npy/{model_idx}',light_field)

try:
    kernels = np.array(kernels.detach().cpu())
except:
    pass