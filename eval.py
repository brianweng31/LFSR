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
model_name = "FilterBankMethod" #FilterBankMethod, LinearFilter, BaselineMethod
model_idx = "dis_0"
dataset_name = "HCI" #HCI, RandomTraining, SR_test_dataset
batch_size = 8

#optimized_losses = [nn.L1Loss()]
optimized_losses = [nn.MSELoss()]
estimate_clear_region = False

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
    #model = FilterBankMethod(device, 3, 3, in_channels=9, out_channels=9, kernel_size=(1, 7, 7), stride=(1, 3, 3), model_idx=model_idx)
    #model = FilterBankMethod(device, 3, 3, in_channels=9, out_channels=9, kernel_size=(1, 7, 7), stride=(1, 1, 1), model_idx=model_idx)
    #model = FilterBankMethod(device, 3, 3, in_channels=9, out_channels=9, kernel_size=(1, 3, 3), stride=(1, 3, 3), model_idx=model_idx)
    # 1d kernels
    model = FilterBankMethod(device, 3, 3, in_channels=9, out_channels=9, kernel_size=13, stride=(1, 3, 3), model_idx=model_idx)
        

elif model_name == "LinearFilter":
    if dataset_name == "HCI":
        h, w = 512, 512
    if dataset_name == "INRIA_Lytro":
        h, w = 379, 379
    model = LinearFilter(device, h, w, s=3, t=3, model_idx=model_idx)
elif model_name == "BaselineMethod":
    model = BaselineMethod(3,3)

#try:
if model != "BaselineMethod":
    print('here')
    print(model.name)
    model.load_model(os.path.join('model',model.name,'best_model'))
    print('here1')
    model.eval_mode()
    print('here2')
    for params in model.net.parameters():
        print('here3')
        print(params.size())
        print(f'params.sum() = {params.sum()}')
#except:
    #pass


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
    losses, metrics, sr_refocused_reshaped, hr_refocused_reshaped = testing(test_dataloader, device, model, 0, estimate_clear_region)
    '''                         
    model.record.loss_history.append([])
    model.record.metric_history.append([])
    log_str = "Downsample %d Epoch [%d/%d] %s: " % (downsample_rate, 0, 10, model.name)
    '''
    loss_history = []
    metric_history = []
    log_str = ""
    for optimized_losses_idx in range(len(optimized_losses)):
        '''
        model.record.tb_writer.add_scalar("Loss/loss_sum", losses[optimized_losses_idx], epoch)
        model.record.loss_history[-1].append(losses[optimized_losses_idx])
        log_str += "Loss %d: %.6f " % (optimized_losses_idx, model.record.loss_history[-1][-1])
        '''
        loss_history.append(losses[optimized_losses_idx])
        log_str += "Loss %d: %.6f " % (optimized_losses_idx, loss_history[-1])
    for refocused_img_metrics_idx in range(len(refocused_img_metrics)):
        '''
        model.record.tb_writer.add_scalar("Metric/%s"%refocused_img_metrics_name[refocused_img_metrics_idx], metrics[refocused_img_metrics_idx], epoch)
        model.record.metric_history[-1].append(metrics[refocused_img_metrics_idx])
        log_str += "Metric %s: %.6f " % (refocused_img_metrics_name[refocused_img_metrics_idx], model.record.metric_history[-1][-1])
        '''
        metric_history.append(metrics[refocused_img_metrics_idx])
        log_str += "Metric %s: %.6f " % (refocused_img_metrics_name[refocused_img_metrics_idx], metric_history[-1])
        
    '''
    if np.sum(model.record.loss_history[-1]) < model.record.best_loss:
        print("Found better model: %.6f < %.6f" % (np.sum(model.record.loss_history[-1]), model.record.best_loss))
        model.record.best_loss = np.sum(model.record.loss_history[-1])
        model.save_model(os.path.join('model',model.name,'best_model'))
    '''

    print(log_str)

#print(light_field.shape)
#print(f'down_lf.shape = {down_lf.shape}')
light_field = np.moveaxis(light_field, 2, -1)
down_lf = np.moveaxis(down_lf, 2, -1)

if not os.path.isdir('npy'):
    os.mkdir('npy')
np.save(f'npy/down_{model_name}',down_lf)
np.save(f'npy/{model_name}',light_field)

print(f'light_field.shape = {light_field.shape}')
print(f'down_lf.shape = {down_lf.shape}')
'''
for i in range(light_field.shape[0]):
    print(light_field[i,5].shape)
    cv2.imshow(f"{i}",light_field[i,5])
    cv2.waitKey(0)
'''