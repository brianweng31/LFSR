import torch
import torch.nn as nn
import piq

### hyperparameters
optimizer = torch.optim.Adam
lr = 0.0001
#lr = 0.00001

### model selection
model = "FilterBankMethod" #FilterBankMethod, LinearFilter, BaselineMethod
model_idx = "test_3"
post_fix = ""
pre_fix = ""

## loss
#optimized_losses = [nn.L1Loss()]
optimized_losses = [nn.MSELoss()]
loss_weights = [1.0]
estimate_clear_region = False
assert len(optimized_losses) == len(loss_weights)

## loss_metrics
refocused_img_metrics = [piq.psnr, piq.ssim, piq.gmsd, ""]
refocused_img_metrics_name = ["PSNR", "SSIM", "GMSD", "LPIPS"]
assert len(refocused_img_metrics) == len(refocused_img_metrics_name)

## training
dataset_name = "HCI" # RandomTraining, HCI, INRIA_Lytro 
training_light_field_downsample_rate = [4,2,1]
#training_light_field_downsample_rate = [1]
training_light_field_epoch = [10,10,10]
#training_light_field_epoch = [30000]
batch_size = 4
assert len(training_light_field_downsample_rate)==len(training_light_field_epoch)

## early stopping
tolerance = 1000
min_percent = 0.001

#tolerance = 20
#min_percent = 1
