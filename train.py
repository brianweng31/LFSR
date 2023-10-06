import torch
from config import optimized_losses, loss_weights
from helper import refocus_pixel_focal_stack_batch, remove_img_margin
from model import BaselineMethod, FilterBankMethod, LinearFilter


def training(dataloader,device,methods,optimizers,optimized_losses,estimate_clear_region,early_stopped):
    for i_batch, sample_batched in enumerate(dataloader):
        reshape_ = (sample_batched.shape[0], -1, sample_batched.shape[3], sample_batched.shape[4], sample_batched.shape[5])
        b, s, t, c, h, w = sample_batched.shape
        sample_batched_reshaped = torch.reshape(sample_batched, reshape_).permute(0,1,4,2,3).to(device)
        
        if estimate_clear_region:
            hr_refocused, estimated_clear_regions = refocus_pixel_focal_stack_batch(sample_batched_reshaped, dataloader.dataset.disparity_range, s, t, estimate_clear_region=True)
            estimated_clear_regions = remove_img_margin(estimated_clear_regions)
        else:
            hr_refocused = refocus_pixel_focal_stack_batch(sample_batched_reshaped, dataloader.dataset.disparity_range, s, t)
        
        hr_refocused = remove_img_margin(hr_refocused)

        for method_idx in range(len(methods)):
            if early_stopped[method_idx] == True:
                continue
            else:
                lr = methods[method_idx].downsampling(sample_batched_reshaped)
                sr = methods[method_idx].enhance_LR_lightfield(lr)
                sr_refocused = refocus_pixel_focal_stack_batch(sr, dataloader.dataset.disparity_range, s, t)
                sr_refocused = remove_img_margin(sr_refocused)

                losses = []
                for optimized_losses_idx in range(len(optimized_losses)):
                    if estimate_clear_region:
                        losses.append(optimized_losses[optimized_losses_idx]((sr_refocused - hr_refocused)*estimated_clear_regions, torch.zeros(hr_refocused.shape).to(device)))
                    else:
                        losses.append(optimized_losses[optimized_losses_idx]((sr_refocused - hr_refocused), torch.zeros(hr_refocused.shape).to(device)))
                    
                loss_sum = torch.sum(torch.stack(losses), dim=0)

                loss_sum.backward()
                optimizers[method_idx].step()
                optimizers[method_idx].zero_grad()
    
    return