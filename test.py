import torch
import numpy as np
import piq
from config import optimized_losses
from config import refocused_img_metrics, refocused_img_metrics_name
from helper import remove_img_margin, refocus_pixel_focal_stack_batch


def testing(dataloader, device, model, epoch=0, estimate_clear_region=False):

    losses = [[] for _ in range(len(optimized_losses))]
    metrics = [[] for _ in range(len(refocused_img_metrics))]
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
        '''
        print('lf[0] top left first row')
        print(sampled_batched_reshaped[0,0,:,0])
        '''
        lr = model.downsampling(sample_batched_reshaped)
        sr = model.enhance_LR_lightfield(lr)
        sr_refocused = refocus_pixel_focal_stack_batch(sr, dataloader.dataset.disparity_range, s, t)
        sr_refocused = remove_img_margin(sr_refocused)

        for optimized_losses_idx in range(len(optimized_losses)):
            if estimate_clear_region:
                loss = optimized_losses[optimized_losses_idx]((sr_refocused - hr_refocused)*(torch.exp(-torch.abs(estimated_clear_regions))), torch.zeros(hr_refocused.shape).to(device))
            else:
                loss = optimized_losses[optimized_losses_idx]((sr_refocused - hr_refocused), torch.zeros(hr_refocused.shape).to(device))
            losses[optimized_losses_idx].append(loss.detach().cpu().numpy())

        reshape_ = (-1, hr_refocused.shape[2], hr_refocused.shape[3], hr_refocused.shape[4])
        sr_refocused_reshaped = sr_refocused.reshape(reshape_).detach().cpu()
        hr_refocused_reshaped = hr_refocused.reshape(reshape_).detach().cpu()

        for refocused_img_metrics_idx in range(len(refocused_img_metrics)):
            if refocused_img_metrics_name[refocused_img_metrics_idx] == "LPIPS":
                loss = piq.LPIPS()
                metric_value = loss(sr_refocused_reshaped, hr_refocused_reshaped)
                metrics[refocused_img_metrics_idx].append(metric_value.detach().cpu().numpy())
            else:
                metric_value = refocused_img_metrics[refocused_img_metrics_idx](sr_refocused_reshaped, hr_refocused_reshaped)
                metrics[refocused_img_metrics_idx].append(metric_value.detach().cpu().numpy())

    return np.mean(np.array(losses), axis=1), np.mean(np.array(metrics), axis=1)