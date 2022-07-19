import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
image_margin = 10

def plot_lf_tensor(lf, batch_axis=0, view_axis=1, color_axis=2, h_axis=3, w_axis=4, figure_size = (5,5)):
    plt.figure()
    lf_numpy = lf.detach().cpu().permute(batch_axis, view_axis, h_axis, w_axis, color_axis).numpy()
    b, n, h, w, c = lf_numpy.shape
    centerview_index = int(np.floor(n/2))
    for b_index in range(b):
        plt.figure(figsize = figure_size)
        plt.imshow(lf_numpy[b_index,centerview_index,:,:,:])
        plt.show()

def remove_img_margin(im):
    return im[...,image_margin:-image_margin,image_margin:-image_margin]

def shift_images(imgs, x_shift, y_shift, alpha=1.0):
    """
        Args:
            imgs: (N, 3, H, W) tensor
            x_shift, y_shift (int): (N, ) sequence
        Returns:
            shifted_img: (N, 3, H, W) tensor
    """
    n, _, h, w = imgs.shape
    y = torch.arange(h).to(imgs.device)  # height = y-axis
    x = torch.arange(w).to(imgs.device)  # width = x-axis
    grid_y, grid_x = torch.meshgrid(y, x) # (h, w) shape original coordinates

    grid_y = torch.cat([grid_y.unsqueeze(0)] * n, dim = 0)
    grid_x = torch.cat([grid_x.unsqueeze(0)] * n, dim = 0)

    grid_y = torch.add(grid_y, y_shift.view(n, 1, 1))
    grid_x = torch.add(grid_x, x_shift.view(n, 1, 1))
    new_coords = torch.stack([grid_x, grid_y], -1).type(imgs.dtype)  # (n, h, w, 2)

    """
        See https://pytorch.org/docs/stable/nn.functional.html?highlight=grid_sample#torch.nn.functional.grid_sample
        Scale coordinates to [-1, 1]
    """
    new_coords[:, :, :, 0] = 2 * (new_coords[:, :, :, 0] / h - 0.5)
    new_coords[:, :, :, 1] = 2 * (new_coords[:, :, :, 1] / w - 0.5)
    shifted_img = F.grid_sample(
        imgs, new_coords, padding_mode="border", align_corners = False, mode='bicubic')
    return torch.clip(shifted_img, 0, 1)

def refocus_pixel(lf, pixels, s, t, estimate_clear_region=False):
    """ 
    Args:
        lf: (N, C, H, W) tensor
        alpha: float, refocusing parameter
    """
    n, c, h, w = lf.shape
    x_shifts, y_shifts = [], []
    for i in range(s): # row
        for j in range(t): # col
            y_vec = i - s // 2
            x_vec = j - t // 2
            x_shifts.append(x_vec * pixels)
            y_shifts.append(y_vec * pixels)

    x_shifts = torch.tensor(x_shifts).repeat(n//s//t).to(lf.device)
    y_shifts = torch.tensor(y_shifts).repeat(n//s//t).to(lf.device)
    shifted_imgs = shift_images(lf, x_shifts, y_shifts)
    shifted_imgs_reshaped = shifted_imgs.reshape(-1, s*t, c, h, w)
    refocused_img = torch.mean(shifted_imgs_reshaped, dim = 1)

    if estimate_clear_region:
        estimated_clear_region = torch.mean(shifted_imgs_reshaped, dim=1) - shifted_imgs_reshaped[:,s*t//2,:,:,:]
        return refocused_img, estimated_clear_region
    else:
        return refocused_img

def refocus_pixel_focal_stack(lf, shift_pixel_list, s, t, estimate_clear_region=False):
    if estimate_clear_region:
        focal_stack = []
        estimated_clear_regions = []
        for p in shift_pixel_list:
            refocused, estimated_clear_region = refocus_pixel(lf, p, s, t, estimate_clear_region)
            focal_stack.append(refocused)
            estimated_clear_regions.append(estimated_clear_region)
        focal_stacks = torch.stack(focal_stack, dim = 1)
        estimated_clear_regions = torch.stack(estimated_clear_regions, dim = 1)
        return focal_stacks, estimated_clear_regions

    else:
        focal_stack = []
        for p in shift_pixel_list:
            refocused = refocus_pixel(lf, p, s, t)
            focal_stack.append(refocused)
        focal_stacks = torch.stack(focal_stack, dim = 1)
        return focal_stacks

def refocus_pixel_focal_stack_batch(batch, shift_pixel_list, s, t, estimate_clear_region=False):
    batch_reshaped = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])

    if estimate_clear_region:
        focal_stacks, estimated_clear_regions = refocus_pixel_focal_stack(batch_reshaped, shift_pixel_list, s, t, estimate_clear_region)
        return focal_stacks, estimated_clear_regions
    else:
        focal_stacks = refocus_pixel_focal_stack(batch_reshaped, shift_pixel_list, s, t)
        return focal_stacks

class EarlyStopping():
    def __init__(self, tolerance=5, min_percent=0.001):

        self.tolerance = tolerance
        self.min_percent = min_percent
        self.counter = 0
        self.early_stop = False

    def __call__(self, test_loss, last_loss):
        #print(last_loss,',',test_loss)
        if (last_loss-test_loss)/last_loss < self.min_percent:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            self.counter = 0
