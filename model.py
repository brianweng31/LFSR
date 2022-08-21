import abc
from torch.utils.tensorboard import SummaryWriter
import sys
from math import floor
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

class Record:
    def __init__(self, name = None, event_idx=0) -> None:
        self.tb_writer = SummaryWriter(os.path.join('model',name,f"{event_idx}"))
        self.loss_history = []
        self.metric_history = []
        self.best_loss = sys.float_info.max

class Method(metaclass=abc.ABCMeta):
    def __init__(self, name):
        self.event_idx = 0
        self.record = Record(name,self.event_idx)
        self.name = name

    def clear_history(self):
        print('clearing')
        self.event_idx += 1
        self.record = Record(self.name,self.event_idx)

    @abc.abstractmethod
    def downsampling(self, hr_lf):
        # Preprocess to get LR-LF from HR-LF. (Optional)
        # hr_lf: (b,s*t,c,h,w)
        return NotImplemented

    @abc.abstractmethod
    def enhance_LR_lightfield(self, lr_lf):
        # Given downsampled LR-LF, return enhanced LR-LF with SR
        # lr_lf: (b,s*t,c,h,w)
        return NotImplemented  

    @abc.abstractmethod
    def train_mode(self):
        return NotImplemented 

    @abc.abstractmethod
    def eval_mode(self):
        return NotImplemented 

    @abc.abstractmethod
    def load_model(self, weight_path):
        return NotImplemented 

    @abc.abstractmethod
    def save_model(self, weight_path):
        return NotImplemented

##########################
######## Baseline ########
class BaselineMethod(Method):
    def __init__(self, s=3, t=3):
        super().__init__(self.__class__.__name__)
        self.s = s
        self.t = t
        self.name = self.__class__.__name__
    def downsampling(self, hr_lf):
        b,st,c,h,w = hr_lf.shape
        return hr_lf[:, :, :, floor(self.s/2)::self.s,floor(self.t/2)::self.t]
    def enhance_LR_lightfield(self, lr_lf):
        return torch.repeat_interleave(torch.repeat_interleave(lr_lf, 3, dim=-2), 3, dim=-1)
    def load_model(self, weight_path):
        return
    def save_model(self, weight_path):
        return
    def train_mode(self):
        return  
    def eval_mode(self):
        return  

############################
######## FilterBank ########
class FilterBankKernel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, layer_num=1):
        super().__init__()
        
        padding = (0, floor(kernel_size[1]/2), floor(kernel_size[2]/2))
        #padding = (0, floor(kernel_size[1]/2)-1, floor(kernel_size[2]/2)-1)
        '''
        m, n = stride[1], stride[2]
        m_2, n_2 = floor(stride[1]/2), floor(stride[2]/2)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias=False)
                
        with torch.no_grad():
            #self.conv1.weight.data = torch.ones(in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])/(in_channels*kernel_size[0]*kernel_size[1]*kernel_size[2])
            self.conv1.weight.data = torch.zeros(self.conv1.weight.data.shape)

            
            
            # delta
            #for i in range(m):
                #for j in range(n):
                    #self.conv1.weight.data[i*n+j, i*n+j,:,padding[1]+(i-m_2),padding[2]+(j-n_2)] = 1.0
                    #self.conv1.weight.data[i*n+j, i*n+j,:,floor(kernel_size[1]/2), floor(kernel_size[2]/2)] = 1.0
            #self.conv1.bias.data = torch.zeros(self.conv1.bias.data.shape)
            #print(self.conv1.weight.data[0,0,0])
            
            

            # Gaussian
            assert kernel_size[1] == kernel_size[2]
            sigma = (kernel_size[1]-1)/6

            axis = torch.arange(-floor(kernel_size[1]/2),floor(kernel_size[1]/2)+1)
            x, y = torch.meshgrid(axis,axis)
            gaussian_kernel = torch.exp(-(x**2+y**2)/(2*sigma**2))
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
            for i in range(m):
                for j in range(n):
                    self.conv1.weight.data[i*n+j, i*n+j,:] = gaussian_kernel
            print(self.conv1.weight.data[0,0,0])
        '''
        self.in_channels = in_channels
        self.layer_num = layer_num
        self.convs = nn.ModuleList()
        for _ in range(in_channels*layer_num):
            self.convs.append(nn.Conv3d(in_channels=1, out_channels=1, kernel_size = kernel_size, stride = stride, padding = padding, bias=False))
                
        with torch.no_grad():
            assert kernel_size[1] == kernel_size[2]
            sigma = (kernel_size[1]-1)/6

            axis = torch.arange(-floor(kernel_size[1]/2),floor(kernel_size[1]/2)+1)
            x, y = torch.meshgrid(axis,axis)
            gaussian_kernel = torch.exp(-(x**2+y**2)/(2*sigma**2))
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

            for k in range(in_channels*layer_num):
                
                #self.convs[k].weight.data = torch.zeros(self.convs[k].weight.data.shape, requires_grad=True)
                #self.convs[k].weight.data[0, 0, :, 1, 1] = 1.0
            
                self.convs[k].weight.data[0,0,:] = gaussian_kernel
    '''  
    def forward(self, x):
        # implement the forward pass
        return self.conv1(x)   
    ''' 
    def forward(self, x):
        outputs = []
        for k in range(self.in_channels):
            out = self.convs[k*self.layer_num](x[:,[k],:,:,:])
            for l in range(1, self.layer_num):
                out = self.convs[k*self.layer_num + l](out)
            outputs.append(out)

        out = torch.cat(outputs, axis=1)
        return out
    

class FilterBankMethod(Method):
    def __init__(self, device, s=3, t=3, in_channels=9, out_channels=9, kernel_size=(1, 7, 7), stride=(1, 3, 3), model_idx=0):
        super().__init__(self.__class__.__name__+f"_{model_idx}")
        self.net = FilterBankKernel(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride).to(device)  
        self.s = s
        self.t = t
        assert self.s == self.t
        self.name = self.__class__.__name__ + f"_{model_idx}"
    def downsampling(self, hr_lf):
        '''
        b,st,c,h,w = hr_lf.shape
        ds_lf = hr_lf[:, :, :, 0::self.s,0::self.t]
        ds_lf = shift_images(ds_lf.reshape(b*st, c, h//2, w//2), 0.75*torch.ones(b*st).to(device), 0.75*torch.ones(b*st).to(device)).reshape(b, st, c, h//2, w//2)
        '''
        return self.net(hr_lf)
    def enhance_LR_lightfield(self, lr_lf):
        '''
        # test
        print(f'down_lf.shape = {lr_lf.shape}')
        print('top-left view first row')
        print(lr_lf[0,0,:,0])
        #
        '''
        modified_lf = torch.repeat_interleave(torch.repeat_interleave(lr_lf, self.s, dim=-2), self.s, dim=-1)
        #modified_lf = torch.repeat_interleave(torch.repeat_interleave(lr_lf, 3, dim=-2), 3, dim=-1)
        for i in range(self.s):
            for j in range(self.t):
                modified_lf[:,i*self.t+j, :, :, :] = torch.roll(modified_lf[:,i*self.t+j, :, :, :], shifts=(i-1, j-1), dims=(-2,-1))
    
        return modified_lf
    def load_model(self, weight_path):
        self.net.load_state_dict(torch.load(weight_path))
    def save_model(self, weight_path):
        torch.save(self.net.state_dict(), weight_path) 
    def train_mode(self):
        self.net.train()  
    def eval_mode(self):
        self.net.eval()  



##############################
######## LinearFilter ########
class LinearFilterKernel(nn.Module):
    def __init__(self, channels, kernel_size, stride, output_size, st, FB_kernels):
        super().__init__()
        # Input: (N, Cin, H, W)
        # Output: (N, Cout, Hout, Wout)
        # Kernel size: (k_H, k_W)
        self.ang_x, self.ang_y = st[0], st[1]
        assert kernel_size[0] == kernel_size[1]
        self.kernel_size = kernel_size[0]
        self.stride = stride
        self.output_size = output_size
        #self.kernels = torch.zeros((self.ang_y*self.ang_x,out_channels, in_channels, output_size[0], output_size[1], self.kernel_size**2))
        self.kernels = torch.zeros(self.ang_y*self.ang_x, self.ang_y*self.ang_x, 1, output_size[0], output_size[1], self.kernel_size**2)
        if FB_kernels == None:
            '''
            for i in range(st[0]):
                for j in range(st[1]):
                    self.kernels[i*st[1]+j,i*st[1]+j,0,:,:,self.kernel_size*(2+i)+2+j] = 1

            '''
            for i in range(self.ang_y*self.ang_x):
                self.kernels[i,i,0,:,:,int(self.kernel_size**2/2)] = 1
            
        else:
            # FB [9,9,1,7,7]
            # LinearFilter [9,9,1,170,170,49]
            FB_kernels = FB_kernels.contiguous().view(*FB_kernels.size()[:-2], -1)
            # FB [9,9,1,49]
            FB_kernels = FB_kernels.view(*FB_kernels.size()[:-1],1,1,-1)

            # LF [out_channel, in_channel]
            # FB [in_channel, out_channel]
            #FB_kernels = torch.moveaxis(FB_kernels, 0, 1)
            # FB [9,9,1,1,1,49]
            FB_kernels = torch.repeat_interleave(torch.repeat_interleave(FB_kernels,output_size[0],dim=3), output_size[1], dim=4)
            # FB [9,9,1,170,170,49]
            self.kernels = FB_kernels
            '''
            print('same view')
            print(self.kernels[0,0,0,0,0])
            print(self.kernels[0,0,0,150,0])
            print(self.kernels[0,0,0,0,150])
            print('different view')
            print(self.kernels[0,1,0,0,0])
            '''
            


        self.bias = torch.zeros(self.ang_y*self.ang_x, output_size[0], output_size[1])
        self.weights =  nn.Parameter(self.kernels)
        # torch.Size([st, st, 1, 170, 170, 49])
        self.biases = nn.Parameter(self.bias)

    def linear_filter(self,lf):
        #print("lf.device() = ",lf.get_device())
        b, st, c, h, w = lf.size()

        pad_x = (self.output_size[1]-1) * self.ang_x + (self.kernel_size-w)
        pad_right = int(pad_x/2)
        pad_left = pad_x - pad_right
        pad_y = (self.output_size[0]-1) * self.ang_y + (self.kernel_size-h)
        pad_bottom = int(pad_y/2)
        pad_top = pad_y - pad_bottom
        #padding = (pad_left,pad_right,pad_top,pad_bottom)
        # test
        #padding = (0,pad_x,0,pad_y)
        padding = (floor(self.kernel_size/2),pad_x-floor(self.kernel_size/2),floor(self.kernel_size/2),pad_y-floor(self.kernel_size/2))
        lf = F.pad(lf, padding, "constant", 0)

        kh, kw = self.kernel_size, self.kernel_size
        dh, dw = self.stride

        lf = lf.unfold(3, kh, dh).unfold(4, kw, dw)
        #print("lf_1.shape = ",lf.shape)
        # torch.Size([b, st, 3, 170, 170, 7, 7])
        lf = lf.contiguous().view(*lf.size()[:-2], -1)
        #print("lf_2.shape = ",lf.shape)
        # torch.Size([b, st, 3, 170, 170, 49])
        down_lf = (lf.unsqueeze(1) * self.weights.unsqueeze(0)).sum([2,-1])
        #      lf = torch.Size([b, 1, st, 3, 170, 170, 49])
        # weights = torch.Size([1, st, st, 1, 170, 170, 49])
        #print("down_lf.shape = ",down_lf.shape)

        down_lf += self.biases.unsqueeze(0).unsqueeze(2)
        down_lf = torch.clamp(down_lf,min=0,max=1)
        # down_lf = torch.Size([b, st, 3, 170, 170])
        #print("down_lf.device = ",down_lf.get_device())
        
        return down_lf


    def forward(self, lf, device):
        """
        Args:
            lf: (N, s*t, 3, H, W) tensor
        Returns:
            lr_lf: (N, s*t, 3, H/3, W/3) tensor
        """
        b,st,c,h,w = lf.shape
        lr_lf = self.linear_filter(lf)
        #print('lr_lf.device = ',lr_lf.get_device())
        return lr_lf


class LinearFilter(Method):
    def __init__(self, device, h, w, s=3, t=3, model_idx=0, FB_kernels=None):
        super().__init__(self.__class__.__name__+f"_{model_idx}")
        self.s = s
        self.t = t
        self.h = h
        self.w = w
        self.name = self.__class__.__name__ + f"_{model_idx}"
        self.device = device

        # output_size to be determined
        self.net = LinearFilterKernel(channels=3, kernel_size=(7, 7), stride=(3, 3), output_size=(int(self.h/self.t),int(self.w/self.s)), st=(self.s,self.t),FB_kernels=FB_kernels).to(device)
    
    def downsampling(self, hr_lf):
        #b,st,c,h,w = hr_lf.shape
        #lr_lf = self.net(hr_lf.to(device))
        return self.net(hr_lf,self.device)
    def enhance_LR_lightfield(self, lr_lf):
        modified_lf = torch.repeat_interleave(torch.repeat_interleave(lr_lf, 3, dim=-2), 3, dim=-1)
        for i in range(self.s):
            for j in range(self.t):
                modified_lf[:,i*self.t+j, :, :, :] = torch.roll(modified_lf[:,i*self.t+j, :, :, :], shifts=(i-1, j-1), dims=(-2,-1))
        
        return modified_lf
    def load_model(self, weight_path):
        self.net.load_state_dict(torch.load(weight_path))
    def save_model(self, weight_path):
        torch.save(self.net.state_dict(), weight_path) 
    def train_mode(self):
        self.net.train()  
    def eval_mode(self):
        self.net.eval()  
