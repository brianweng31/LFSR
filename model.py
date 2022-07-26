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
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        padding = (0, floor(kernel_size[1]/2), floor(kernel_size[2]/2))
        m, n = stride[1], stride[2]
        m_2, n_2 = floor(stride[1]/2), floor(stride[2]/2)

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias=False)
                
        with torch.no_grad():
            #self.conv1.weight.data = torch.ones(in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])/(in_channels*kernel_size[0]*kernel_size[1]*kernel_size[2])
            self.conv1.weight.data = torch.zeros(self.conv1.weight.data.shape)
            for i in range(m):
                for j in range(n):
                    self.conv1.weight.data[i*n+j, i*n+j,:,padding[1]+(i-m_2),padding[2]+(j-n_2)] = 1.0
            #self.conv1.bias.data = torch.zeros(self.conv1.bias.data.shape)
        
    def forward(self, x):
        # implement the forward pass
        return self.conv1(x)

class FilterBankMethod(Method):
    def __init__(self, device, s=3, t=3, in_channels=9, out_channels=9, kernel_size=(1, 7, 7), stride=(1, 3, 3), model_idx=0):
        super().__init__(self.__class__.__name__+f"_{model_idx}")
        self.net = FilterBankKernel(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride).to(device)  
        self.s = 3
        self.t = 3
        self.name = self.__class__.__name__ + f"_{model_idx}"
    def downsampling(self, hr_lf):
        return self.net(hr_lf)
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



##############################
######## LinearFilter ########
class LinearFilterKernel(nn.Module):
    def __init__(self, channels, kernel_size, stride, output_size, st):
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
        self.kernels = torch.zeros((self.ang_y*self.ang_x, 1, output_size[0], output_size[1], self.kernel_size**2))
        self.kernels[:,0,:,:,int(self.kernel_size**2/2)] = 1
        self.bias = torch.zeros((self.ang_y*self.ang_x, output_size[0], output_size[1]))

        self.weights = nn.ParameterList(
            nn.Parameter(self.kernels[i]) for i in range(self.ang_y*self.ang_x))
        # torch.Size([b, st, 3, 170, 170])
        self.biases = nn.ParameterList(
            nn.Parameter(self.bias[i]) for i in range(self.ang_y*self.ang_x))

        print(f"self.weights.shape = {self.weights.shape}")
        print(f"self.biases.shape = {self.biases.shape}")
    '''
    def linear_filter(self,img,y,x):
        # link: https://discuss.pytorch.org/t/locally-connected-layers/26979
        b, c, h, w = img.size()

        pad_x = (self.output_size[1]-1) * self.ang_x + (self.kernel_size-w)
        pad_right = int(pad_x/2)
        pad_left = pad_x - pad_right
        pad_y = (self.output_size[0]-1) * self.ang_y + (self.kernel_size-h)
        pad_bottom = int(pad_y/2)
        pad_top = pad_y - pad_bottom
        padding = (pad_left,pad_right,pad_top,pad_bottom)
        #img = F.pad(img, (2,2,2,2), "constant", 0)
        img = F.pad(img, padding, "constant", 0)
        
        kh, kw = self.kernel_size, self.kernel_size
        dh, dw = self.stride

        img = img.unfold(2, kh, dh).unfold(3, kw, dw)
        # torch.Size([1, 3, 170, 170, 7, 7])
        img = img.contiguous().view(*img.size()[:-2], -1)
        # torch.Size([1, 3, 170, 170, 49])
        out = (img.unsqueeze(1) * self.weights[y*self.ang_x+x]).sum([2, -1])
        if self.biases[y*self.ang_x+x] is not None:
            out += self.biases[y*self.ang_x+x]
        out = torch.clamp(out,min=0,max=1)
        down_img = out[0]

        return down_img
    '''
    def linear_filter(self,lf):
        b, st, c, h, w = lf.size()

        pad_x = (self.output_size[1]-1) * self.ang_x + (self.kernel_size-w)
        pad_right = int(pad_x/2)
        pad_left = pad_x - pad_right
        pad_y = (self.output_size[0]-1) * self.ang_y + (self.kernel_size-h)
        pad_bottom = int(pad_y/2)
        pad_top = pad_y - pad_bottom
        padding = (pad_left,pad_right,pad_top,pad_bottom)
        #print(f"padding = {padding}")
        #img = F.pad(img, (2,2,2,2), "constant", 0)
        lf = F.pad(lf, padding, "constant", 0)

        kh, kw = self.kernel_size, self.kernel_size
        dh, dw = self.stride

        lf = lf.unfold(3, kh, dh).unfold(4, kw, dw)
        print("lf_1.shape = ",lf.shape)
        # torch.Size([b, st, 3, 170, 170, 7, 7])
        lf = lf.contiguous().view(*lf.size()[:-2], -1)
        print("lf_2.shape = ",lf.shape)
        # torch.Size([b, st, 3, 170, 170, 49])
        down_lf = (lf * self.weights.unsqueeze(0)).sum(-1)
        print("down_lf_1.shape = ",down_lf.shape)

        down_lf += self.biases.unsqueeze(0).unsqueeze(2)
        down_lf = torch.clamp(out,min=0,max=1)
        
        return out


    def forward(self, lf, device):
        """
        Args:
            lf: (N, s*t, 3, H, W) tensor
        Returns:
            lr_lf: (N, s*t, 3, H/3, W/3) tensor
        """
        b,st,c,h,w = lf.shape

        ## modeified
        lr_lf = self.linear_filter(lf)
        ##
        '''
        lr_lf = torch.zeros((b,st,c,int(h/self.ang_y),int(w/self.ang_x)))
        for y in range(self.ang_y):
            for x in range(self.ang_x):
                for i in range(b):
                    img = lf[i,y*self.ang_x+x]
                    down_img = self.linear_filter(torch.unsqueeze(img,0),y,x)
                    lr_lf[i,y*self.ang_x+x] = down_img
        '''
        return lr_lf.to(device)


class LinearFilter(Method):
    def __init__(self, device, h, w, s=3, t=3, model_idx=0):
        super().__init__(self.__class__.__name__+f"_{model_idx}")
        self.s = s
        self.t = t
        self.h = h
        self.w = w
        self.name = self.__class__.__name__ + f"_{model_idx}"
        self.device = device

        # output_size to be determined
        self.net = LinearFilterKernel(channels=3, kernel_size=(7, 7), stride=(3, 3), output_size=(int(self.h/self.t),int(self.w/self.s)), st=(self.s,self.t)).to(device)
    
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
