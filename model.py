import abc
from torch.utils.tensorboard import SummaryWriter
import sys
from math import floor
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np

from helper import shift_images

filter_a = [0.08433, 0.1705, 0.1576, 0.1174, 0.06184, 0.01709]
filter_omega = 3.442
#filter_a = [1., 0., 0., 0., 0., 0.]
#filter_a = [0.1720, 0.2558, 0.1051, 0.0239, 0.0030, 0.0002]
#filter_omega = 1.

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
    #def __init__(self, in_channels, out_channels, kernel_size, stride, layer_num=1):
    def __init__(self, s, t, kernel_size):
        super().__init__()
        '''
        #padding = (0, floor(kernel_size[1]/2), floor(kernel_size[2]/2))
        padding = (0, floor(kernel_size[1]/2)-1, floor(kernel_size[2]/2)-1)
        m, n = stride[1], stride[2]
        m_2, n_2 = floor(stride[1]/2), floor(stride[2]/2)
        '''
        '''
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
        '''
        # two id kernels, sub-view to sub-view
        self.in_channels = in_channels
        self.convs_vertical = nn.ModuleList()
        self.convs_horizontal = nn.ModuleList()
        
        vertical_kernel_size = (1, kernel_size[1], 1)
        horizontal_kernel_size = (1, 1,  kernel_size[2])
        vertical_stride = (1, stride[1], 1)
        horizontal_stride = (1,  1,  stride[2])
        vertical_padding = (0, padding[1], 0)
        horizontal_padding = (0,  0,  padding[2])
        
        for _ in range(in_channels):
            self.convs_vertical.append(nn.Conv3d(in_channels=1, out_channels=1, kernel_size = vertical_kernel_size, stride = vertical_stride, padding = vertical_padding, bias=False))
            self.convs_horizontal.append(nn.Conv3d(in_channels=1, out_channels=1, kernel_size = horizontal_kernel_size, stride = horizontal_stride, padding = horizontal_padding, bias=False))

        with torch.no_grad():
            # modified
            assert kernel_size[1] == kernel_size[2]
            sigma = (kernel_size[1]-1)/6

            x = torch.arange(-floor(kernel_size[1]/2),floor(kernel_size[1]/2)+1)
            gaussian_kernel = torch.exp(-(x**2)/(2*sigma**2))
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
            print(gaussian_kernel)
            #
            for k in range(self.in_channels):
                self.convs_vertical[k].weight.data = torch.zeros(self.convs_vertical[k].weight.data.shape, requires_grad=True)
                self.convs_horizontal[k].weight.data = torch.zeros(self.convs_horizontal[k].weight.data.shape, requires_grad=True)
            for i in range(m):
                for j in range(n):
                    
                    self.convs_vertical[i*n+j].weight.data[0, 0,:,padding[1]+i-1,0] = 1.0
                    self.convs_horizontal[i*n+j].weight.data[0, 0,:,0,padding[2]+j-1] = 1.0
                    #self.convs_vertical[i*n+j].weight.data[0,0,0,:,0] = gaussian_kernel
                    #self.convs_horizontal[i*n+j].weight.data[0,0,0,0,:] = gaussian_kernel
        '''
        
        # 1d cosine
        self.s = s
        self.t = t
        # cosine
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 == 1
        self.filter_weight = torch.nn.parameter.Parameter(data=torch.tensor(filter_a), requires_grad=True)
        #self.filter_weight_ver = nn.ParameterList([nn.Parameter(data=torch.tensor(filter_a), requires_grad=True) for i in range(9)])
        #self.filter_weight_hor = nn.ParameterList([nn.Parameter(data=torch.tensor(filter_a), requires_grad=True) for i in range(9)])
        self.filter_omega = torch.nn.parameter.Parameter(data=torch.tensor(filter_omega), requires_grad=True)   
        #self.filter_omega_ver = nn.ParameterList([nn.Parameter(data=torch.tensor(filter_omega), requires_grad=True) for i in range(9)])
        #self.filter_omega_hor = nn.ParameterList([nn.Parameter(data=torch.tensor(filter_omega), requires_grad=True) for i in range(9)])
        self.a_subscript = torch.nn.parameter.Parameter(data=torch.arange(0, self.filter_weight.shape[0]), requires_grad=False) 
        
        '''
        # 2d sinc
        self.s = s
        self.t = t
        
        x, y = np.linspace(-10,10,21), np.linspace(-10,10,21)
        X, Y = np.meshgrid(x, y)

        f = np.sinc(np.hypot(X, Y))
        filter_ = torch.tensor(f/f.sum(), dtype=torch.float)
        dh, dw = filter_.shape
        
        #filter_ = filter_.unsqueeze(0).unsqueeze(0)
        #filter_ = torch.repeat_interleave(filter_,3,dim=1)
        #self.filter_ = torch.repeat_interleave(filter_,3,dim=0)

        self.filter = torch.zeros((3,3,dh,dw), dtype=torch.float).to("cuda:0")
        for i in range(3):
            self.filter[i,i] = filter_

        print(f"filter_.size() = {self.filter.size()}")
        '''

        # gaussian
        '''
        device = "cuda:0"
        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            filter_sigma = kernel_size/6
        else:
            filter_sigma = (kernel_size-1)/6
        #self.filter_sigma = torch.nn.parameter.Parameter(data=torch.tensor(filter_sigma), requires_grad=True)   
        
        self.filter_sigma_ver = nn.ParameterList([nn.Parameter(data=torch.tensor(filter_sigma), requires_grad=True) for i in range(9)])
        self.filter_sigma_hor = nn.ParameterList([nn.Parameter(data=torch.tensor(filter_sigma), requires_grad=True) for i in range(9)])
        
        self.x = torch.linspace(-floor(kernel_size/2), floor(kernel_size/2), kernel_size).to(device)
        gaussian_kernel = torch.exp(-(self.x**2)/(2*filter_sigma**2)).to(device)
        #self.filter_weight = torch.nn.parameter.Parameter(data=torch.tensor(1/gaussian_kernel.sum()), requires_grad=True)
        
        self.filter_weight_ver = nn.ParameterList([nn.Parameter(data=torch.tensor(1/gaussian_kernel.sum()), requires_grad=True) for i in range(9)])
        self.filter_weight_hor = nn.ParameterList([nn.Parameter(data=torch.tensor(1/gaussian_kernel.sum()), requires_grad=True) for i in range(9)])
        
        self.filter_mean_ver = nn.ParameterList([nn.Parameter(data=torch.zeros(1), requires_grad=True) for i in range(9)])
        self.filter_mean_hor = nn.ParameterList([nn.Parameter(data=torch.zeros(1), requires_grad=True) for i in range(9)])
        '''
        
    
    #def lowpass(self,s,t,axis):
    def lowpass(self):
        '''
        # sinc
        fc = 1/6
        #x = np.linspace(-100,100,201)
        x = np.linspace(-10,10,21)
        #print(x)
        #first_filter = (10*np.pi*x*np.sin(np.pi*x)+6*np.cos(np.pi*x)-6) / (12*np.pi**2*x**2)
        #first_filter[100] = 5/6-1/4

        sinc_filter = 2*fc*np.sinc(2*fc*x)
        #print(f'sinc_filter.sum() = {sinc_filter.sum()}')
        #print(first_filter)
        #print(sinc_filter)
        #y = np.convolve(first_filter,sinc_filter)
        #y2 = y[190:211]/y[190:211].sum()

        filter_ = torch.tensor(sinc_filter/sinc_filter.sum(), dtype=torch.float).to(self.filter_omega.device)
        '''
        '''
        y = np.convolve(first_filter,sinc_filter)
        print(f'y.sum() = {y.sum()}')
        #print(y)
        x1 = np.linspace(0,len(y)-1,len(y))

        y2 = y[190:211]
        x2 = np.linspace(-10,10,21)
        '''
        
        
        # cosine
        normalized_ratio = self.kernel_size/13.0
        x = torch.linspace(-normalized_ratio, normalized_ratio, self.kernel_size).to(self.filter_omega.device)
        inner_cosine = self.filter_omega * torch.outer(x, self.a_subscript)
        cos_terms = torch.cos(inner_cosine)
        filter_ = torch.matmul(cos_terms, self.filter_weight)
        
        '''
        if axis == 'ver':
            x = torch.linspace(-normalized_ratio, normalized_ratio, self.kernel_size).to(self.filter_omega_ver[s*self.t+t].device)
            inner_cosine = self.filter_omega_ver[s*self.t+t] * torch.outer(x, self.a_subscript)
            cos_terms = torch.cos(inner_cosine)
            filter_ = torch.matmul(cos_terms, self.filter_weight_ver[s*self.t+t])
        else: # axis == 'hor'
            x = torch.linspace(-normalized_ratio, normalized_ratio, self.kernel_size).to(self.filter_omega_hor[s*self.t+t].device)
            inner_cosine = self.filter_omega_hor[s*self.t+t] * torch.outer(x, self.a_subscript)
            cos_terms = torch.cos(inner_cosine)
            filter_ = torch.matmul(cos_terms, self.filter_weight_hor[s*self.t+t])
        '''
        '''
        # gaussian
        device = "cuda:0"
        if axis == 'ver':
            gaussian_kernel = torch.exp(-((self.x-self.filter_mean_ver[s*self.t+t])**2)/(2*self.filter_sigma_ver[s*self.t+t]**2)).to(device)
            filter_  = self.filter_weight_ver[s*self.t+t] * gaussian_kernel
        else: # axis == 'hor'
            gaussian_kernel = torch.exp(-((self.x-self.filter_mean_hor[s*self.t+t])**2)/(2*self.filter_sigma_hor[s*self.t+t]**2)).to(device)
            filter_  = self.filter_weight_hor[s*self.t+t] * gaussian_kernel
        '''
        return filter_
    
    
    # 1d cosine
    def forward(self, x):
        b, st, c, h, w = x.size()
        #print(f'x.size() = {x.size()}')
        original_shape = x[:,[0],:,:,:].shape
        #print(f'original_shape = {original_shape}')
        # original_shape = (b, 1, c, h, w)
        filter_ = self.lowpass()
        padding = int((self.kernel_size-3)/2)
        
        outputs = []
        for i in range(self.s):
            for j in range(self.t):
                x1 = x[:,[i*self.t+j],:,:,:].reshape(-1, 1, x.shape[-1])
                # x1.shape = (b*1*c*h, 1, w)
                #filter_hor = self.lowpass(i,j,'hor')
                #x1_out = F.conv1d(x1, filter_hor.view(1,1,self.kernel_size), padding='same')
                #x1_out = F.conv1d(x1, filter_.view(1,1,self.kernel_size), padding='same')
                x1_out = F.conv1d(x1, filter_.reshape(1,1,self.kernel_size), stride=3, padding=padding)
                #print(f'x1_out.shape = {x1_out.shape}')
                # x1_out.shape = (b*1*c*h, 1, w/3)
                x2 = x1_out.reshape(b,1,c,h,int(w/3)).permute(0,1,2,4,3).reshape(-1, 1, x.shape[-2])
                #print(f'x2.shape = {x2.shape}')
                #filter_ver = self.lowpass(i,j,'ver')
                #x2_out = F.conv1d(x2, filter_ver.reshape(1,1,self.kernel_size), padding='same')
                #x2_out = F.conv1d(x2, filter_.reshape(1,1,self.kernel_size), padding='same')
                x2_out = F.conv1d(x2, filter_.reshape(1,1,self.kernel_size), stride=3, padding=padding)
                #print(f'x2_out.shape = {x2_out.shape}')
                #output = x2_out.reshape(b,1,c,w,h).permute(0,1,2,4,3)
                output = x2_out.reshape(b,1,c,int(w/3),int(h/3)).permute(0,1,2,4,3)
                #outputs.append(output[:,:,:,i::self.s,j::self.t])
                outputs.append(output)
                               
        out = torch.cat(outputs, axis=1)
        # modeified
        out = torch.clamp(out,min=0,max=1)
        return out
    

    '''
    def forward(self, x):
        # implement the forward pass
        return self.conv1(x)   
    ''' 
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
    '''
    '''
    # two 1d kernels, sub-view to sub-view
    def forward(self, x):
        outputs = []
        for k in range(self.in_channels):
            o1 = self.convs_vertical[k](x[:,[k],:,:,:])
            o2 = self.convs_horizontal[k](o1)
            outputs.append(o2)

        out = torch.cat(outputs, axis=1)
        return out
    '''
    '''
    # 2d sinc
    def forward(self, x_):
        #b, st, c, h, w = x.size()
        original_shape = x_[:,[0],:,:,:].shape

        outputs = []
        for i in range(self.s):
            for j in range(self.t):
                x1 = x_[:,i*self.t+j,:,:,:]
                x1 = torch.roll(x1, shifts=(-(i-1), -(j-1)), dims=(-2,-1))
                x1_out = F.conv2d(x1, self.filter, stride=3, padding=(9,9))
                print(x1_out.shape)
                x1_out = x1_out.unsqueeze(1)
                outputs.append(x1_out)
                               
        out = torch.cat(outputs, axis=1)
        # modeified
        out = torch.clamp(out,min=0,max=1)
        return out
    '''
    
    

class FilterBankMethod(Method):
    def __init__(self, device, s=3, t=3, in_channels=9, out_channels=9, kernel_size=(1, 7, 7), stride=(1, 3, 3), model_idx=0):
        super().__init__(self.__class__.__name__+f"_{model_idx}")
        #self.net = FilterBankKernel(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride).to(device)  
        # id kernel
        self.net = FilterBankKernel(s=s, t=t, kernel_size=kernel_size).to(device)  
        self.s = s
        self.t = t
        assert self.s == self.t
        self.name = self.__class__.__name__ + f"_{model_idx}"
    def downsampling(self, hr_lf):
        for i in range(self.s):
            for j in range(self.t):
                hr_lf[:,i*self.t+j, :, :, :] = torch.roll(hr_lf[:,i*self.t+j, :, :, :], shifts=(-(i-1), -(j-1)), dims=(-2,-1))
        
        '''
        ds_lf = hr_lf[:, :, :, 1::self.s,1::self.t]
        #ds_lf = shift_images(ds_lf.reshape(b*st, c, h//3, w//3), 0.75*torch.ones(b*st).to(device), 0.75*torch.ones(b*st).to(device)).reshape(b, st, c, h//3, w//3)
        return ds_lf
        '''
        '''
        b,st,c,h,w = hr_lf.shape
        print(hr_lf.shape)
        ds_lf = hr_lf[:, :, :, 0::self.s,0::self.t]
        print(ds_lf.shape)
        for i in range(self.s):
            for j in range(self.t):
                ds_lf[:,i*self.t+j,:,:,:] = hr_lf[:, i*self.t+j, :, i::self.s,j::self.t]
        #ds_lf = shift_images(ds_lf.reshape(b*st, c, h//2, w//2), 0.75*torch.ones(b*st).to(device), 0.75*torch.ones(b*st).to(device)).reshape(b, st, c, h//2, w//2)
        return ds_lf
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
