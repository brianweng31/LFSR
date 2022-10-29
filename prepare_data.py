import numpy as np
from PIL import Image
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset as ConcatDataset
from torchvision import transforms
from skimage import transform
from math import floor

torch.manual_seed(0)

def generate_lf_from_img(img, disparity, ang_res=[2, 2], shape=None):
    h, w, c = img.shape
    H, W = shape[0], shape[1]
    #print(f'h,w = {h},{w}')
    #print(f'H,W = {H},{W}')
    margin = [(h-H)//2, (w-W)//2]
    s, t = ang_res[0], ang_res[1]
    
    shift_vec = []
    for s_idx in range(s):
        for t_idx in range(t):
            shift_vec.append((s_idx, t_idx))
  
    fixedPoints = np.array([[0, 0], [w/2, 0], [w, 0], [0, h/2], [w/2, h/2], [w, 0], [0, h], [w/2, h], [w, h]])
    center_point = np.tile(np.array([w/2, h/2]), [9, 1])
    img_warps = []
    for img_idx in range(len(shift_vec)):
        movingPoints = (center_point+disparity*np.tile(shift_vec[img_idx], [9, 1]))/(h/2)*(fixedPoints - center_point) + center_point
        tform = transform.estimate_transform('polynomial', movingPoints[:,[-1,-2]], fixedPoints[:,[-1,-2]])
        img_warps.append(transform.warp(img, tform, mode = 'constant'))

    lightfield = np.array(img_warps).reshape(s,t,h,w,c)

    if (h-H) % 2 == 1:
        lightfield = lightfield[:,:,margin[0]:-margin[0]-1,margin[1]:-margin[1]-1,:]
    else: #(h-H) % 2 == 0
        lightfield = lightfield[:,:,margin[0]:-margin[0],margin[1]:-margin[1],:]

    return lightfield

def generate_random_pixel_imgs(height, width, img_num = 1):
    R_value = list(range(0,256))
    G_value = list(range(0,256))
    B_value = list(range(0,256))

    RGB_possibles = np.array(np.meshgrid(R_value, G_value, B_value)).T.reshape(-1,3)
    RGB_possibles = RGB_possibles[np.random.permutation(RGB_possibles.shape[0]),:]
    

    img = RGB_possibles[:height*width,:]
    return img.reshape(height, width, 3)/255

class LFDataset(Dataset):
    def __init__(self, light_field_dataset_path, used_index, light_field_size = [3,3,510,510,3], disparity_range=range(-5,6,1)):
        self.s = light_field_size[0]
        self.t = light_field_size[1]
        assert self.s == self.t
        self.using_index = used_index
        self.lfdata_folder_path = [x[0] for x in os.walk(light_field_dataset_path)]
        self.lfdata_folder_path.pop(0)
        self.light_field_size = light_field_size
        self.disparity_range = disparity_range
        self.transform = transforms.Compose([
                transforms.Resize(size=(light_field_size[2], light_field_size[3]))
            ])


    def __len__(self):
        return len(self.lfdata_folder_path)

    def __getitem__(self, idx):
        light_field = np.zeros(self.light_field_size)
        for i in range(self.s):
            for j in range(self.t):
                subview = Image.open(os.path.join(self.lfdata_folder_path[idx], "input_Cam%03d.png" % self.using_index[i][j]))
                light_field[i,j,:,:,:] = self.transform(subview)
        return (torch.tensor(light_field)/255).type(torch.float32)

class SR_test_Dataset(Dataset):
    def __init__(self, light_field_dataset_path, light_field_size = [3,3,1080,1920,3], disparity_range=range(0,1,1), use_transform=True):
        self.s = light_field_size[0]
        self.t = light_field_size[1]
        assert self.s == self.t
        self.light_field_dataset_path = light_field_dataset_path
        self.lfdata_file_path = [x for x in os.walk(light_field_dataset_path)]
        self.lfdata_file_path = self.lfdata_file_path[0][2]
        print(self.lfdata_file_path)
        self.light_field_size = light_field_size
        self.disparity_range = disparity_range
        self.use_transform = use_transform
        if use_transform:
            self.transform = transforms.Compose([
                    transforms.Resize(size=(light_field_size[2], light_field_size[3]))
                ])
    def __len__(self):
        return len(self.lfdata_file_path)

    def __getitem__(self, idx):
        light_field = np.zeros(self.light_field_size)
        print(self.light_field_size)
        for i in range(self.s):
            for j in range(self.t):
                #print(self.lfdata_file_path[idx])
                subview = Image.open(os.path.join(self.light_field_dataset_path, self.lfdata_file_path[idx])).convert('RGB')
                #print(subview.size)
                if self.use_transform:
                    #print(self.transform(subview).size)
                    light_field[i,j,:,:,:] = self.transform(subview)

        return (torch.tensor(light_field)/255).type(torch.float32)

class RandomLFDataset(Dataset):
    def __init__(self, light_field_dataset_path=None, used_index=None, light_field_size=[3,3,512,512,3], disparity_range=range(-5,6,1), img_num = 100):
        np.random.seed(0)
        self.light_field_size = light_field_size
        self.disparity_range = disparity_range
        self.max_disparity = np.max(np.abs(self.disparity_range)).astype(int)

        np.random.seed(0)
        self.rand_seed = np.random.randint(img_num*img_num, size=img_num)
        self.rand_imgs = []

        for idx in range(len(self.rand_seed)):
            print("Generating Random Image %d..." % idx)
            np.random.seed(self.rand_seed[idx])
            self.rand_imgs.append(generate_random_pixel_imgs(self.light_field_size[2]+self.max_disparity, self.light_field_size[3]+self.max_disparity))
    
    def __len__(self):
        return len(self.rand_imgs)

    def __getitem__(self, idx):
        #print(f'self.light_field.shape = {self.light_field_size}')
        light_field = generate_lf_from_img(self.rand_imgs[idx], self.max_disparity, self.light_field_size[0:2], shape=self.light_field_size[2:4])
        #print(f'light_field.shape = {light_field.shape}')
        return (torch.tensor(light_field)).type(torch.float32)

class CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets) -> None:
        super().__init__(datasets)
        self.disparity_range = datasets[0].disparity_range

def get_dataloaders(dataset_name, batch_size=4, shuffle=True, num_workers=4, downsample_rate=1):
    if dataset_name == "HCI":
        light_field_size = [3, 3, floor(512/downsample_rate)//3*3, floor(512/downsample_rate)//3*3, 3]
        #disparity_range = np.arange(-5,6)/downsample_rate
        disparity_range = np.arange(0,1)/downsample_rate

        train_lfdataset_1 = LFDataset(
            "../Datasets/4D_Light_Field_Benchmark/additional", 
            [[0,4,8],[36,40,44],[72,76,80]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        train_lfdataset_2 = LFDataset(
            "../Datasets/4D_Light_Field_Benchmark/training", 
            [[0,4,8],[36,40,44],[72,76,80]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        test_lfdataset = LFDataset(
            "../Datasets/4D_Light_Field_Benchmark/test", 
            [[0,4,8],[36,40,44],[72,76,80]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)

        train_lfdataset = CustomConcatDataset([train_lfdataset_1, train_lfdataset_2])

        train_dataloader = DataLoader(train_lfdataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_dataloader = DataLoader(test_lfdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    if dataset_name == "INRIA_Lytro":
        light_field_size = [3, 3, floor(379/downsample_rate)//3*3, floor(379/downsample_rate)//3*3, 3]
        disparity_range = np.arange(-5,6)/downsample_rate

        train_lfdataset = LFDataset(
            "../Datasets/INRIADataset_Lytro1G/Training", 
            [[0,3,6],[21,24,27],[42,45,48]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        test_lfdataset = LFDataset(
            "../Datasets/INRIADataset_Lytro1G/Testing", 
            [[0,3,6],[21,24,27],[42,45,48]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        
        train_dataloader = DataLoader(train_lfdataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_dataloader = DataLoader(test_lfdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    
    if dataset_name == "Stanford_Archive":
        light_field_size = [3, 3, floor(800/downsample_rate)//3*3, floor(1400/downsample_rate)//3*3, 3]
        disparity_range = np.arange(-5,6)/downsample_rate

        train_lfdataset = LFDataset(
            "/content/gdrive/MyDrive/LF_Dataset/TheNewStanfordLightFieldArchive/Training", 
            [[0,8,16],[136,144,152],[272,280,288]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        test_lfdataset = LFDataset(
            "/content/gdrive/MyDrive/LF_Dataset/TheNewStanfordLightFieldArchive/Testing", 
            [[0,8,16],[136,144,152],[272,280,288]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        
        train_dataloader = DataLoader(train_lfdataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_dataloader = DataLoader(test_lfdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    
    if dataset_name == "RandomTraining":
        light_field_size = [3, 3, floor(512/downsample_rate)//3*3, floor(512/downsample_rate)//3*3, 3]
        disparity_range = np.arange(-5,6)/downsample_rate

        train_lfdataset = RandomLFDataset(
            light_field_size = light_field_size,
            disparity_range = disparity_range,
            img_num = 10)
        test_lfdataset_1 = LFDataset(
            "../Datasets/4D_Light_Field_Benchmark/test", 
            [[0,4,8],[36,40,44],[72,76,80]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        test_lfdataset_2 = LFDataset(
            "../Datasets/INRIADataset_Lytro1G/Testing", 
            [[0,3,6],[21,24,27],[42,45,48]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        
        test_lfdataset = CustomConcatDataset([test_lfdataset_1, test_lfdataset_2])
        
        train_dataloader = DataLoader(train_lfdataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_dataloader = DataLoader(test_lfdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if dataset_name == "SR_test_dataset":
        light_field_size = [3, 3, floor(1080/downsample_rate)//3*3, floor(1920/downsample_rate)//3*3, 3]
        disparity_range = np.arange(0,1)/downsample_rate

        train_lfdataset = SR_test_Dataset(
            "../Datasets/SR_test_dataset",
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        test_lfdataset = SR_test_Dataset(
            "../Datasets/SR_test_dataset",
            light_field_size = light_field_size,
            disparity_range = disparity_range)

        train_dataloader = DataLoader(train_lfdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dataloader = DataLoader(test_lfdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    return train_dataloader, test_dataloader