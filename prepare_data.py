import numpy as np
from PIL import Image
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset as ConcatDataset
from torchvision import transforms
from math import floor


torch.manual_seed(0)
    
class LFDataset(Dataset):
    def __init__(self, light_field_dataset_path, used_index, light_field_size = [3,3,510,510,3], disparity_range=range(-5,6,1)):
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
        for i in range(3):
            for j in range(3):
                subview = Image.open(os.path.join(self.lfdata_folder_path[idx], "input_Cam%03d.png" % self.using_index[i][j]))
                light_field[i,j,:,:,:] = self.transform(subview)
        return (torch.tensor(light_field)/255).type(torch.float32)


class CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets) -> None:
        super().__init__(datasets)
        self.disparity_range = datasets[0].disparity_range

def get_dataloaders(dataset_name, batch_size=4, shuffle=True, num_workers=4, downsample_rate=1):
    if dataset_name == "HCI":
        light_field_size = [3, 3, floor(512/downsample_rate)//3*3, floor(512/downsample_rate)//3*3, 3]
        disparity_range = np.arange(-5,6)/downsample_rate

        '''
        train_lfdataset_1 = LFDataset(
            "/content/gdrive/MyDrive/LF_Dataset/4D_Light_Field_Benchmark/additional", 
            [[0,4,8],[36,40,44],[72,76,80]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        train_lfdataset_2 = LFDataset(
            "/content/gdrive/MyDrive/LF_Dataset/4D_Light_Field_Benchmark/training", 
            [[0,4,8],[36,40,44],[72,76,80]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        test_lfdataset = LFDataset(
            "/content/gdrive/MyDrive/LF_Dataset/4D_Light_Field_Benchmark/test", 
            [[0,4,8],[36,40,44],[72,76,80]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        '''
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
        disparity_range = np.arange(-3,4)/downsample_rate

        train_lfdataset = LFDataset(
            "/content/gdrive/MyDrive/LF_Dataset/INRIADataset_Lytro1G/Training", 
            [[0,3,6],[21,24,27],[42,45,48]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        test_lfdataset = LFDataset(
            "/content/gdrive/MyDrive/LF_Dataset/INRIADataset_Lytro1G/Testing", 
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

    
    if dataset_name == "SRFilterTraining":
        light_field_size = [3, 3, floor(512/downsample_rate)//3*3, floor(512/downsample_rate)//3*3, 3]
        disparity_range = np.arange(-3,4)/downsample_rate

        train_lfdataset = LFDataset(
            "/content/gdrive/MyDrive/LF_Dataset/SRFilterTraining/Training", 
            [[0,1,2],[3,4,5],[6,7,8]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        test_lfdataset_1 = LFDataset(
            "/content/gdrive/MyDrive/LF_Dataset/4D_Light_Field_Benchmark/test", 
            [[0,4,8],[36,40,44],[72,76,80]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        test_lfdataset_2 = LFDataset(
            "/content/gdrive/MyDrive/LF_Dataset/INRIADataset_Lytro1G/Testing", 
            [[0,3,6],[21,24,27],[42,45,48]],
            light_field_size = light_field_size,
            disparity_range = disparity_range)
        
        test_lfdataset = CustomConcatDataset([test_lfdataset_1, test_lfdataset_2])
        
        train_dataloader = DataLoader(train_lfdataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_dataloader = DataLoader(test_lfdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    return train_dataloader, test_dataloader