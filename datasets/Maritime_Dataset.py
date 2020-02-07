import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np


class Maritime_Dataset(data.Dataset):
    def __init__(self, root, bboxes_df, filename_df,
     transform=None, img_size = (1080,1920)):
        self.root = root
        self.transform = transform
        self.bboxes_df = bboxes_df
        self.filename_df = filename_df
        self.H, self.W = img_size

    def __getitem__(self, index):
        img_id = self.filename_df.iloc[index,0]
        bboxes = self.bboxes_df[self.bboxes_df.fname == img_id]

        img = cv2.imread(osp.join(self.root, img_id))
        
        bbox = bboxes.iloc[:,1:5].values
        labels = bboxes.iloc[:,5].values
        if self.transform is not None:
            annotation = {'image': img, 'bboxes': bbox, 'category_id': labels}
            augmentation = self.transform(**annotation)
            img = augmentation['image']
            bbox = augmentation['bboxes']
            labels = augmentation['category_id']
#         print(bbox)
        return {'image': img, 'bboxes': bbox, 'category_id': labels} 


    def __len__(self):
        return len(self.filename_df)
