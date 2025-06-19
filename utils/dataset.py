import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random
import pytorch_lightning as pl

class DeblurDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data  
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.data[idx]
        try:
            # Load images with PIL
            blur_img = Image.open(blur_path).convert("RGB")
            sharp_img = Image.open(sharp_path).convert("RGB")
            
            # Convert PIL images to numpy arrays for Albumentations
            blur_img = np.array(blur_img)
            sharp_img = np.array(sharp_img)
            
            # Apply transforms
            if self.transform:
                augmented = self.transform(image=blur_img, target=sharp_img)
                blur_img = augmented["image"].float() / 255.0  # Ensure float32
                sharp_img = augmented["target"].float() / 255.0 # Ensure float32
            else:
                # Manual conversion if no transform
                blur_img = torch.tensor(blur_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                sharp_img = torch.tensor(sharp_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
            return blur_img, sharp_img
        except Exception as e:
            print(f"Error loading image pair {idx}: blur={blur_path}, sharp={sharp_path}, error={e}")
            raise
class DeblurDataModule(pl.LightningDataModule):
    def __init__(self,data_dir,batch_size,seed=100):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for logging
        self.data_dir=data_dir
        self.batch_size=batch_size
        self.seed=seed
    def split_data(self):
        random.seed(self.seed)
        
        # Define directories
        train_blur_dir = os.path.join(self.data_dir, "train/blur")
        train_sharp_dir = os.path.join(self.data_dir, "train/sharp")
        test_blur_dir = os.path.join(self.data_dir, "test/blur")
        test_sharp_dir = os.path.join(self.data_dir, "test/sharp")
        
        # Get lists of files with full paths and ensure they are sorted
        blur_files = [os.path.join(train_blur_dir, f) for f in sorted(os.listdir(train_blur_dir))]
        sharp_files = [os.path.join(train_sharp_dir, f) for f in sorted(os.listdir(train_sharp_dir))]
        
        # Ensure the number of blur and sharp images match
        assert len(blur_files) == len(sharp_files), "Mismatch between blur and sharp image counts"
        
        # Create paired list of tuples with full paths
        temp_train_data = list(zip(blur_files, sharp_files))
        
        # Get test data with full paths
        test_blur_files = [os.path.join(test_blur_dir, f) for f in sorted(os.listdir(test_blur_dir))]
        test_sharp_files = [os.path.join(test_sharp_dir, f) for f in sorted(os.listdir(test_sharp_dir))]
        assert len(test_blur_files) == len(test_sharp_files), "Mismatch in test data"
        test_data = list(zip(test_blur_files, test_sharp_files))
        
        # Split train data into train and validation (80% train, 20% validation)
        train_data = random.sample(temp_train_data, k=int(0.8 * len(temp_train_data)))
        
        # Validation data: remaining items not in train_data
        val_data = [pair for pair in temp_train_data if pair not in train_data]
        
        return train_data, val_data, test_data
    def setup(self, stage=None):
        train_data, val_data, test_data = self.split_data()
        train_transform = A.Compose([
            A.Resize(256,256),
            A.RandomCrop(height=224, width=224), 
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ], additional_targets={"target": "image"})
        test_transform = A.Compose([
            A.Resize(224, 224), 
            ToTensorV2()
        ], additional_targets={"target": "image"})
        if stage == "fit" or stage is None:
            self.train_dataset = DeblurDataset(train_data, train_transform)
            self.val_dataset = DeblurDataset(val_data, test_transform)
        if stage == "test" or stage is None:
            self.test_dataset = DeblurDataset(test_data, test_transform)
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=3)
    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=False,num_workers=3)
    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=False)
