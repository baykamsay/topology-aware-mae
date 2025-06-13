"""
Classes for loading and processing datasets for segementation tasks.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RoadsSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None):
        """
        Args:
            root_dir (str): Directory with all the images and masks.
                            Assumes subdirectories 'images' and 'masks'.
            image_transform (callable, optional): Optional transform to be applied on an image.
            mask_transform (callable, optional): Optional transform to be applied on a mask.
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        
        self.image_filenames = sorted(os.listdir(self.image_dir))
        
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        # Assuming mask has the same filename in the mask_dir
        mask_name = os.path.join(self.mask_dir, self.image_filenames[idx])

        try:
            image = Image.open(img_name).convert('RGB')
            mask = Image.open(mask_name).convert('L') # Load as grayscale
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find image or mask for index {idx}: {self.image_filenames[idx]}. Original error: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading image/mask {self.image_filenames[idx]} at index {idx}. Original error: {e}")


        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Default mask transformation if none provided
            # Convert PIL Image to Tensor, scales to [0,1]
            mask_to_tensor = transforms.ToTensor()
            mask = mask_to_tensor(mask) # Shape: [1, H, W], values [0.0, 1.0]
            # Ensure it's binary 0.0 or 1.0 and float
            mask = (mask > 0.5).float() # If 255 became 1.0, this keeps it 1.0. If 0 became 0.0, it stays 0.0.

        return image, mask