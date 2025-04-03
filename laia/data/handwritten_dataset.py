import json
import os
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from laia.data.image_dataset import ImageDataset


class HandwrittenDataset(ImageDataset):
    """Dataset for handwritten text line recognition."""

    def __init__(
        self,
        data_file: str,
        char_map: Dict[str, int],
        img_dir: str = "",
        img_transform=None,
    ):
        """Initialize the dataset.
        
        Args:
            data_file: Path to JSON file containing image paths and transcriptions
            char_map: Dictionary mapping characters to indices
            img_dir: Base directory for images
            img_transform: Optional transform to be applied to images
        """
        # Load data
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
        # Get list of image paths
        self.img_dir = img_dir
        img_paths = [os.path.join(img_dir, "images", item["image"]) for item in self.data]
        
        # Default image transforms if none provided
        if img_transform is None:
            img_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
        # Initialize parent class
        super().__init__(imgs=img_paths, transform=img_transform)
        
        self.char_map = char_map

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[int]]:
        """Get an item from the dataset.
        
        Returns:
            tuple: (image_tensor, text_indices)
        """
        # Get image using parent class
        img_tensor = super().__getitem__(index)["img"]
        
        # Convert text to character indices
        text = self.data[index]["text"]
        char_indices = [self.char_map[c] for c in text if c in self.char_map]
        
        return img_tensor, char_indices

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, List[int]]]) -> Tuple[torch.Tensor, List[List[int]]]:
        """Collate function for DataLoader.
        
        Args:
            batch: List of (image_tensor, text_indices) tuples
            
        Returns:
            tuple: (batched_images, text_indices_list)
        """
        images, texts = zip(*batch)
        
        # Get max dimensions
        max_height = max(img.size(-2) for img in images)
        max_width = max(img.size(-1) for img in images)
        
        # Pad images to max height and width
        padded_images = []
        for img in images:
            h_padding = max_height - img.size(-2)
            w_padding = max_width - img.size(-1)
            
            if h_padding > 0 or w_padding > 0:
                padded_images.append(
                    torch.nn.functional.pad(
                        img, 
                        (0, w_padding, 0, h_padding),  # pad last dim (W) then second-to-last (H)
                        value=0
                    )
                )
            else:
                padded_images.append(img)
                
        # Stack images into batch
        batched_images = torch.stack(padded_images)
        
        return batched_images, list(texts) 