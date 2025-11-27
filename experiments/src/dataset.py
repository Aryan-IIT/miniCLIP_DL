"""Dataset utilities shared by training scripts."""

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from transformers import ViTImageProcessor, RobertaTokenizer
from src.config import Config

class FlickrDataset(Dataset):
    def __init__(self, df, root_dir):
        self.df = df
        self.root_dir = root_dir
        self.tokenizer = RobertaTokenizer.from_pretrained(Config.TEXT_MODEL)
        
        # Stick with the processor for the active vision backbone.
        model_cfg = Config.VISION_MODELS.get(Config.VISION_MODEL_TYPE)
        if model_cfg is None:
            raise ValueError(f"Unsupported vision encoder type: {Config.VISION_MODEL_TYPE}")
        self.processor = ViTImageProcessor.from_pretrained(model_cfg["name"])
        self.transform = None  # Processor handles image normalization

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        caption = row['caption']
        
        # keep caption as string for tokenizer
        if not isinstance(caption, str):
            if pd.isna(caption):
                caption = "unknown"  # Fallback for missing data
            else:
                caption = str(caption)
        
        img_path = os.path.join(self.root_dir, img_name)
        
        # Try to open, otherwise fall back to blank image.
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        image_tensor = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        
        # 2. Process Text
        tokenized = self.tokenizer(
            caption, 
            padding="max_length", 
            max_length=Config.MAX_LEN, 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "image": image_tensor,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "caption": caption
        }

def make_loader(df, root_dir, mode="train"):
    dataset = FlickrDataset(df, root_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        shuffle=(mode == "train"),
        pin_memory=(Config.DEVICE == "cuda")
    )
    return dataloader
