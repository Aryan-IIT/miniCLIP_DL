"""Shared training configuration for the experiments scripts."""

import os
import torch
from pathlib import Path


def _get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Config:
    # Data Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_ROOT = BASE_DIR / "data" / "flickr30k"
    IMAGE_PATH = str(DATASET_ROOT / "Images")
    CAPTIONS_PATH = str(DATASET_ROOT / "captions.txt")
    
    # MODEL SELECTION
    VISION_MODEL_TYPE = "vit_small"  # options: vit_base, vit_small
    
    # Model Names / Dimensions
    VISION_MODELS = {
        "vit_base": {"name": "google/vit-base-patch16-224", "embed_dim": 768},
        "vit_small": {"name": "facebook/dino-vits16", "embed_dim": 384},
    }
    VISION_MODEL_NAME = VISION_MODELS[VISION_MODEL_TYPE]["name"]
    TEXT_MODEL = "roberta-base"
    
    # Training Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 5e-4
    PATIENCE = 2 
    
    # Fast dev run options (set FAST_DEV_RUN=True for tiny subset debugging)
    FAST_DEV_RUN = False
    FAST_DEV_TRAIN_SAMPLES = 1024
    FAST_DEV_VAL_SAMPLES = 256
    
    # Dimensions
    VISION_EMBED_DIM = VISION_MODELS[VISION_MODEL_TYPE]["embed_dim"]
    TEXT_EMBED_DIM = 768  # RoBERTa base
    
    PROJECTION_DIM = 256  # Shared embedding space size
    MAX_LEN = 128 
    
    # Compute / Data loading
    DEVICE = _get_default_device()
    NUM_WORKERS = min(8, (os.cpu_count() or 2))
    
    # Saving
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    MODEL_SAVE_PATH = CHECKPOINT_DIR / "best_miniclip.pt"
    # MODEL_SAVE_PATH = CHECKPOINT_DIR / "epoch_02.pt"
