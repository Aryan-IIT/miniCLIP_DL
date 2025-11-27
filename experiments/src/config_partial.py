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
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_ROOT = BASE_DIR / "data" / "flickr30k"
    IMAGE_PATH = str(DATASET_ROOT / "Images")
    CAPTIONS_PATH = str(DATASET_ROOT / "captions.txt")

    VISION_MODEL_TYPE = "vit_small"
    VISION_MODELS = {
        "vit_base": {"name": "google/vit-base-patch16-224", "embed_dim": 768},
        "vit_small": {"name": "facebook/dino-vits16", "embed_dim": 384},
    }
    VISION_MODEL_NAME = VISION_MODELS[VISION_MODEL_TYPE]["name"]
    TEXT_MODEL = "roberta-base"

    # Training parameters for partial fine-tuning
    BATCH_SIZE = 24
    EPOCHS = 6
    HEAD_LEARNING_RATE = 3e-4
    BACKBONE_LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    PATIENCE = 2

    FAST_DEV_RUN = False
    FAST_DEV_TRAIN_SAMPLES = 512
    FAST_DEV_VAL_SAMPLES = 128

    VISION_EMBED_DIM = VISION_MODELS[VISION_MODEL_TYPE]["embed_dim"]
    TEXT_EMBED_DIM = 768
    PROJECTION_DIM = 256
    MAX_LEN = 128

    DEVICE = _get_default_device()
    NUM_WORKERS = min(8, (os.cpu_count() or 2))

    PRETRAINED_CKPT = BASE_DIR / "checkpoints_full" / "best_miniclip_full.pt"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints_partial"
    MODEL_SAVE_PATH = CHECKPOINT_DIR / "best_miniclip_partial.pt"
