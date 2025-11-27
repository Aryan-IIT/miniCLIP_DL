"""Utility to precompute embeddings for the Streamlit demo."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parent.parent
TRIAL_DIR = REPO_DIR / "trial"
sys.path.append(str(TRIAL_DIR))

from src.config import Config as TrainConfig
from src.dataset import FlickrDataset
from src.model import VisionEncoder, TextEncoder, MiniCLIP
from src.evaluate import get_embeddings as eval_get_embeddings


def resolve_path(path_like: str) -> Path:
    """Allow users to pass either absolute or repo-relative paths."""
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_DIR / path
    return path


def load_model(checkpoint: Path):
    # Infer backbone from checkpoint filename (resnet vs vit)
    ck_name = checkpoint.stem.lower()
    model_type = "resnet" if "resnet" in ck_name else "vit"

    vision_enc = VisionEncoder(model_type=model_type)
    text_enc = TextEncoder()
    model = MiniCLIP(vision_enc, text_enc)

    state = torch.load(checkpoint, map_location="cpu")
    
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def collect_embeddings(model, loader, dataset_images, dataset_captions, device="cpu"):
    image_embeddings = []
    text_embeddings = []
    image_ids = []
    captions = []

    model = model.to(device)
    offset = 0
    for batch in tqdm(loader, desc="Building embeddings"):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        img_emb = model.vision_encoder(images)
        txt_emb = model.text_encoder(input_ids, attention_mask)

        img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=1, keepdim=True)

        image_embeddings.append(img_emb.cpu())
        text_embeddings.append(txt_emb.cpu())
        batch_size = images.size(0)
        image_ids.extend(dataset_images[offset : offset + batch_size])
        captions.extend(dataset_captions[offset : offset + batch_size])
        offset += batch_size

    return {
        "image_embeddings": torch.cat(image_embeddings),
        "text_embeddings": torch.cat(text_embeddings),
        "image_ids": image_ids,
        "captions": captions,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute dataset embeddings for the Streamlit app.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(TRIAL_DIR / "checkpoints_partial" / "best_miniclip_partial.pt"),
        help="Model checkpoint to load.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "data" / "embeddings.pt"),
        help="Path to save the embeddings file.",
    )
    parser.add_argument(
        "--captions",
        type=str,
        default=str(TrainConfig.CAPTIONS_PATH),
        help="Path to the captions CSV/TSV file.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=str(TrainConfig.IMAGE_PATH),
        help="Directory containing dataset images.",
    )
    parser.add_argument("--batch-size", type=int, default=TrainConfig.BATCH_SIZE)
    parser.add_argument("--device", type=str, default=TrainConfig.DEVICE)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of samples for quick tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = resolve_path(args.checkpoint)
    output_path = resolve_path(args.output)
    captions_path = resolve_path(args.captions)
    image_root = resolve_path(args.image_root)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found at {captions_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found at {image_root}")

    print(f"Loading dataset from {captions_path}")
    df = pd.read_csv(captions_path).dropna(subset=["caption"])
    if args.limit:
        df = df.head(args.limit)

    dataset = FlickrDataset(df, str(image_root))

    # Resolve device (support 'cuda', 'mps', 'cpu')
    requested = args.device or TrainConfig.DEVICE
    if requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if requested == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available; falling back to CPU.")
        if requested == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=TrainConfig.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Loading model from {checkpoint}")
    model = load_model(checkpoint)

    # If the checkpoint/backbone is ResNet, reuse the evaluation helper
    # `get_embeddings` which returns (image_embeddings, text_embeddings)
    ck_name = checkpoint.stem.lower()
    if "resnet" in ck_name:
        print("Using evaluation helper to extract ResNet embeddings...")
        img_embs, txt_embs = eval_get_embeddings(model, loader, device=device)
        embeddings = {
            "image_embeddings": img_embs.cpu(),
            "text_embeddings": txt_embs.cpu(),
            "image_ids": dataset.df["image"].tolist(),
            "captions": dataset.df["caption"].tolist(),
        }
    else:
        embeddings = collect_embeddings(
            model,
            loader,
            dataset.df["image"].tolist(),
            dataset.df["caption"].tolist(),
            device=device,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    main()
