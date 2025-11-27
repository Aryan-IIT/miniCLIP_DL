"""CLI helper to inspect retrieval quality on random Flickr30k samples."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from tqdm import tqdm
from transformers import ViTImageProcessor, RobertaTokenizer

from src.config_full import Config
from src.model import VisionEncoder, TextEncoder, MiniCLIP


def load_model(device):
    """Load the active checkpoint and place it on the requested device."""
    vision_enc = VisionEncoder()
    text_enc = TextEncoder()
    model = MiniCLIP(vision_enc, text_enc).to(device)
    if Path(Config.MODEL_SAVE_PATH).exists():
        state = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {Config.MODEL_SAVE_PATH}")
    model.eval()
    return model


def sample_dataset(num_samples, seed):
    if not os.path.exists(Config.CAPTIONS_PATH):
        raise FileNotFoundError(f"Captions file not found at {Config.CAPTIONS_PATH}")

    df = pd.read_csv(Config.CAPTIONS_PATH)
    df = df.dropna(subset=["image", "caption"])
    df["caption"] = df["caption"].astype(str)
    if len(df) == 0:
        raise ValueError("No valid samples found in the captions file.")
    sample_df = df.sample(n=min(num_samples, len(df)), random_state=42).reset_index(drop=True)
    return sample_df


def load_images(sample_df):
    images = []
    for img_name in sample_df["image"]:
        path = os.path.join(Config.IMAGE_PATH, img_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def encode_batch(model, images, captions, device):
    processor = ViTImageProcessor.from_pretrained(Config.VISION_MODEL_NAME)
    tokenizer = RobertaTokenizer.from_pretrained(Config.TEXT_MODEL)

    image_inputs = processor(images=images, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)

    text_inputs = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=Config.MAX_LEN,
        return_tensors="pt",
    )
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    with torch.no_grad():
        img_emb = model.vision_encoder(pixel_values)
        txt_emb = model.text_encoder(input_ids, attention_mask)

    img_emb = F.normalize(img_emb, dim=1)
    txt_emb = F.normalize(txt_emb, dim=1)
    return img_emb, txt_emb


def compute_bi_directional_scores(img_emb, txt_emb, k):
    logits = img_emb @ txt_emb.T
    i2t = logits.topk(k=k, dim=1)
    t2i = logits.T.topk(k=k, dim=1)
    return logits, i2t, t2i


def recall_at_k(logits, k_list):
    n = logits.shape[0]
    if n == 0:
        return {}
    max_k = min(max(k_list), n)
    k_vals = [k for k in k_list if k <= n]
    if not k_vals:
        k_vals = [max_k]
    labels = torch.arange(n, device=logits.device)
    metrics = {}
    
    _, preds_i2t = logits.topk(max_k, dim=1)
    for k in k_vals:
        correct = preds_i2t[:, :k].eq(labels.view(-1, 1))
        metrics[f"I2T_R@{k}"] = correct.any(dim=1).float().mean().item()
    
    _, preds_t2i = logits.T.topk(max_k, dim=1)
    for k in k_vals:
        correct = preds_t2i[:, :k].eq(labels.view(-1, 1))
        metrics[f"T2I_R@{k}"] = correct.any(dim=1).float().mean().item()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Batch inference for MiniCLIP.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of random samples to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k predictions to display.")
    args = parser.parse_args()

    device = Config.DEVICE
    model = load_model(device)
    sample_df = sample_dataset(args.num_samples, args.seed)
    images = load_images(sample_df)

    print(f"Running inference on {len(sample_df)} sampled pairs...")
    img_emb, txt_emb = encode_batch(model, images, sample_df["caption"].tolist(), device)

    logits, i2t, t2i = compute_bi_directional_scores(img_emb, txt_emb, args.topk)
    captions = sample_df["caption"].tolist()
    image_names = sample_df["image"].tolist()

    print("\nImage-to-Text predictions:")
    for idx, (scores, indices) in enumerate(zip(i2t.values, i2t.indices)):
        print(f"\nImage {idx} ({image_names[idx]}):")
        for rank, (score, jdx) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
            print(f"  Top {rank}: Caption {jdx} | Score: {score:.3f} | Text: {captions[jdx][:80]}")

    print("\nText-to-Image predictions:")
    for idx, (scores, indices) in enumerate(zip(t2i.values, t2i.indices)):
        caption = captions[idx][:80]
        print(f"\nCaption {idx}: {caption}")
        for rank, (score, jdx) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
            print(f"  Top {rank}: Image {jdx} | Score: {score:.3f} | File: {image_names[jdx]}")

    metrics = recall_at_k(logits, [1, 5, 10])
    print("\nBi-directional recall on sampled data:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2%}")


if __name__ == "__main__":
    main()
