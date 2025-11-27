"""Partial fine-tuning script where only the top blocks remain trainable."""

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

from src.config_partial import Config
from src.dataset import make_loader
from src.model import VisionEncoder, TextEncoder, MiniCLIP
from src.utils import SymmetricLoss, AvgMeter


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def freeze_for_partial_finetune(model):
    # Freeze everything by default
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze projection heads and temperature
    for module in [model.vision_encoder.projection, model.text_encoder.projection]:
        for param in module.parameters():
            param.requires_grad = True
    model.temperature.requires_grad = True

    # Helper to unfreeze last two transformer blocks + final LayerNorm
    def unfreeze_last_transformer_blocks(transformer_module, ln_module=None):
        if hasattr(transformer_module, "layer"):
            blocks = transformer_module.layer
        elif hasattr(transformer_module, "encoder") and hasattr(transformer_module.encoder, "layer"):
            blocks = transformer_module.encoder.layer
        else:
            raise ValueError("Unexpected transformer module structure.")

        for block in blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True

        if ln_module is not None:
            for param in ln_module.parameters():
                param.requires_grad = True

    unfreeze_last_transformer_blocks(
        model.vision_encoder.model.encoder,
        model.vision_encoder.model.layernorm,
    )
    unfreeze_last_transformer_blocks(
        model.text_encoder.model.encoder,
        getattr(model.text_encoder.model, "layer_norm", None),
    )


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    loss_meter = AvgMeter()
    tqdm_object = tqdm(loader, total=len(loader), desc=f"[Partial] Train Epoch {epoch+1}")

    for batch in tqdm_object:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        logits = model(image, input_ids, attention_mask)
        loss = criterion(logits)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), image.size(0))
        tqdm_object.set_postfix(train_loss=loss_meter.avg)

    return loss_meter.avg


def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AvgMeter()
    tqdm_object = tqdm(loader, total=len(loader), desc="[Partial] Validation")

    with torch.no_grad():
        for batch in tqdm_object:
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(image, input_ids, attention_mask)
            loss = criterion(logits)
            loss_meter.update(loss.item(), image.size(0))
            tqdm_object.set_postfix(val_loss=loss_meter.avg)

    return loss_meter.avg


def get_embeddings(model, loader, device):
    base_model = unwrap_model(model)
    base_model.eval()
    image_embeddings = []
    text_embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Partial] Embeddings"):
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            img_emb = base_model.vision_encoder(image)
            txt_emb = base_model.text_encoder(input_ids, attention_mask)

            img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=1, keepdim=True)

            image_embeddings.append(img_emb)
            text_embeddings.append(txt_emb)

    return torch.cat(image_embeddings), torch.cat(text_embeddings)


def calculate_recall(image_embeddings, text_embeddings, k_list=(1, 5, 10)):
    logits = image_embeddings @ text_embeddings.T
    n = logits.shape[0]
    labels = torch.arange(n).to(logits.device)
    metrics = {}

    _, preds_i2t = logits.topk(max(k_list), dim=1)
    for k in k_list:
        correct = preds_i2t[:, :k].eq(labels.view(-1, 1))
        metrics[f"val/I2T_R@{k}"] = correct.sum(dim=1).float().mean().item()

    _, preds_t2i = logits.T.topk(max(k_list), dim=1)
    for k in k_list:
        correct = preds_t2i[:, :k].eq(labels.view(-1, 1))
        metrics[f"val/T2I_R@{k}"] = correct.sum(dim=1).float().mean().item()

    return metrics


def main():
    print("Loading Data...")
    if not os.path.exists(Config.CAPTIONS_PATH):
        raise FileNotFoundError(f"Dataset missing at {Config.CAPTIONS_PATH}")

    df = pd.read_csv(Config.CAPTIONS_PATH).dropna(subset=["caption"])
    df["caption"] = df["caption"].astype(str)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    if Config.FAST_DEV_RUN:
        train_df = train_df.sample(
            n=min(len(train_df), Config.FAST_DEV_TRAIN_SAMPLES), random_state=42
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.FAST_DEV_VAL_SAMPLES), random_state=42
        ).reset_index(drop=True)
        print(f"[FAST_DEV_RUN] train={len(train_df)} | val={len(val_df)}")

    train_loader = make_loader(train_df, Config.IMAGE_PATH, mode="train")
    val_loader = make_loader(val_df, Config.IMAGE_PATH, mode="val")

    print("Initializing MiniCLIP (partial fine-tuning)...")
    vision_enc = VisionEncoder()
    text_enc = TextEncoder()
    model = MiniCLIP(vision_enc, text_enc).to(Config.DEVICE)

    if not Config.PRETRAINED_CKPT.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found at {Config.PRETRAINED_CKPT}")
    state_dict = torch.load(Config.PRETRAINED_CKPT, map_location=Config.DEVICE)
    model.load_state_dict(state_dict)
    print(f"Loaded base weights from {Config.PRETRAINED_CKPT}")

    freeze_for_partial_finetune(model)

    if Config.DEVICE == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel enabled across {torch.cuda.device_count()} GPUs")

    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    head_params = list(model.vision_encoder.projection.parameters())
    head_params += list(model.text_encoder.projection.parameters())
    head_params.append(model.temperature)
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids]

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": Config.HEAD_LEARNING_RATE},
        {"params": backbone_params, "lr": Config.BACKBONE_LEARNING_RATE},
    ], weight_decay=Config.WEIGHT_DECAY)

    criterion = SymmetricLoss()
    best_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        print(f"\n[Partial] Epoch {epoch+1}/{Config.EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE, epoch)
        val_loss = validate(model, val_loader, criterion, Config.DEVICE)

        print("Computing recall metrics...")
        img_embs, txt_embs = get_embeddings(model, val_loader, Config.DEVICE)
        recall_metrics = calculate_recall(img_embs, txt_embs)

        print("-" * 40)
        print(f"[Partial] Epoch {epoch+1} Summary")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        for key, value in recall_metrics.items():
            print(f"{key}: {value:.2%}")
        print("-" * 40)

        checkpoint_path = Config.CHECKPOINT_DIR / f"partial_epoch_{epoch+1:02d}.pt"
        torch.save(unwrap_model(model).state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(unwrap_model(model).state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved to {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
