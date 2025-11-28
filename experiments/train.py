"""Baseline training loop for the experiments suite."""

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import wandb

from src.config import Config
from src.dataset import make_loader
from src.model import VisionEncoder, TextEncoder, MiniCLIP
from src.utils import SymmetricLoss, AvgMeter


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def train_epoch(model, loader, optimizer, criterion, device, epoch, scheduler=None):
    model.train()
    loss_meter = AvgMeter()
    
    tqdm_object = tqdm(loader, total=len(loader), desc=f"Training Epoch {epoch+1}")
    for step, batch in enumerate(tqdm_object):
        image = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(image, input_ids, attention_mask)
        
        # Compute loss
        loss = criterion(logits)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        loss_meter.update(loss.item(), image.size(0))
        tqdm_object.set_postfix(train_loss=loss_meter.avg)
        
        # Log batch-level loss for granular detail
        wandb.log({
            "train/batch_loss": loss.item(),
            "train/learning_rate": optimizer.param_groups[0]['lr']
        })
        
    return loss_meter.avg

def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AvgMeter()
    
    tqdm_object = tqdm(loader, total=len(loader), desc="Validation")
    with torch.no_grad():
        for batch in tqdm_object:
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
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
    
    print("Extracting embeddings for metrics...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embeddings"):
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            img_emb = base_model.vision_encoder(image)
            text_emb = base_model.text_encoder(input_ids, attention_mask)
            
            img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
            
            image_embeddings.append(img_emb)
            text_embeddings.append(text_emb)
            
    return torch.cat(image_embeddings), torch.cat(text_embeddings)

def calculate_recall(image_embeddings, text_embeddings, k_list=[1, 5, 10]):
    logits = image_embeddings @ text_embeddings.T
    n = logits.shape[0]
    labels = torch.arange(n).to(logits.device)
    metrics = {}
    
    # Image-to-Text
    _, preds_i2t = logits.topk(max(k_list), dim=1)
    for k in k_list:
        correct = preds_i2t[:, :k].eq(labels.view(-1, 1))
        metrics[f"val/I2T_R@{k}"] = correct.sum(dim=1).float().mean().item()
        
    # Text-to-Image
    _, preds_t2i = logits.T.topk(max(k_list), dim=1)
    for k in k_list:
        correct = preds_t2i[:, :k].eq(labels.view(-1, 1))
        metrics[f"val/T2I_R@{k}"] = correct.sum(dim=1).float().mean().item()
        
    return metrics

def main():
    # --- WANDB INIT ---
    wandb.init(
        project="MiniCLIP-Course-Project",
        config={
            "vision_model": Config.VISION_MODEL_TYPE,
            "text_model": Config.TEXT_MODEL,
            "epochs": Config.EPOCHS,
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.LEARNING_RATE,
            "projection_dim": Config.PROJECTION_DIM
        }
    )
    
    print("Loading Data...")
    if not os.path.exists(Config.CAPTIONS_PATH):
        print(f"Error: Dataset not found at {Config.CAPTIONS_PATH}")
        return

    df = pd.read_csv(Config.CAPTIONS_PATH)
    if 'image' not in df.columns or 'caption' not in df.columns:
        # Fallback logic if needed
        pass
    
    df = df.dropna(subset=['caption'])
    df['caption'] = df['caption'].astype(str)
    
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    if Config.FAST_DEV_RUN:
        train_limit = min(len(train_df), Config.FAST_DEV_TRAIN_SAMPLES)
        val_limit = min(len(val_df), Config.FAST_DEV_VAL_SAMPLES)
        train_df = train_df.sample(n=train_limit, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(n=val_limit, random_state=42).reset_index(drop=True)
        print(f"[FAST_DEV_RUN] Using {len(train_df)} training and {len(val_df)} validation samples for smoke test.")
    
    train_loader = make_loader(train_df, Config.IMAGE_PATH, mode="train")
    val_loader = make_loader(val_df, Config.IMAGE_PATH, mode="val")
    
    print(f"Initializing Models ({Config.VISION_MODEL_TYPE} + RoBERTa)...")
    vision_enc = VisionEncoder() 
    text_enc = TextEncoder()
    model = MiniCLIP(vision_enc, text_enc).to(Config.DEVICE)
    # Load previously saved weights so additional runs continue fine-tuning rather than restarting.
    if Config.MODEL_SAVE_PATH.exists():
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint)
        print(f"Resumed weights from {Config.MODEL_SAVE_PATH}")
    else:
        print("No existing checkpoint detected; training from scratch.")
    if Config.DEVICE == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel enabled across {torch.cuda.device_count()} GPUs")

    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Watch model gradients
    wandb.watch(model, log="all")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader)
    )
    criterion = SymmetricLoss()
    
    best_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE, epoch, scheduler)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Metrics
        print("Calculating Quantitative Metrics (Recall@K)...")
        img_embs, txt_embs = get_embeddings(model, val_loader, Config.DEVICE)
        recall_metrics = calculate_recall(img_embs, txt_embs)
        
        # Print Console Summary
        print("-" * 30)
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"I2T R@1: {recall_metrics['val/I2T_R@1']:.2%} | T2I R@1: {recall_metrics['val/T2I_R@1']:.2%}")
        print("-" * 30)

        # --- LOG EPOCH METRICS TO WANDB ---
        wandb_log_dict = {
            "epoch": epoch + 1,
            "train/epoch_loss": train_loss,
            "val/epoch_loss": val_loss,
        }
        wandb_log_dict.update(recall_metrics) # Add recall scores
        wandb.log(wandb_log_dict)

        checkpoint_path = Config.CHECKPOINT_DIR / f"epoch_{epoch+1:02d}.pt"
        torch.save(unwrap_model(model).state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(unwrap_model(model).state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Saved Best Model to {Config.MODEL_SAVE_PATH}")
            
    wandb.finish()

if __name__ == "__main__":

    main()
