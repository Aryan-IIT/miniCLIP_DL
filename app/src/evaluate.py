"""Quick validation script reused by experiments and the app."""

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import sys

# Add the current directory to path so we can import src
sys.path.append(".")

from src.config import Config
from src.dataset import make_loader
from src.model import VisionEncoder, TextEncoder, MiniCLIP

def get_embeddings(model, loader, device):
    """
    Extracts all embeddings from the loader to calculate full-batch metrics.
    """
    # Accept either a string like 'cpu'/'cuda'/'mps' or a torch.device
    if isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    model.eval()
    image_embeddings = []
    text_embeddings = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding"):
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 1. Get Raw Features
            img_emb = model.vision_encoder(image)
            text_emb = model.text_encoder(input_ids, attention_mask)
            
            # 2. Normalize (Crucial for Cosine Similarity)
            img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
            
            image_embeddings.append(img_emb)
            text_embeddings.append(text_emb)
            
    return torch.cat(image_embeddings), torch.cat(text_embeddings)

def calculate_recall(image_embeddings, text_embeddings, k_list=[1, 5, 10]):
    """
    Calculates Recall@K for Image-to-Text and Text-to-Image retrieval.
    """
    # 1. Compute Similarity Matrix (N_images x N_texts)
    # Since we normalized embeddings, dot product == cosine similarity
    print("Calculating similarity matrix...")
    logits = image_embeddings @ text_embeddings.T
    
    n = logits.shape[0]
    # Ground truth: The i-th image corresponds to the i-th text
    labels = torch.arange(n).to(logits.device)
    
    metrics = {}
    
    # --- Image-to-Text (I2T) ---
    # For each image, find top-K text indices
    _, preds_i2t = logits.topk(max(k_list), dim=1)
    
    for k in k_list:
        # check if the correct label is in the top K predictions
        correct = preds_i2t[:, :k].eq(labels.view(-1, 1))
        # Average over the batch
        metrics[f"I2T_R@{k}"] = correct.sum(dim=1).float().mean().item()
        
    # --- Text-to-Image (T2I) ---
    # For each text, find top-K image indices (Transpose logits)
    _, preds_t2i = logits.T.topk(max(k_list), dim=1)
    
    for k in k_list:
        correct = preds_t2i[:, :k].eq(labels.view(-1, 1))
        metrics[f"T2I_R@{k}"] = correct.sum(dim=1).float().mean().item()
        
    return metrics

def main():
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Prepare Data
    # We re-create the validation split to ensure we test on the same data
    if not os.path.exists(Config.CAPTIONS_PATH):
        print(f"Error: Dataset not found at {Config.CAPTIONS_PATH}")
        return

    df = pd.read_csv(Config.CAPTIONS_PATH)
    
    # Basic cleaning
    df = df.dropna(subset=['caption'])
    df['caption'] = df['caption'].astype(str)
    
    # Use the exact same random_state as training to get the same validation set
    _, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    print(f"Evaluating on {len(val_df)} validation pairs.")
    val_loader = make_loader(val_df, Config.IMAGE_PATH, mode="val")

    # 2. Load Model
    print("Loading Model...")
    vision_enc = VisionEncoder()
    text_enc = TextEncoder()
    model = MiniCLIP(vision_enc, text_enc).to(device)
    
    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    else:
        print(f"Error: Model weights not found at {Config.MODEL_SAVE_PATH}. Train first!")
        return

    # 3. Calculate Metrics
    img_embs, txt_embs = get_embeddings(model, val_loader, device)
    metrics = calculate_recall(img_embs, txt_embs)
    
    # 4. Print Results
    print("\n" + "="*40)
    print("   Mini-CLIP Evaluation Results")
    print("="*40)
    
    print("\n📸 Image-to-Text Retrieval (Given Image, find Caption):")
    print(f"   R@1:  {metrics['I2T_R@1']:.2%}")
    print(f"   R@5:  {metrics['I2T_R@5']:.2%}")
    print(f"   R@10: {metrics['I2T_R@10']:.2%}")
    
    print("\n📝 Text-to-Image Retrieval (Given Caption, find Image):")
    print(f"   R@1:  {metrics['T2I_R@1']:.2%}")
    print(f"   R@5:  {metrics['T2I_R@5']:.2%}")
    print(f"   R@10: {metrics['T2I_R@10']:.2%}")
    print("="*40)

if __name__ == "__main__":
    main()
