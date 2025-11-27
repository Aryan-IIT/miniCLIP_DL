"""Quick matplotlib demo to sanity-check a trained MiniCLIP checkpoint."""

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from src.config import Config
from src.model import VisionEncoder, TextEncoder, MiniCLIP
from transformers import ViTImageProcessor, RobertaTokenizer

def predict(model, image_path, texts, device):
    """
    Given an image and a list of texts, returns probability distribution.
    """
    model.eval()
    processor = ViTImageProcessor.from_pretrained(Config.VISION_MODEL_NAME)
    tokenizer = RobertaTokenizer.from_pretrained(Config.TEXT_MODEL)
    
    # Process Image
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt")
    pixel_values = image_inputs['pixel_values'].to(device)
    
    # Process Texts
    text_inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=Config.MAX_LEN, 
        return_tensors="pt"
    )
    input_ids = text_inputs['input_ids'].to(device)
    attention_mask = text_inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        # Get embeddings directly
        img_emb = model.vision_encoder(pixel_values)
        text_emb = model.text_encoder(input_ids, attention_mask)
        
        # Normalize
        img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
        
        # Calculate Similarity
        logits = (img_emb @ text_emb.T) * torch.exp(model.temperature)
        probs = F.softmax(logits, dim=1)
        
    return probs.cpu().numpy()[0]

def main():
    # Load Model
    device = Config.DEVICE
    vision_enc = VisionEncoder()
    text_enc = TextEncoder()
    model = MiniCLIP(vision_enc, text_enc).to(device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("No trained model found. Please run train.py first.")
        return

    # --- DEMO ---
    # Update these paths to a real image on your machine
    test_image = str(Config.DATASET_ROOT / "Images" / "1000268201.jpg")
    
    # List of queries (One matches, others are distractors)
    queries = [
        "A child in a pink dress is climbing a set of stairs in an entry way.",
        "A dog running on the grass.",
        "A car parked on the street.",
        "A view of the ocean."
    ]
    
    probs = predict(model, test_image, queries, device)
    
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(test_image))
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.barh(range(len(queries)), probs, color='skyblue')
    plt.yticks(range(len(queries)), queries)
    plt.xlabel("Probability")
    plt.title("CLIP Prediction")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
