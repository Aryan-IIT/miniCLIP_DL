import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, RobertaModel
import torchvision.models as models
from src.config import Config

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=Config.PROJECTION_DIM, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected # Residual connection
        x = self.layer_norm(x)
        return x

class VisionEncoder(nn.Module):
    def __init__(self, model_name=Config.VISION_MODEL_NAME, model_type=Config.VISION_MODEL_TYPE, pretrained=True):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "vit":
            # Respect Config mapping for ViT model name and embed dim
            model_cfg = Config.VISION_MODELS.get(model_type, {})
            model_name = model_name or model_cfg.get("name")
            self.hidden_dim = model_cfg.get("embed_dim", Config.VISION_EMBED_DIM)

            if pretrained:
                self.model = ViTModel.from_pretrained(model_name)
            else:
                vit_config = ViTConfig.from_pretrained(model_name)
                self.model = ViTModel(vit_config)

            # Freeze ViT backbone to fine-tune only the projection head
            for param in self.model.parameters():
                param.requires_grad = False

            self.projection = ProjectionHead(self.hidden_dim)
            
        elif model_type == "resnet":
            # Load specific ResNet version
            if "resnet101" in model_name:
                weights = models.ResNet101_Weights.DEFAULT if pretrained else None
                self.model = models.resnet101(weights=weights)
            else:
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                self.model = models.resnet50(weights=weights)
            
            # Determine feature dim from the original fc layer
            feat_dim = getattr(self.model.fc, "in_features", Config.VISION_EMBED_DIM)
            self.model.fc = nn.Identity()
            self.projection = ProjectionHead(feat_dim)

    def set_trainable(self):
        """
        Locks early layers, trains semantic layers.
        """
        print(f"🔓 setting {self.model_type} trainable layers...")
        
        if self.model_type == "resnet":
            # Freeze shallow layers (textures/edges)
            for param in self.model.parameters():
                param.requires_grad = False
            
            # UNFREEZE Layer 4 (Semantic Concepts)
            for param in self.model.layer4.parameters():
                param.requires_grad = True
                
            # UNFREEZE AttnPool/FC/Projection
            for param in self.projection.parameters():
                param.requires_grad = True

    def forward(self, x):
        if self.model_type == "vit":
            output = self.model(x)
            features = output.last_hidden_state[:, 0, :]
        else:
            features = self.model(x)
        return self.projection(features)


class TextEncoder(nn.Module):
    def __init__(self, model_name=Config.TEXT_MODEL, pretrained=True):
        super().__init__()
        self.model = RobertaModel.from_pretrained(model_name) if pretrained else RobertaModel(Config())
        self.projection = ProjectionHead(Config.TEXT_EMBED_DIM)

    def set_trainable(self):
        print("🔓 setting Text backbone trainable layers...")
        
        # Freeze all
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze last 4 layers (RoBERTa needs more depth to adapt to captions)
        for layer in self.model.encoder.layer[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
                
        # Unfreeze Pooler
        if hasattr(self.model, 'pooler') and self.model.pooler:
            for param in self.model.pooler.parameters():
                param.requires_grad = True

        for param in self.projection.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]
        return self.projection(cls_token)


class MiniCLIP(nn.Module):
    def __init__(self, vision_encoder, text_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.Parameter(torch.tensor(2.6592))

    def forward(self, image, input_ids, attention_mask):
        image_features = self.vision_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        image_embeddings = image_features / image_features.norm(dim=1, keepdim=True)
        text_embeddings = text_features / text_features.norm(dim=1, keepdim=True)
        
        logit_scale = self.temperature.exp()
        logits = (image_embeddings @ text_embeddings.T) * logit_scale
        
        return logits
