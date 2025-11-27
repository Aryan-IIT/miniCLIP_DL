import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SymmetricLoss(nn.Module):
    """
    Computes the contrastive loss for CLIP.
    It calculates the CrossEntropyLoss for both:
    1. Images -> Text (Which text matches this image?)
    2. Text -> Images (Which image matches this text?)
    """
    def __init__(self):
        super(SymmetricLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits):
        # logits shape: (batch_size, batch_size)
        batch_size = logits.shape[0]
        
        # The diagonal elements are the correct pairs (image_i matches text_i)
        labels = torch.arange(batch_size).to(logits.device)
        
        # Loss for image -> text
        loss_i = self.loss_fn(logits, labels)
        
        # Loss for text -> image
        loss_t = self.loss_fn(logits.T, labels)
        
        # Return average
        return (loss_i + loss_t) / 2

class AvgMeter:
    """Helper class to keep track of average metrics."""
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"

def show_image(image, title=None):
    # Un-normalize for display if using standard ImageNet stats
    # (Simplified for quick visualization)
    image = image.permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()