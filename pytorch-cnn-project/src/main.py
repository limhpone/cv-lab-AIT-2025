import torch
from models.simple_cnn import SimpleCNN

if __name__ == "__main__":
    model = SimpleCNN()
    print(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")