import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
from PIL import Image

from encoder.vanilla_ae import AutoEncoder

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
loss_fn = torch.nn.MSELoss()

# Define transforms (adjust size and normalization if needed)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # adjust based on model
    transforms.ToTensor(),          # converts to [C, H, W] and float in [0, 1]
])

# Load image and preprocess
img = Image.open('data/images/1/frame0.jpg').convert('RGB')  # ensure 3 channels
img_tensor = transform(img).unsqueeze(0).to(device)  # [1, C, H, W]

# Load model
model = AutoEncoder()
model.load_state_dict(torch.load("checkpoints/0806_1745_model.pth", map_location=device))
model.to(device)
model.eval()

# Forward pass
with torch.inference_mode():
    pred = model(img_tensor)

# Compute loss
print(f"Loss: {loss_fn(pred, img_tensor).item():.6f}")

# Convert back to image
to_pil = transforms.ToPILImage()

img_pil = to_pil(img_tensor.squeeze(0).cpu().clamp(0, 1))
pred_pil = to_pil(pred.squeeze(0).cpu().clamp(0, 1))

# Show images
img_pil.show(title="Original")
pred_pil.show(title="Reconstruction")
