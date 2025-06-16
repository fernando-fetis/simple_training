import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import yaml
import random
import os

from model import SimpleNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar configuración:
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Cargar modelo:
model = SimpleNN().to(device)
model.load_state_dict(torch.load(config["save_model_path"], map_location=device))
model.eval()

# Dataset de test:
transform = transforms.ToTensor()
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Inferencia:
os.makedirs("outputs", exist_ok=True)
for i in range(5):
    idx = random.randint(0, len(test_set) - 1)
    image, label = test_set[idx]
    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).item()

    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Real: {label} — Predicha: {pred}")
    plt.axis("off")
    plt.savefig(f"outputs/pred_{i}_real{label}_pred{pred}.png")
    print(f"Guardado: outputs/pred_{i}_real{label}_pred{pred}.png")
    plt.close()