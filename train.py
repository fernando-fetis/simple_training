import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
import wandb

from model import SimpleNN

# Cargar configuración:
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Inicializar W&B:
wandb.init(project=config["wandb"]["project"], config=config)

# Datos de entrenamiento:
transform = transforms.ToTensor()
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)

# Modelo:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config["lr"])

# Entrenamiento:
for epoch in range(config["epochs"]):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})

# Guardar modelo:
torch.save(model.state_dict(), config["save_model_path"])