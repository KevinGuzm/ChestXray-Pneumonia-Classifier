import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from utils import PneumoniaCNN

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directorios
train_dir = './data/train/'

# Parámetros
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Transformaciones
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Cargar datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# Modelo
model = PneumoniaCNN().to(device)

# Función de pérdida y optimizador
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Evaluación del modelo con AUROC
def evaluate_model_auroc(model, loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images).squeeze()
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return roc_auc_score(all_labels, all_predictions)

# Entrenamiento del modelo
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_predictions.extend(outputs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_auroc = roc_auc_score(all_labels, all_predictions)
        val_auroc = evaluate_model_auroc(model, val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Train AUROC: {train_auroc:.4f}, Val AUROC: {val_auroc:.4f}")

# Entrenar y guardar el modelo
train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
torch.save(model.state_dict(), 'model_weights.pth')
print("Model weights saved as 'model_weights.pth'")