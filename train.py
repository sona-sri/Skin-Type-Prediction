import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.1
LABEL_INDEX = {"dry": 0, "normal": 1, "oily": 2}

# Dataset Class
class SkinDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Function to create DataFrame
def create_df(base):
    dd = {"images": [], "labels": []}
    for label in os.listdir(base):
        label_path = os.path.join(base, label)
        for img in os.listdir(label_path):
            img_path = os.path.join(label_path, img)
            dd["images"].append(img_path)
            dd["labels"].append(LABEL_INDEX[label])
    return pd.DataFrame(dd)

# Main Training Function
def train_model():
    # Load Data
    train_df = create_df("data/Oily-Dry-Skin-Types/train")
    val_df = create_df("data/Oily-Dry-Skin-Types/valid")
    train, val = train_test_split(train_df, test_size=0.2, random_state=42)

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = SkinDataset(train, transform)
    val_ds = SkinDataset(val, transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model Definition
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(LABEL_INDEX))
    device = "cpu"  # Always use CPU
    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    best_acc = 0.0

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = total_correct / len(train_ds)
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(train_loader)}, Accuracy: {train_acc}')

        # Validate
        model.eval()
        total_val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total_val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = total_val_correct / len(val_ds)
        print(f'Validation Accuracy: {val_acc}')

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')  # Save only the model state dict

if __name__ == "__main__":
    train_model()
