import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import os
import sys

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_dataloaders
from src.models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_disease(data_dir="data/leaf_disease", epochs=20, batch_size=32, learning_rate=1e-5):
    """
    Entraîne le modèle de détection de maladies
    """
    print("🩺 Initialisation de l'entraînement détection de maladies...")

    # Load data avec augmentation
    train_loader, val_loader, classes = get_dataloaders(data_dir, batch_size, augment=True)
    print(f"🎯 Classes détectées: {classes}")

    # Create model
    model = get_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)  # Plus petit learning rate

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_accuracy = 0
    print(f"🚀 Début de l'entraînement pour {epochs} époques...")

    for epoch in range(epochs):
        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader))

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        # Phase de validation
        val_accuracy = validate(model, val_loader)
        scheduler.step()

        # Sauvegarder le meilleur modèle
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/disease_model.pth")
            print(f"💾 Meilleur modèle sauvegardé (Accuracy: {val_accuracy:.2f}%)")

    print(f"🩺 Entraînement terminé! Meilleure accuracy: {best_accuracy:.2f}%")
    return model


def validate(model, val_loader):
    """Calcule la précision sur le set de validation"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"📊 Validation Accuracy: {accuracy:.2f}%")
    return accuracy