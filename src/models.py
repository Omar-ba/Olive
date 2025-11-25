import os
import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes):
    """
    Crée un modèle ResNet18 propre pour l'entraînement
    et cohérent pour l'inférence.
    """
    # Charger ResNet18 pré-entraîné sur ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Adapter la dernière couche fully-connected
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def load_model(model_path, num_classes, device="cpu", weights_only=True):
    """
    Charge un modèle ResNet18 EXACTEMENT comme utilisé à l'entraînement.
    Évite les erreurs 'unexpected key' et les inversions de classes.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Fichier modèle introuvable : {model_path}")

    model = get_model(num_classes)

    state = torch.load(model_path, map_location=device, weights_only=weights_only)

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    return model


def save_model(model, save_dir="models", model_name="model"):
    """
    Sauvegarde propre et standardisée.
    Compatible avec inference.py
    """
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)

    print(f"✅ Modèle sauvegardé : {save_path}")
