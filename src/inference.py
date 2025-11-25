from PIL import Image
import torch
from torchvision import transforms
import os
import sys

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict(image_path, model_path, classes, weights_only=True):
    """
    Fonction de prédiction CORRECTE qui respecte l'ordre RÉEL des classes
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    model = get_model(len(classes)).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=weights_only)
    )
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)[0]

        pred_index = torch.argmax(probabilities).item()
        confidence = probabilities[pred_index].item()

    return {
        "class": classes[pred_index],
        "confidence": confidence,
        "all_probabilities": probabilities.tolist(),
        "class_index": pred_index
    }


def predict_tree(image_path):
    """
    Prédit olive / non olive selon l'ordre RÉEL du dataset
    Dataset ImageFolder crée toujours un mapping ALPHABÉTIQUE :
    0 = Not Olive
    1 = Olive
    """
    classes = [ "Olive","Not Olive"]
    model_path = "models/tree_model.pth"
    return _predict(image_path, model_path, classes)


def predict_disease(image_path):
    """
    Prédit healthy / diseased selon l'ordre RÉEL du dataset
    ImageFolder :
    0 = Diseased
    1 = Healthy
    """
    classes = ["Diseased", "Healthy"]
    model_path = "models/disease_model.pth"
    return _predict(image_path, model_path, classes)


def predict_complete_pipeline(image_path):
    """
    Pipeline complet :
    1. Olive ou pas
    2. Si olive -> maladie
    """
    try:
        # Étape 1 : olive / non olive
        tree_result = predict_tree(image_path)
        is_olive = tree_result["class"] == "Olive"

        result = {
            "image_path": image_path,
            "is_olive": is_olive,
            "tree_confidence": tree_result["confidence"],
            "tree_class": tree_result["class"],
            "tree_class_index": tree_result["class_index"]
        }

        # Étape 2 : healthy / diseased
        if is_olive:
            disease_result = predict_disease(image_path)
            is_diseased = disease_result["class"] == "Diseased"

            result.update({
                "is_diseased": is_diseased,
                "disease_confidence": disease_result["confidence"],
                "health_status": disease_result["class"],
                "disease_class": disease_result["class"],
                "disease_class_index": disease_result["class_index"]
            })
        else:
            result.update({
                "is_diseased": None,
                "disease_confidence": None,
                "health_status": "Non applicable",
                "disease_class": None,
                "disease_class_index": None
            })

        return result

    except Exception as e:
        return {
            "error": str(e),
            "image_path": image_path
        }


def debug_class_orders(image_path):
    """
    Debug simple : montre le prédict réel sans inversion
    """
    print("\n=== DEBUG TREE ===")
    print(predict_tree(image_path))

    print("\n=== DEBUG DISEASE ===")
    print(predict_disease(image_path))
