import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_dataloaders(data_dir, batch_size=32, val_split=0.2, augment=True):
    """
    Charge et prépare les données avec data augmentation
    """
    if augment:
        # Transformation avec data augmentation pour l'entraînement
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Transformation simple
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Transformation pour la validation (sans augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Charger le dataset complet
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    # Afficher les informations du dataset
    print(f"📊 Dataset: {len(full_dataset)} images")
    print(f"📁 Classes: {full_dataset.classes}")
    for class_name, class_idx in full_dataset.class_to_idx.items():
        class_count = sum(1 for _, label in full_dataset.samples if label == class_idx)
        print(f"   {class_name}: {class_count} images")

    # Split en train/validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Appliquer la transformation de validation au dataset de validation
    val_dataset.dataset.transform = val_transform

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, full_dataset.classes