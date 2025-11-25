import argparse
import os
import sys

# Ajouter automatiquement la racine du projet
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

import setup_paths

from src.train_tree import train_tree
from src.train_disease import train_disease


def main():
    parser = argparse.ArgumentParser(description="Entraînement des modèles Olive AI")

    parser.add_argument(
        "--model",
        type=str,
        choices=["tree", "disease", "both"],
        default="both",
        help="Sélection du modèle à entraîner"
    )

    parser.add_argument("--epochs", type=int, default=15, help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille des batchs")

    parser.add_argument(
        "--data_dir_tree",
        type=str,
        default=os.path.join("data", "tree_classification"),
        help="Dossier des données olive / non olive"
    )

    parser.add_argument(
        "--data_dir_disease",
        type=str,
        default=os.path.join("data", "leaf_disease"),
        help="Dossier des données maladie healthy / diseased"
    )

    args = parser.parse_args()

    print("🚀 LANCEMENT DE L'ENTRAÎNEMENT OLIVE-AI")
    print("=" * 60)
    print(f"📌 Modèle demandé : {args.model}")
    print(f"📌 Epochs        : {args.epochs}")
    print(f"📌 Batch Size    : {args.batch_size}")
    print("=" * 60)

    try:
        # ✔ Entraînement olive / non olive
        if args.model in ["tree", "both"]:
            print("\n🌳 ENTRAÎNEMENT DU MODÈLE OLIVE / NON-OLIVE")
            print("-" * 60)
            train_tree(
                data_dir=args.data_dir_tree,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=1e-4
            )

        # ✔ Entraînement maladie
        if args.model in ["disease", "both"]:
            print("\n🩺 ENTRAÎNEMENT DU MODÈLE HEALTHY / DISEASED")
            print("-" * 60)
            train_disease(
                data_dir=args.data_dir_disease,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=1e-5
            )

        print("\n✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
        print("📁 Les modèles sauvegardés se trouvent dans : models/")

    except Exception as e:
        print("\n❌ ERREUR LORS DE L'ENTRAÎNEMENT:")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
