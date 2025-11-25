import os
import sys
from PIL import Image

# Configuration des chemins
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import setup_paths

from src.inference import predict_complete_pipeline, debug_class_orders


def main():
    """Test complet du pipeline de détection"""

    print("🧪 TEST DU PIPELINE DE DÉTECTION D'OLIVIERS")
    print("=" * 60)

    # Dossier de test
    test_folder = "C:\\Users\\P16v GEN1\\Desktop\\maching learning\\PROGET\\Olive\\Olive\\data\\test"

    if not os.path.exists(test_folder):
        print(f"❌ Dossier de test '{test_folder}' non trouvé.")
        print("📁 Création du dossier 'test_images'...")
        os.makedirs(test_folder)
        print("✅ Veuillez ajouter vos images de test dans 'test_images/' et relancer le script.")
        return

    # Lister les images de test
    test_images = [f for f in os.listdir(test_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not test_images:
        print(f"❌ Aucune image trouvée dans '{test_folder}'")
        print("📸 Veuillez ajouter des images dans le dossier 'test_images/'")
        return

    print(f"🔍 {len(test_images)} image(s) de test trouvée(s)")
    print("=" * 60)

    # Tester chaque image
    for i, image_file in enumerate(test_images, 1):
        image_path = os.path.join(test_folder, image_file)

        print(f"\n📁 Image {i}/{len(test_images)}: {image_file}")
        print("-" * 40)

        try:
            # Pipeline complet
            result = predict_complete_pipeline(image_path)

            if 'error' in result:
                print(f"❌ Erreur: {result['error']}")
                continue

            # Affichage des résultats
            print(f"🌳 Est un olivier: {'✅ OUI' if result['is_olive'] else '❌ NON'}")
            print(f"📊 Confiance classification: {result['tree_confidence'] * 100:.2f}%")

            if result['is_olive']:
                status_emoji = "🚨" if result['is_diseased'] else "✅"
                print(f"💊 État de santé: {status_emoji} {result['health_status']}")
                print(f"📊 Confiance maladie: {result['disease_confidence'] * 100:.2f}%")

            # Debug optionnel (décommentez pour activer)
            # debug_class_orders(image_path)

        except Exception as e:
            print(f"❌ Erreur lors du traitement: {e}")

    print("\n" + "=" * 60)
    print("✅ TEST TERMINÉ")


def test_single_image(image_path):
    """Test une image spécifique avec debug détaillé"""
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        return

    print(f"🧪 TEST DÉTAILLÉ: {os.path.basename(image_path)}")
    print("=" * 60)

    # Debug des ordres de classes
    debug_class_orders(image_path)

    # Pipeline complet
    result = predict_complete_pipeline(image_path)

    print("\n📊 RÉSULTAT FINAL:")
    print(f"🌳 Olive: {result['is_olive']} (confiance: {result['tree_confidence'] * 100:.2f}%)")
    if result['is_olive']:
        print(f"💊 Santé: {result['health_status']} (confiance: {result['disease_confidence'] * 100:.2f}%)")


if __name__ == "__main__":
    # Test de toutes les images dans le dossier test_images
    main()

    # Pour tester une image spécifique avec debug détaillé, décommentez:
    # test_single_image("chemin/vers/votre/image.jpg")