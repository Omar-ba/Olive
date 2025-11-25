import sys
import os

# Ajouter le dossier src au chemin Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
print("✅ Chemins configurés pour les imports")