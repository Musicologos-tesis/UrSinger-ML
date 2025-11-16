from pathlib import Path

# Ajusta esta ruta si tu VocalSet está en otro sitio.
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # sube desde src/ursinger_ml/ hasta la raíz
DATA_DIR = PROJECT_ROOT / "data"
VOCALSET_ROOT = DATA_DIR / "vocalset"

if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    print("Data dir:", DATA_DIR)
    print("VocalSet root:", VOCALSET_ROOT)
    print("Existe VocalSet?:", VOCALSET_ROOT.exists())
