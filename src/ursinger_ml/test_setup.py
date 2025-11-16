import numpy as np
import pandas as pd
import librosa
import sklearn

def main():
    print("✅ Entorno listo para UrSinger-ML")
    print("Versión numpy:", np.__version__)
    print("Versión pandas:", pd.__version__)
    print("Versión librosa:", librosa.__version__)
    print("Versión scikit-learn:", sklearn.__version__)

if __name__ == "__main__":
    main()
