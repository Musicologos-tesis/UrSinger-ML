"""
Script principal para generar el dataset de entrenamiento.

Uso:
    python -m ursinger_ml.preprocessing.build_dataset

O desde la raíz del proyecto:
    python -m src.ursinger_ml.preprocessing.build_dataset
"""

if __name__ == "__main__":
    from ursinger_ml.preprocessing.build_dataset import build_training_dataset, save_training_dataset
    import pandas as pd

    # Configurar pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    print("=" * 70)
    print("GENERADOR DE DATASET DE ENTRENAMIENTO - UrSinger ML")
    print("=" * 70)

    # Construir dataset
    df = build_training_dataset()

    # Mostrar resumen
    print("\n" + "=" * 70)
    print("RESUMEN DEL DATASET")
    print("=" * 70)

    print(f"\nDimensiones: {len(df)} filas x {len(df.columns)} columnas")
    print(f"\nColumnas: {', '.join(df.columns.tolist())}")

    print(f"\nPrimeras 10 filas:")
    print(df.head(10))

    print(f"\n\nDistribucion por cantante:")
    dist = df.groupby('singer_id').size()
    print(dist)

    print(f"\n\nDistribucion por genero:")
    print(df['gender'].value_counts())

    print(f"\n\nDistribucion por vocal:")
    print(df['vowel'].value_counts().sort_index())

    # Mostrar estadísticas de las métricas
    print(f"\n\nESTADISTICAS DE METRICAS")
    print("=" * 70)
    metric_columns = [col for col in df.columns if col not in ['singer_id', 'gender', 'vowel']]
    for col in metric_columns:
        print(f"\n{col}:")
        print(df[col].describe())

    # Guardar dataset
    print("\n" + "=" * 70)
    save_training_dataset(df)
    print("=" * 70)
    print("\nDataset generado exitosamente!")

