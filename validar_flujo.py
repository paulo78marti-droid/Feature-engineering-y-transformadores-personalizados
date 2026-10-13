"""
validar_flujo.py
================
Script de validación del flujo de trabajo de Feature Engineering y
Transformadores Personalizados para Scikit-Learn.

Ejecuta el ciclo completo:
1. Carga de datos desde la base de datos SQLite (weather_analytics.db).
2. División en conjuntos de entrenamiento y prueba (train_test_split).
3. Ajuste y transformación en entrenamiento: .fit_transform().
4. Transformación exclusiva en prueba: .transform().
5. Pruebas de verificación de buenas prácticas:
   - Prevención de mutación del DataFrame original (X.copy()).
   - Prevención estricta de Data Leakage.
   - Compatibilidad e integración en un Pipeline de Scikit-Learn.
"""

import os
import sys
from typing import Optional

# Asegurar que el directorio actual esté en el sys.path para importar módulos locales
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from custom_transformers import (
    WeatherFeatureEngineer,
    TemporalFeatureExtractor,
    ThermalIndexTransformer,
)


def cargar_datos_sqlite(db_path: Optional[str] = None) -> pd.DataFrame:
    """Carga los datos meteorológicos desde la base de datos SQLite local."""
    if db_path is None:
        db_path = os.path.join(CURRENT_DIR, "data", "weather_analytics.db")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No se encontró el archivo de base de datos en: {db_path}")

    conn = sqlite3.connect(db_path)
    query = """
        SELECT 
            id,
            city,
            timestamp,
            temperature_c,
            humidity_pct,
            precipitation_mm,
            wind_speed_kmh,
            wind_category,
            is_rainy
        FROM weather_hourly
        ORDER BY timestamp ASC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def main():
    print("=" * 70)
    print("VALIDACIÓN DEL FLUJO DE TRABAJO - TRANSFORMADORES SCIKIT-LEARN")
    print("=" * 70)

    # 1. Cargar datos
    print("\n[Paso 1] Cargando datos meteorológicos desde SQLite...")
    df = cargar_datos_sqlite()
    print(f" -> Registros cargados: {df.shape[0]}")
    print(f" -> Columnas originales: {list(df.columns)}")

    # 2. Separar características y target
    # Target: 'is_rainy' (clasificación binaria: 1 = llueve, 0 = no llueve)
    target_col = "is_rainy"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\n[Paso 2] Dividiendo dataset en entrenamiento y prueba (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f" -> X_train shape: {X_train.shape}")
    print(f" -> X_test shape:  {X_test.shape}")
    print(f" -> Distribución de clases en train: {dict(y_train.value_counts())}")
    print(f" -> Distribución de clases en test:  {dict(y_test.value_counts())}")

    # Preservar copias para verificar que no sean modificadas (Test de Inmutabilidad)
    X_train_original_copy = X_train.copy(deep=True)
    X_test_original_copy = X_test.copy(deep=True)

    # 3. Instanciar Transformador Personalizado
    print("\n[Paso 3] Inicializando WeatherFeatureEngineer()...")
    transformer = WeatherFeatureEngineer(
        timestamp_col="timestamp",
        temp_col="temperature_c",
        humidity_col="humidity_pct",
        wind_col="wind_speed_kmh",
        wind_threshold=12.0,
    )

    # 4. Ajustar y transformar en entrenamiento (.fit_transform)
    print("\n[Paso 4] Aplicando .fit_transform() sobre conjunto de ENTRENAMIENTO...")
    X_train_transformed = transformer.fit_transform(X_train)
    print(f" -> Parámetro aprendido mean_temp_ (train): {transformer.mean_temp_:.2f} °C")
    print(f" -> Parámetro aprendido median_humidity_:   {transformer.median_humidity_:.1f} %")
    print(f" -> Shape tras transformación (train):      {X_train_transformed.shape}")

    # 5. Transformar en prueba (.transform exclusivamente)
    print("\n[Paso 5] Aplicando .transform() sobre conjunto de PRUEBA (sin .fit)...")
    X_test_transformed = transformer.transform(X_test)
    print(f" -> Shape tras transformación (test):       {X_test_transformed.shape}")

    # 6. Validaciones de Calidad y Criterios de Aceptación
    print("\n" + "=" * 70)
    print("VERIFICACIÓN DE CRITERIOS DE ACEPTACIÓN")
    print("=" * 70)

    # Criterio A: Detección y creación de nuevas variables
    nuevas_vars = [
        "hora_del_dia",
        "dia_de_la_semana",
        "es_fin_de_semana",
        "punto_rocio_aprox",
        "sensacion_termica",
        "ratio_humedad_temperatura",
        "es_viento_moderado_fuerte",
        "desviacion_temp_media",
    ]
    vars_presentes = [var for var in nuevas_vars if var in X_train_transformed.columns]
    print(f"\n[Check 1] Variables derivadas creadas ({len(vars_presentes)}/8):")
    for var in vars_presentes:
        print(f"   [OK] {var}")
    assert len(vars_presentes) >= 3, "Error: Deben crearse al menos 3 variables derivadas."

    # Criterio B: Error de Copia (Inmutabilidad del DataFrame original)
    pd.testing.assert_frame_equal(X_train, X_train_original_copy)
    pd.testing.assert_frame_equal(X_test, X_test_original_copy)
    print("\n[Check 2] Error de copia prevenido: X_train y X_test originales NO fueron modificados.")

    # Criterio C: Prevención de Data Leakage
    test_temp_mean = X_test["temperature_c"].mean()
    # La desviación en test debe usar la media de TRAIN, no la de TEST
    esperado_desv_test = X_test["temperature_c"] - transformer.mean_temp_
    np.testing.assert_allclose(
        X_test_transformed["desviacion_temp_media"],
        esperado_desv_test,
        err_msg="Data Leakage detectado en desviación de temperatura media."
    )
    print(f"\n[Check 3] Cero Data Leakage validado:")
    print(f"   - Media de temperatura aprendida en Train: {transformer.mean_temp_:.3f} °C")
    print(f"   - Media real en Test (desconocida por fit): {test_temp_mean:.3f} °C")
    print(f"   - Las desviaciones en Test utilizan estrictamente la media de Train.")

    # 7. Demostración en un Pipeline de Scikit-Learn
    print("\n[Paso 6] Demostración de integración en Pipeline de Scikit-Learn...")

    # Columnas numéricas finales para el modelo predictivo
    feature_columns = [
        "temperature_c",
        "humidity_pct",
        "wind_speed_kmh",
        "hora_del_dia",
        "dia_de_la_semana",
        "es_fin_de_semana",
        "punto_rocio_aprox",
        "sensacion_termica",
        "ratio_humedad_temperatura",
        "es_viento_moderado_fuerte",
        "desviacion_temp_media",
    ]

    class FeatureSelector(WeatherFeatureEngineer):
        """Subclase para seleccionar las columnas procesadas antes del clasificador."""
        def transform(self, X):
            X_feats = super().transform(X)
            return X_feats[feature_columns]

    pipeline = Pipeline(
        steps=[
            ("feature_engineer", FeatureSelector()),
            ("classifier", RandomForestClassifier(n_estimators=50, random_state=42)),
        ]
    )

    # Entrenar pipeline completo
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f" -> Accuracy del modelo en set de prueba: {acc:.4f} ({acc*100:.1f}%)")

    # Muestra de datos transformados
    print("\n[Muestra de Variables Derivadas Generadas (Primeros 3 registros de Test)]:")
    print(X_test_transformed[nuevas_vars].head(3).to_string())

    print("\n" + "=" * 70)
    print("¡TODAS LAS VALIDACIONES PASARON EXITOSAMENTE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
