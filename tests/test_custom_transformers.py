"""
test_custom_transformers.py
===========================
Suite oficial de pruebas unitarias para evaluar el cumplimiento de
estándares de ingeniería de Scikit-Learn y criterios de aceptación.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Asegurar que el directorio raíz esté en sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for path in [CURRENT_DIR, PARENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from custom_transformers import (
    WeatherFeatureEngineer,
    CustomTransformer,
    TemporalFeatureExtractor,
    ThermalIndexTransformer,
)


@pytest.fixture
def sample_weather_data() -> pd.DataFrame:
    """Fixture que provee un DataFrame sintético con la misma estructura que los datos crudos."""
    np.random.seed(42)
    n_samples = 100
    timestamps = pd.date_range("2026-10-01 00:00:00", periods=n_samples, freq="h")
    return pd.DataFrame(
        {
            "id": np.arange(1, n_samples + 1),
            "city": ["Buenos Aires"] * n_samples,
            "timestamp": timestamps.astype(str),
            "temperature_c": np.random.uniform(10.0, 25.0, size=n_samples).round(1),
            "humidity_pct": np.random.uniform(40.0, 95.0, size=n_samples).round(1),
            "precipitation_mm": np.random.choice([0.0, 0.5, 2.0, 8.0], size=n_samples, p=[0.7, 0.15, 0.1, 0.05]),
            "wind_speed_kmh": np.random.uniform(3.0, 25.0, size=n_samples).round(1),
            "wind_category": ["Calma"] * n_samples,
            "is_rainy": np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25]),
        }
    )


def test_inheritance_base_estimator_and_transformer_mixin():
    """Criterio: El transformador debe seguir la estructura oficial de Scikit-Learn."""
    transformer = WeatherFeatureEngineer()
    assert isinstance(transformer, BaseEstimator), "Debe heredar de BaseEstimator."
    assert isinstance(transformer, TransformerMixin), "Debe heredar de TransformerMixin."

    alias_transformer = CustomTransformer()
    assert isinstance(alias_transformer, BaseEstimator)
    assert isinstance(alias_transformer, TransformerMixin)


def test_fit_returns_self(sample_weather_data):
    """Criterio: El método fit debe retornar self."""
    transformer = WeatherFeatureEngineer()
    ret = transformer.fit(sample_weather_data)
    assert ret is transformer, "fit() debe retornar self."
    assert hasattr(transformer, "mean_temp_"), "fit() debe calcular y guardar atributos aprendidos (con sufijo _)."


def test_fit_transform_creates_at_least_three_new_features(sample_weather_data):
    """Criterio: Diseñar y crear al menos 3 nuevas variables derivadas."""
    transformer = WeatherFeatureEngineer()
    original_cols = set(sample_weather_data.columns)
    transformed_df = transformer.fit_transform(sample_weather_data)

    nuevas_cols = set(transformed_df.columns) - original_cols
    assert len(nuevas_cols) >= 3, f"Se esperaban al menos 3 nuevas variables derivadas, se crearon {len(nuevas_cols)}: {nuevas_cols}"
    
    # Comprobar presencia de variables de alto impacto
    esperadas = ["hora_del_dia", "punto_rocio_aprox", "sensacion_termica", "es_viento_moderado_fuerte"]
    for col in esperadas:
        assert col in transformed_df.columns, f"Columna esperada '{col}' no encontrada en el DataFrame transformado."


def test_transform_prevents_dataframe_mutation_error_de_copia(sample_weather_data):
    """Error común a evitar: No usar X.copy() puede modificar el DataFrame original."""
    original_copy = sample_weather_data.copy(deep=True)
    transformer = WeatherFeatureEngineer()
    transformer.fit(sample_weather_data)
    _ = transformer.transform(sample_weather_data)

    # Verificar que el DataFrame original no fue alterado
    pd.testing.assert_frame_equal(
        sample_weather_data,
        original_copy,
        check_dtype=True,
        obj="El DataFrame original fue modificado dentro de transform() (falta X.copy())"
    )


def test_no_data_leakage_between_train_and_test(sample_weather_data):
    """Criterio: No debe haber evidencia de Data Leakage."""
    X = sample_weather_data.drop(columns=["is_rainy"])
    y = sample_weather_data["is_rainy"]

    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    transformer = WeatherFeatureEngineer()
    # Ajuste exclusivo en train
    transformer.fit(X_train)

    media_train = transformer.mean_temp_
    media_test_real = X_test["temperature_c"].mean()

    # Asegurar que las medias sean numéricamente distintas
    assert media_train != media_test_real

    # Transformar conjunto de prueba SIN reajustar (sin llamar a fit)
    X_test_transformed = transformer.transform(X_test)

    # La columna 'desviacion_temp_media' en test DEBE haber usado media_train, NO media_test_real
    desviaciones_esperadas = X_test["temperature_c"] - media_train
    np.testing.assert_allclose(
        X_test_transformed["desviacion_temp_media"],
        desviaciones_esperadas,
        err_msg="Data Leakage: transform(X_test) utilizó estadísticas del conjunto de prueba en lugar de entrenamiento."
    )


def test_fit_transform_consistency(sample_weather_data):
    """Verifica que fit_transform(X) sea equivalente a fit(X).transform(X)."""
    t1 = WeatherFeatureEngineer()
    t2 = WeatherFeatureEngineer()

    out1 = t1.fit_transform(sample_weather_data)
    out2 = t2.fit(sample_weather_data).transform(sample_weather_data)

    pd.testing.assert_frame_equal(out1, out2)


def test_integration_in_sklearn_pipeline(sample_weather_data):
    """Verifica que el transformador se integre sin problemas en un Pipeline oficial de Scikit-Learn."""
    pipeline = Pipeline(
        steps=[
            ("feature_engineer", WeatherFeatureEngineer()),
        ]
    )

    result = pipeline.fit_transform(sample_weather_data)
    assert isinstance(result, pd.DataFrame)
    assert "punto_rocio_aprox" in result.columns
    assert "sensacion_termica" in result.columns


def test_missing_required_column_raises_key_error(sample_weather_data):
    """Valida el manejo defensivo de errores si falta alguna columna esperada."""
    df_invalido = sample_weather_data.drop(columns=["temperature_c"])
    transformer = WeatherFeatureEngineer()

    with pytest.raises(KeyError):
        transformer.fit(df_invalido)
