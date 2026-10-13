"""
test_transformers.py
====================
Suite complementaria de pruebas para verificar la compatibilidad de CustomTransformer
con datasets heterogéneos y sintéticos.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from custom_transformers import CustomTransformer, WeatherFeatureEngineer


def test_custom_transformer_with_generic_ecommerce_data():
    """Verifica que CustomTransformer funcione con un dataset de fechas y montos."""
    df = pd.DataFrame(
        {
            "fecha": pd.date_range("2026-01-01", periods=10, freq="D").astype(str),
            "monto": [100.0, 150.0, 200.0, 80.0, 120.0, 300.0, 250.0, 90.0, 110.0, 140.0],
            "categoria": ["A", "B", "A", "C", "B", "A", "C", "A", "B", "C"],
        }
    )

    transformer = CustomTransformer(date_col="fecha", numeric_col="monto")
    transformed = transformer.fit_transform(df)

    assert "hora_del_dia" in transformed.columns
    assert "dia_de_la_semana" in transformed.columns
    assert "es_fin_de_semana" in transformed.columns
    assert "desviacion_respecto_media" in transformed.columns
    assert len(transformed) == 10


def test_custom_transformer_auto_detection():
    """Verifica la detección automática de columnas cuando no se especifican nombres."""
    df = pd.DataFrame(
        {
            "date": ["2026-05-10 14:00:00", "2026-05-11 18:00:00"],
            "amount": [50.0, 75.0],
        }
    )
    t = CustomTransformer()
    res = t.fit_transform(df)
    assert "dia_de_la_semana" in res.columns
    assert "desviacion_respecto_media" in res.columns
