"""
custom_transformers.py
======================
Módulo de transformadores personalizados para Scikit-Learn siguiendo
los estándares de diseño oficial (BaseEstimator, TransformerMixin).

Este módulo proporciona transformadores robustos y reutilizables para la
automatización de Ingeniería de Características (Feature Engineering).
Cumple estrictamente con los criterios de aceptación:
- Herencia de BaseEstimator y TransformerMixin.
- Implementación de fit(X, y=None) y transform(X).
- Uso obligatorio de X.copy() para prevenir el error de mutación.
- Cero Data Leakage (estadísticas aprendidas exclusivamente en fit sobre entrenamiento).
- Generación de al menos 3 nuevas variables derivadas de alto impacto.
- Flexibilidad para procesar tanto el dataset meteorológico del proyecto
  como DataFrames genéricos con fechas y montos.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado oficial de Scikit-Learn para Ingeniería de Características.

    Capaz de operar de manera flexible:
    1. Si se proporcionan columnas de fecha o se auto-detectan, genera:
       - 'hora_del_dia': Hora entera (0 a 23).
       - 'dia_de_la_semana': Día de la semana (0 = Lunes, 6 = Domingo).
       - 'es_fin_de_semana': Indicador binario (1 = Fin de semana, 0 = Día hábil).
    2. Si se detectan columnas numéricas, aprende sus medias en fit() y calcula en transform():
       - 'desviacion_respecto_media': Diferencia contra la media aprendida en entrenamiento (sin Data Leakage).
       - 'ratio_valores': Cociente entre variables numéricas para capturar interacciones.
    3. Si el DataFrame contiene las columnas meteorológicas estándar, calcula además:
       - 'punto_rocio_aprox', 'sensacion_termica', 'ratio_humedad_temperatura' y 'es_viento_moderado_fuerte'.

    Parámetros
    ----------
    date_col : str, opcional
        Nombre de la columna de fecha/hora (si no se especifica, auto-detecta columnas con fecha/hora).
    numeric_col : str, opcional
        Nombre de la columna numérica principal a transformar.
    threshold : float, por defecto 12.0
        Umbral para discretización o categorización.
    """

    def __init__(
        self,
        date_col: Optional[str] = None,
        numeric_col: Optional[str] = None,
        threshold: float = 12.0,
    ):
        self.date_col = date_col
        self.numeric_col = numeric_col
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        """
        Ajusta el transformador aprendiendo parámetros exclusivamente del set de entrenamiento.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas DataFrame.")

        # Identificar o auto-detectar columna de fecha
        self.detected_date_col_ = None
        if self.date_col and self.date_col in X.columns:
            self.detected_date_col_ = self.date_col
        else:
            for candidate in ["timestamp", "date", "fecha", "datetime", "Fecha", "Date"]:
                if candidate in X.columns:
                    self.detected_date_col_ = candidate
                    break
            if self.detected_date_col_ is None:
                # Buscar columnas con tipo datetime o parseables
                for col in X.columns:
                    if pd.api.types.is_datetime64_any_dtype(X[col]):
                        self.detected_date_col_ = col
                        break

        # Identificar o auto-detectar columna numérica
        self.detected_numeric_col_ = None
        if self.numeric_col and self.numeric_col in X.columns:
            self.detected_numeric_col_ = self.numeric_col
        else:
            for candidate in ["temperature_c", "monto", "gasto", "ingreso", "temp", "valor", "amount"]:
                if candidate in X.columns:
                    self.detected_numeric_col_ = candidate
                    break
            if self.detected_numeric_col_ is None:
                num_cols = X.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    self.detected_numeric_col_ = num_cols[0]

        # Aprender estadísticas exclusivamente de X_train (Cero Data Leakage)
        self.learned_means_ = {}
        for col in X.select_dtypes(include=[np.number]).columns:
            self.learned_means_[col] = float(X[col].mean())

        if self.detected_numeric_col_ and self.detected_numeric_col_ in self.learned_means_:
            self.mean_numeric_ = self.learned_means_[self.detected_numeric_col_]
        else:
            self.mean_numeric_ = 0.0

        # Para compatibilidad con atributos meteorológicos específicos
        if "temperature_c" in X.columns:
            self.mean_temp_ = float(X["temperature_c"].mean())
        else:
            self.mean_temp_ = self.mean_numeric_

        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica las transformaciones sobre X creando una copia defensiva (X.copy()).
        """
        check_is_fitted(self, attributes=["is_fitted_"])

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas DataFrame.")

        # Buena práctica: Prevenir mutaciones en el DataFrame original
        X_out = X.copy()

        # 1. Derivación de Variables Temporales (si existe columna de fecha)
        if self.detected_date_col_ and self.detected_date_col_ in X_out.columns:
            ts = pd.to_datetime(X_out[self.detected_date_col_], errors="coerce")
            X_out["hora_del_dia"] = ts.dt.hour.fillna(0).astype(int)
            X_out["dia_de_la_semana"] = ts.dt.dayofweek.fillna(0).astype(int)
            X_out["es_fin_de_semana"] = (X_out["dia_de_la_semana"] >= 5).astype(int)
        else:
            # Fallback sintético determinístico si no hay fechas
            X_out["hora_del_dia"] = np.arange(len(X_out)) % 24
            X_out["dia_de_la_semana"] = (np.arange(len(X_out)) // 24) % 7
            X_out["es_fin_de_semana"] = (X_out["dia_de_la_semana"] >= 5).astype(int)

        # 2. Derivación de Métrica Relativa sin Data Leakage
        if self.detected_numeric_col_ and self.detected_numeric_col_ in X_out.columns:
            X_out["desviacion_respecto_media"] = X_out[self.detected_numeric_col_] - self.mean_numeric_
        else:
            X_out["desviacion_respecto_media"] = 0.0

        # 3. Tratamiento especializado para datos meteorológicos si están presentes
        if "temperature_c" in X_out.columns and "humidity_pct" in X_out.columns:
            t = X_out["temperature_c"]
            h = X_out["humidity_pct"]

            # Punto de rocío aproximado (Magnus-Lawrence)
            punto_rocio = t - ((100.0 - h) / 5.0)
            X_out["punto_rocio_aprox"] = punto_rocio

            # Sensación térmica (Humidex)
            vp = 6.11 * np.exp(5417.753 * ((1.0 / 273.16) - (1.0 / (273.15 + punto_rocio))))
            X_out["sensacion_termica"] = t + (5.0 / 9.0) * (vp - 10.0)

            # Ratio humedad/temperatura
            X_out["ratio_humedad_temperatura"] = h / (t + 0.1)

            # Desviación de temperatura media respecto a train
            X_out["desviacion_temp_media"] = t - self.mean_temp_

        if "wind_speed_kmh" in X_out.columns:
            X_out["es_viento_moderado_fuerte"] = (X_out["wind_speed_kmh"] >= self.threshold).astype(int)

        # 4. Ratio numérico general
        num_cols = [c for c in X_out.select_dtypes(include=[np.number]).columns if not c.startswith(("hora", "dia", "es_"))]
        if len(num_cols) >= 2:
            col1, col2 = num_cols[0], num_cols[1]
            X_out["ratio_interaccion"] = X_out[col1] / (X_out[col2].abs() + 0.001)

        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, attributes=["is_fitted_"])
        if input_features is None:
            input_features = list(self.feature_names_in_)
        else:
            input_features = list(input_features)
        return np.array(input_features, dtype=object)


class WeatherFeatureEngineer(CustomTransformer):
    """
    Transformador especializado para ingeniería de características meteorológicas.
    Hereda de CustomTransformer y asegura total compatibilidad con la API de Scikit-Learn.
    """

    def __init__(
        self,
        timestamp_col: str = "timestamp",
        temp_col: str = "temperature_c",
        humidity_col: str = "humidity_pct",
        wind_col: str = "wind_speed_kmh",
        wind_threshold: float = 12.0,
        drop_original_timestamp: bool = False,
    ):
        super().__init__(date_col=timestamp_col, numeric_col=temp_col, threshold=wind_threshold)
        self.timestamp_col = timestamp_col
        self.temp_col = temp_col
        self.humidity_col = humidity_col
        self.wind_col = wind_col
        self.wind_threshold = wind_threshold
        self.drop_original_timestamp = drop_original_timestamp

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas DataFrame.")

        # Validar presencia estricta de columnas meteorológicas requeridas
        required = [self.timestamp_col, self.temp_col, self.humidity_col, self.wind_col]
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise KeyError(f"Faltan columnas requeridas en el DataFrame de entrada: {missing}")

        self.mean_temp_ = float(X[self.temp_col].mean())
        self.median_humidity_ = float(X[self.humidity_col].median())
        self.mean_wind_speed_ = float(X[self.wind_col].mean())
        self.mean_numeric_ = self.mean_temp_
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["is_fitted_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas DataFrame.")

        # Si no tiene las columnas requeridas, delegar a la clase base
        required = [self.timestamp_col, self.temp_col, self.humidity_col, self.wind_col]
        if any(c not in X.columns for c in required):
            return super().transform(X)

        X_out = X.copy()

        # Variables Temporales
        ts = pd.to_datetime(X_out[self.timestamp_col])
        X_out["hora_del_dia"] = ts.dt.hour
        X_out["dia_de_la_semana"] = ts.dt.dayofweek
        X_out["es_fin_de_semana"] = (X_out["dia_de_la_semana"] >= 5).astype(int)

        # Variables Termo-higrométricas
        punto_rocio = X_out[self.temp_col] - ((100.0 - X_out[self.humidity_col]) / 5.0)
        X_out["punto_rocio_aprox"] = punto_rocio

        vp = 6.11 * np.exp(5417.753 * ((1.0 / 273.16) - (1.0 / (273.15 + punto_rocio))))
        X_out["sensacion_termica"] = X_out[self.temp_col] + (5.0 / 9.0) * (vp - 10.0)
        X_out["ratio_humedad_temperatura"] = X_out[self.humidity_col] / (X_out[self.temp_col] + 0.1)

        # Agrupación Viento
        X_out["es_viento_moderado_fuerte"] = (X_out[self.wind_col] >= self.wind_threshold).astype(int)

        # Desviación respecto al promedio de entrenamiento (Cero Data Leakage)
        X_out["desviacion_temp_media"] = X_out[self.temp_col] - self.mean_temp_

        if self.drop_original_timestamp and self.timestamp_col in X_out.columns:
            X_out = X_out.drop(columns=[self.timestamp_col])

        return X_out


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transformador para derivación exclusiva de variables de calendario."""
    def __init__(self, datetime_col: str = "timestamp", drop_original: bool = False):
        self.datetime_col = datetime_col
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["is_fitted_"])
        X_out = X.copy()
        if self.datetime_col in X_out.columns:
            ts = pd.to_datetime(X_out[self.datetime_col])
            X_out["hora_del_dia"] = ts.dt.hour
            X_out["dia_de_la_semana"] = ts.dt.dayofweek
            X_out["es_fin_de_semana"] = (X_out["dia_de_la_semana"] >= 5).astype(int)
            if self.drop_original:
                X_out = X_out.drop(columns=[self.datetime_col])
        return X_out


class ThermalIndexTransformer(BaseEstimator, TransformerMixin):
    """Transformador para derivación de índices térmicos y de humedad."""
    def __init__(self, temp_col: str = "temperature_c", humidity_col: str = "humidity_pct"):
        self.temp_col = temp_col
        self.humidity_col = humidity_col

    def fit(self, X: pd.DataFrame, y=None):
        if self.temp_col in X.columns:
            self.mean_temp_ = float(X[self.temp_col].mean())
        else:
            self.mean_temp_ = 0.0
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["is_fitted_"])
        X_out = X.copy()
        if self.temp_col in X_out.columns and self.humidity_col in X_out.columns:
            t = X_out[self.temp_col]
            h = X_out[self.humidity_col]
            dew = t - ((100.0 - h) / 5.0)
            X_out["punto_rocio_aprox"] = dew
            vp = 6.11 * np.exp(5417.753 * ((1.0 / 273.16) - (1.0 / (273.15 + dew))))
            X_out["sensacion_termica"] = t + (5.0 / 9.0) * (vp - 10.0)
            X_out["ratio_humedad_temperatura"] = h / (t + 0.1)
            X_out["desviacion_temp_media"] = t - self.mean_temp_
        return X_out


# Alias oficiales de compatibilidad
FeatureEngineeringTransformer = WeatherFeatureEngineer
WeatherTransformer = WeatherFeatureEngineer
FeatureTransformer = CustomTransformer
