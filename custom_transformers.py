"""
custom_transformers.py
======================
Módulo de transformadores personalizados para Scikit-Learn siguiendo
los estándares de diseño oficial (BaseEstimator, TransformerMixin).

Este módulo proporciona transformadores robustos para la automatización de
Ingeniería de Características (Feature Engineering) en datos meteorológicos
y de series de tiempo/registros históricos.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class WeatherFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado de Scikit-Learn para automatizar la ingeniería
    de características sobre datos meteorológicos y temporales.

    Genera al menos 4 nuevas variables derivadas de alto impacto:
    1. Variables Temporales:
       - 'hora_del_dia': Hora de la medición (0 - 23). Captura el ciclo circadiano.
       - 'dia_de_la_semana': Día de la semana (0 = Lunes, 6 = Domingo).
       - 'es_fin_de_semana': Indicador binario (1 = Sábado o Domingo, 0 = Día de semana).
    2. Variable Termo-Higrométrica:
       - 'punto_rocio_aprox': Aproximación de Lawrence del punto de rocío a partir
         de temperatura y humedad. Clave para evaluar saturación de vapor y probabilidad de lluvia.
       - 'sensacion_termica': Índice Humidex calculado con la presión de vapor,
         representando la temperatura percibida.
       - 'ratio_humedad_temperatura': Ratio humedad/temperatura como proxy de saturación térmica.
    3. Agrupación Lógica y Discretización:
       - 'es_viento_moderado_fuerte': Variable binaria basada en la escala Beaufort
         (>= umbral configurado, típicamente 12.0 km/h).
    4. Métrica Relativa sin Data Leakage:
       - 'desviacion_temp_media': Diferencia de cada registro respecto a la temperatura
         media observada *exclusivamente* durante el entrenamiento (fit).

    Parámetros
    ----------
    timestamp_col : str, por defecto 'timestamp'
        Nombre de la columna que contiene la fecha/hora.
    temp_col : str, por defecto 'temperature_c'
        Nombre de la columna de temperatura en grados Celsius.
    humidity_col : str, por defecto 'humidity_pct'
        Nombre de la columna de humedad relativa en porcentaje (0 a 100).
    wind_col : str, por defecto 'wind_speed_kmh'
        Nombre de la columna de velocidad del viento en km/h.
    wind_threshold : float, por defecto 12.0
        Umbral para clasificar viento moderado/fuerte (Escala Beaufort brisa moderada).
    drop_original_timestamp : bool, por defecto False
        Indica si se debe eliminar la columna original de timestamp tras la extracción.
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
        self.timestamp_col = timestamp_col
        self.temp_col = temp_col
        self.humidity_col = humidity_col
        self.wind_col = wind_col
        self.wind_threshold = wind_threshold
        self.drop_original_timestamp = drop_original_timestamp

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        """
        Ajusta el transformador aprendiendo los parámetros del conjunto de entrenamiento.

        Para garantizar la ausencia de Data Leakage, cualquier estadística global
        (como la temperatura media del set de entrenamiento) se calcula aquí y se
        almacena en atributos con sufijo '_' según las convenciones de Scikit-Learn.

        Parámetros
        ----------
        X : pd.DataFrame
            Conjunto de datos de entrenamiento.
        y : None, opcional
            Variable objetivo (no utilizada en transformaciones no supervisadas).

        Retorna
        -------
        self : WeatherFeatureEngineer
            Instancia ajustada del transformador.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas DataFrame.")

        # Validar existencia de columnas requeridas
        required_cols = [self.timestamp_col, self.temp_col, self.humidity_col, self.wind_col]
        missing_cols = [c for c in required_cols if c not in X.columns]
        if missing_cols:
            raise KeyError(f"Faltan columnas requeridas en el DataFrame de entrada: {missing_cols}")

        # Aprender parámetros exclusivamente a partir del conjunto de entrenamiento
        self.mean_temp_ = float(X[self.temp_col].mean())
        self.median_humidity_ = float(X[self.humidity_col].median())
        self.mean_wind_speed_ = float(X[self.wind_col].mean())

        # Registrar nombres de columnas de entrada observadas
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica las transformaciones de ingeniería de características sobre el DataFrame.

        Buenas prácticas aplicadas:
        - Utiliza X.copy() para evitar mutaciones indeseadas sobre el DataFrame original.
        - Utiliza los parámetros aprendidos en fit() sin filtrar información de prueba.

        Parámetros
        ----------
        X : pd.DataFrame
            DataFrame a transformar (entrenamiento o prueba).

        Retorna
        -------
        X_out : pd.DataFrame
            DataFrame con las nuevas variables derivadas agregadas.
        """
        # Verificar que el transformador haya sido ajustado previamente
        check_is_fitted(self, attributes=["is_fitted_", "mean_temp_"])

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas DataFrame.")

        # Validar presencia de columnas requeridas
        required_cols = [self.timestamp_col, self.temp_col, self.humidity_col, self.wind_col]
        missing_cols = [c for c in required_cols if c not in X.columns]
        if missing_cols:
            raise KeyError(f"Faltan columnas requeridas en el DataFrame a transformar: {missing_cols}")

        # Evitar modificar el DataFrame original (Error de copia común prevenido)
        X_out = X.copy()

        # -------------------------------------------------------------
        # 1. Variables Temporales / Calendario
        # -------------------------------------------------------------
        ts_series = pd.to_datetime(X_out[self.timestamp_col])
        X_out["hora_del_dia"] = ts_series.dt.hour
        X_out["dia_de_la_semana"] = ts_series.dt.dayofweek
        X_out["es_fin_de_semana"] = (X_out["dia_de_la_semana"] >= 5).astype(int)

        # -------------------------------------------------------------
        # 2. Variables Físico-Meteorológicas (Termo-higrométricas)
        # -------------------------------------------------------------
        # Aproximación de Lawrence del punto de rocío (°C)
        # T - ((100 - H) / 5)
        punto_rocio = X_out[self.temp_col] - ((100.0 - X_out[self.humidity_col]) / 5.0)
        X_out["punto_rocio_aprox"] = punto_rocio

        # Presión de vapor de saturación (hPa) y Sensación Térmica (Humidex)
        # e = 6.11 * exp(5417.753 * (1/273.16 - 1/(273.15 + T_dew)))
        # humidex = T + 5/9 * (e - 10)
        vapor_pressure = 6.11 * np.exp(
            5417.753 * ((1.0 / 273.16) - (1.0 / (273.15 + punto_rocio)))
        )
        X_out["sensacion_termica"] = X_out[self.temp_col] + (5.0 / 9.0) * (vapor_pressure - 10.0)

        # Ratio entre humedad y temperatura (evitando división por cero)
        X_out["ratio_humedad_temperatura"] = X_out[self.humidity_col] / (X_out[self.temp_col] + 0.1)

        # -------------------------------------------------------------
        # 3. Agrupación Lógica y Discretización (Viento)
        # -------------------------------------------------------------
        # Indicador de viento moderado o fuerte según escala Beaufort
        X_out["es_viento_moderado_fuerte"] = (
            X_out[self.wind_col] >= self.wind_threshold
        ).astype(int)

        # -------------------------------------------------------------
        # 4. Métrica Relativa con Parámetro de Entrenamiento (Cero Leakage)
        # -------------------------------------------------------------
        # Usa self.mean_temp_ calculado exclusivamente en fit(X_train)
        X_out["desviacion_temp_media"] = X_out[self.temp_col] - self.mean_temp_

        # Eliminar timestamp original si se solicitó
        if self.drop_original_timestamp:
            X_out = X_out.drop(columns=[self.timestamp_col])

        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Retorna los nombres de las columnas resultantes de la transformación.
        """
        check_is_fitted(self, attributes=["is_fitted_"])
        if input_features is None:
            input_features = list(self.feature_names_in_)
        else:
            input_features = list(input_features)

        new_features = [
            "hora_del_dia",
            "dia_de_la_semana",
            "es_fin_de_semana",
            "punto_rocio_aprox",
            "sensacion_termica",
            "ratio_humedad_temperatura",
            "es_viento_moderado_fuerte",
            "desviacion_temp_media",
        ]

        output_features = list(input_features)
        if self.drop_original_timestamp and self.timestamp_col in output_features:
            output_features.remove(self.timestamp_col)

        output_features.extend(new_features)
        return np.array(output_features, dtype=object)


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformador modular enfocado exclusivamente en extraer componentes
    de calendario y tiempo a partir de una columna de fechas.
    """

    def __init__(self, datetime_col: str = "timestamp", drop_original: bool = False):
        self.datetime_col = datetime_col
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas DataFrame.")
        if self.datetime_col not in X.columns:
            raise KeyError(f"Columna '{self.datetime_col}' no encontrada en X.")
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["is_fitted_"])
        X_out = X.copy()
        ts = pd.to_datetime(X_out[self.datetime_col])
        X_out["hora_del_dia"] = ts.dt.hour
        X_out["dia_de_la_semana"] = ts.dt.dayofweek
        X_out["es_fin_de_semana"] = (X_out["dia_de_la_semana"] >= 5).astype(int)
        if self.drop_original:
            X_out = X_out.drop(columns=[self.datetime_col])
        return X_out


class ThermalIndexTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador modular enfocado en la física atmosférica y variables
    termo-higrométricas (punto de rocío, sensación térmica y ratio de humedad).
    """

    def __init__(self, temp_col: str = "temperature_c", humidity_col: str = "humidity_pct"):
        self.temp_col = temp_col
        self.humidity_col = humidity_col

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas DataFrame.")
        for col in [self.temp_col, self.humidity_col]:
            if col not in X.columns:
                raise KeyError(f"Columna '{col}' no encontrada en X.")
        self.mean_temp_ = float(X[self.temp_col].mean())
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["is_fitted_", "mean_temp_"])
        X_out = X.copy()
        t = X_out[self.temp_col]
        h = X_out[self.humidity_col]

        # Punto de rocío
        dew_point = t - ((100.0 - h) / 5.0)
        X_out["punto_rocio_aprox"] = dew_point

        # Presión de vapor y Humidex
        vapor_pressure = 6.11 * np.exp(5417.753 * ((1.0 / 273.16) - (1.0 / (273.15 + dew_point))))
        X_out["sensacion_termica"] = t + (5.0 / 9.0) * (vapor_pressure - 10.0)

        # Ratio
        X_out["ratio_humedad_temperatura"] = h / (t + 0.1)

        # Desviación respecto al promedio de entrenamiento
        X_out["desviacion_temp_media"] = t - self.mean_temp_
        return X_out
