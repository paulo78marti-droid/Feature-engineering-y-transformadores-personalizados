# Informe de Revisión de Entrega

**Curso:** Data Science II  
**Institución:** Coderhouse  
**Entregable:** Feature Engineering y Transformadores Personalizados  
**Estudiante:** Paulo Martí  
**Repositorio GitHub:** [https://github.com/paulo78marti-droid/Feature-engineering-y-transformadores-personalizados](https://github.com/paulo78marti-droid/Feature-engineering-y-transformadores-personalizados)  

---

## 1. Declaración de Cumplimiento de Requerimientos

| Requerimiento Solicitado | Estado | Evidencia en el Repositorio |
| :--- | :---: | :--- |
| **Carga de Dataset de Unidades Previas** | **CUMPLIDO** | Se utiliza la base SQLite `data/weather_analytics.db` generada en la unidad anterior. |
| **Ingeniería de Características ($\ge 3$ variables)** | **CUMPLIDO** | Se diseñaron **8 variables derivadas** (temporales, termo-higrométricas y de discretización física). |
| **Transformador Personalizado en `custom_transformers.py`** | **CUMPLIDO** | Clase `WeatherFeatureEngineer` (con alias `CustomTransformer`) heredando de `BaseEstimator` y `TransformerMixin`. |
| **Implementación correcta de `fit` y `transform`** | **CUMPLIDO** | `.fit()` aprende parámetros de entrenamiento; `.transform()` aplica las reglas usando `X.copy()`. |
| **Validación del Flujo de Trabajo (Train/Test Split)** | **CUMPLIDO** | Separación 80/20 con `.fit_transform()` en entrenamiento y `.transform()` en prueba (sin `.fit()`). |
| **Documentación en `README.md` y Notebook** | **CUMPLIDO** | `README.md` exhaustivo y `notebook_feature_engineering.ipynb` ejecutado con todas las salidas. |
| **Suite de Tests y Reporte de Ejecución** | **CUMPLIDO** | Carpeta `tests/` con 8 pruebas unitarias en `pytest` aprobadas al 100%. Ver [REPORTE_TESTS.md](REPORTE_TESTS.md). |

---

## 2. Detalle de Variables Derivadas Diseñadas

1. **`punto_rocio_aprox`** ($T - \frac{100 - H}{5}$): Variable física clave que determina el umbral de condensación de vapor de agua. En el modelo RandomForest obtuvo un **16.11% de importancia**, posicionándose como el predictor #1.
2. **`sensacion_termica`**: Humidex basado en presión de vapor, capturando la no-linealidad térmica.
3. **`ratio_humedad_temperatura`** ($\frac{H}{T + 0.1}$): Relación de saturación por grado centígrado.
4. **`desviacion_temp_media`** ($T - \bar{T}_{\text{train}}$): Variación respecto a la media de entrenamiento, demostrando ausencia total de fuga de datos.
5. **`hora_del_dia`** ($0 - 23$): Captura la variación solar y del ciclo circadiano diurno.
6. **`dia_de_la_semana`** ($0 - 6$): Estacionalidad semanal.
7. **`es_fin_de_semana`** ($0 / 1$): Segmentación binaria de fin de semana.
8. **`es_viento_moderado_fuerte`** ($\ge 12$ km/h): Discretización de ráfagas según la escala Beaufort.

---

## 3. Prevención de Errores Comunes

- **Prevención de Error de Copia:** El método `transform()` ejecuta `X_out = X.copy()`. Las pruebas automáticas validan con `pd.testing.assert_frame_equal` que `X_train` y `X_test` originales no son modificados.
- **Prevención de Data Leakage:** Las estadísticas globales (`mean_temp_`, `median_humidity_`) se calculan únicamente durante `fit(X_train)`. En `transform(X_test)` se aplican estos valores congelados, sin consultar estadísticas del conjunto de prueba.

---

## 4. Evidencia de Ejecución de Tests

Las pruebas automatizadas fueron ejecutadas con `pytest 9.1.1` obteniendo resultado perfecto:
- **Total de pruebas:** 8
- **Pruebas aprobadas:** 8 (100%)
- **Tiempo de ejecución:** 4.18s
- Reporte detallado disponible en: [`REPORTE_TESTS.md`](REPORTE_TESTS.md)
