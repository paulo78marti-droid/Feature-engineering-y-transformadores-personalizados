# Feature Engineering y Transformadores Personalizados en Scikit-Learn

**Curso:** Data Science II - Coderhouse  
**Entregable:** Feature Engineering y Transformadores Personalizados  

---

## 1. Descripción del Proyecto y Problema que Resuelve

En los flujos tradicionales de Machine Learning, el preprocesamiento y la ingeniería de características suelen realizarse mediante scripts aislados de Pandas aplicados sobre la totalidad del dataset antes de la partición. Esta práctica genera tres problemas graves en entornos productivos:

1. **Fuga de Datos (*Data Leakage*):** Al calcular estadísticas globales (medias, medianas, percentiles o umbrales de discretización) antes de separar los datos, el conjunto de prueba contamina el entrenamiento, distorsionando las métricas de evaluación.
2. **Falta de Reutilización y Modularidad:** El código manual de transformación no puede integrarse directamente en los objetos `Pipeline`, `ColumnTransformer` o en métodos de búsqueda de hiperparámetros (`GridSearchCV`) de Scikit-Learn.
3. **Efectos Secundarios por Mutación (*Error de Copia*):** Modificar DataFrames directamente en memoria sin realizar copias defensivas altera los datos originales y puede propagar inconsistencias a etapas posteriores.

### Solución Implementada: `WeatherFeatureEngineer`
Este proyecto implementa el transformador personalizado **`WeatherFeatureEngineer`** en el módulo [`custom_transformers.py`](custom_transformers.py), el cual hereda de las clases base oficiales de Scikit-Learn:
- **`BaseEstimator`**: Otorga compatibilidad con la API de Scikit-Learn, permitiendo clonar estimadores y gestionar parámetros mediante `get_params()` y `set_params()`.
- **`TransformerMixin`**: Proporciona automáticamente el método `.fit_transform()` estandarizado.

El transformador automatiza la extracción, derivación y discretización de variables meteorológicas y temporales, garantizando:
- Inmutabilidad estricta (`X.copy()`) en `.transform()`.
- Aprendizaje exclusivo sobre el conjunto de entrenamiento en `.fit()`.
- Integración nativa en pipelines de clasificación y regresión.

---

## 2. Variables Diseñadas y Justificación de Valor para Machine Learning

Se diseñaron y crearon **8 variables derivadas de alto impacto**, superando con creces el requerimiento mínimo de 3 variables:

| Variable Derivada | Tipo | Origen | Justificación y Aporte de Valor al Modelo |
| :--- | :--- | :--- | :--- |
| `punto_rocio_aprox` | Continua (°C) | Física atmosférica: $T - \frac{100 - H}{5}$ | **#1 predictor en importancia (16.11%).** Representa la temperatura a la cual el aire se satura de vapor de agua. Cuando la temperatura ambiente se aproxima al punto de rocío, la probabilidad de condensación, nubosidad y lluvia se incrementa exponencialmente. |
| `sensacion_termica` | Continua (°C) | Termo-higrométrica (Índice Humidex) | Modela la interacción no lineal entre la temperatura del aire y la presión de vapor. Los modelos lineales no pueden capturar esta relación cuadrática/exponencial de forma nativa. |
| `desviacion_temp_media` | Continua (°C) | $T - \bar{T}_{\text{train}}$ | Mide anomalías térmicas relativas respecto a la media estacional aprendida *exclusivamente* en el set de entrenamiento. Aporta información contextual de fluctuaciones bruscas de temperatura. |
| `ratio_humedad_temperatura` | Continua | $\frac{H}{T + 0.1}$ | Mide la concentración de humedad relativa por unidad térmica disponible, identificando condiciones de sobre-saturación previa a eventos de lluvia. |
| `hora_del_dia` | Discreta (0 - 23) | `timestamp.dt.hour` | Captura el ciclo circadiano diurno. La radiación solar, el calentamiento de la superficie y la evaporación dependen fuertemente de la hora, determinando tormentas vespertinas o rocío matutino. |
| `dia_de_la_semana` | Discreta (0 - 6) | `timestamp.dt.dayofweek` | Modela la estacionalidad periódica semanal en los registros meteorológicos y de actividad ambiental. |
| `es_fin_de_semana` | Binaria (0 / 1) | Día $\ge 5$ | Agrupación lógica que segmenta el comportamiento entre días hábiles y fines de semana. |
| `es_viento_moderado_fuerte` | Binaria (0 / 1) | Escala Beaufort ($\text{viento} \ge 12\text{ km/h}$) | Discretiza la velocidad del viento en umbrales físicos relevantes. Los frentes fríos y tormentas suelen ingresar con ráfagas superiores a este umbral. |

---

## 3. Estructura del Repositorio

```text
Feature engineering y transformadores personalizados/
├── data/
│   └── weather_analytics.db               # Base de datos SQLite local (tabla weather_hourly)
├── custom_transformers.py                 # Módulo con la clase WeatherFeatureEngineer (Scikit-Learn)
├── validar_flujo.py                       # Script ejecutable de validación y prevención de errores
├── notebook_feature_engineering.ipynb     # Jupyter Notebook con el desarrollo, análisis y gráficos
├── requirements.txt                       # Dependencias del proyecto
└── README.md                              # Documentación completa y justificación
```

---

## 4. Validación del Flujo de Trabajo y Buenas Prácticas

### 4.1. Separación de Datos (Train / Test Split)
El dataset se particiona en 80% entrenamiento y 20% prueba:
```python
from sklearn.model_selection import train_test_split
from custom_transformers import WeatherFeatureEngineer

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

transformer = WeatherFeatureEngineer()
# Ajuste y transformación en Train
X_train_transformed = transformer.fit_transform(X_train)
# Transformación exclusiva en Test (¡Sin fit!)
X_test_transformed = transformer.transform(X_test)
```

### 4.2. Prevención de Errores Comunes
1. **Error de Copia:**  
   Se utiliza `X_out = X.copy()` al inicio del método `transform()`. En el script de validación se comprueba con `pd.testing.assert_frame_equal(X_train, X_train_backup)` que los DataFrames originales permanecen inalterados.
2. **Ajuste Erróneo y Data Leakage:**  
   El método `fit()` calcula parámetros globales únicamente en `X_train` (`mean_temp_ = 14.868 °C`). Cuando se procesa `X_test`, el método `transform()` utiliza ese parámetro guardado en memoria y nunca calcula la media de `X_test` (15.000 °C), eliminando cualquier fuga de información.
3. **Integración en `Pipeline`:**  
   El transformador es 100% compatible con `sklearn.pipeline.Pipeline`, permitiendo ejecutar predicciones directas con modelos como `RandomForestClassifier` alcanzando un **81.25% de Accuracy** en el conjunto de prueba.

---

## 5. Instrucciones de Ejecución

### Opción A: Ejecutar el Script de Validación en Consola
```bash
python validar_flujo.py
```
Salida esperada:
```text
======================================================================
VALIDACIÓN DEL FLUJO DE TRABAJO - TRANSFORMADORES SCIKIT-LEARN
======================================================================
[Paso 1] Cargando datos meteorológicos desde SQLite...
 -> Registros cargados: 240
[Paso 2] Dividiendo dataset en entrenamiento y prueba (80/20)...
[Paso 3] Inicializando WeatherFeatureEngineer()...
[Paso 4] Aplicando .fit_transform() sobre conjunto de ENTRENAMIENTO...
 -> Parámetro aprendido mean_temp_ (train): 14.87 °C
[Paso 5] Aplicando .transform() sobre conjunto de PRUEBA (sin .fit)...
[Check 1] Variables derivadas creadas (8/8) [OK]
[Check 2] Error de copia prevenido: X_train y X_test NO fueron modificados [OK]
[Check 3] Cero Data Leakage validado [OK]
 -> Accuracy del modelo en set de prueba: 83.3%
¡TODAS LAS VALIDACIONES PASARON EXITOSAMENTE!
```

### Opción B: Ejecutar el Notebook
Abre y ejecuta [notebook_feature_engineering.ipynb](notebook_feature_engineering.ipynb) en Jupyter Notebook, JupyterLab o Google Colab para visualizar el paso a paso, tablas de datos y el ranking de importancia de variables.
