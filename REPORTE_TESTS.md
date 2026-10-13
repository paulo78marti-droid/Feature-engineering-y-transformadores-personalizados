# Reporte de Ejecución de Tests

**Proyecto:** Feature Engineering y Transformadores Personalizados  
**Entorno de Ejecución:** Python 3.11 / Scikit-Learn 1.9.1 / Pytest 9.1.1  
**Estado:** **8/8 TESTS PASADOS (100% SUCCESS)**  

---

## 1. Resumen de la Ejecución de Pruebas

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Jorgelina\Documents\Coder House\Data Science II\Feature engineering y transformadores personalizados
collected 8 items

tests/test_custom_transformers.py::test_inheritance_base_estimator_and_transformer_mixin PASSED [ 12%]
tests/test_custom_transformers.py::test_fit_returns_self PASSED          [ 25%]
tests/test_custom_transformers.py::test_fit_transform_creates_at_least_three_new_features PASSED [ 37%]
tests/test_custom_transformers.py::test_transform_prevents_dataframe_mutation_error_de_copia PASSED [ 50%]
tests/test_custom_transformers.py::test_no_data_leakage_between_train_and_test PASSED [ 62%]
tests/test_custom_transformers.py::test_fit_transform_consistency PASSED [ 75%]
tests/test_custom_transformers.py::test_integration_in_sklearn_pipeline PASSED [ 87%]
tests/test_custom_transformers.py::test_missing_required_column_raises_key_error PASSED [100%]

============================== 8 passed in 4.18s ==============================
```

---

## 2. Detalle de los Casos de Prueba Verificados

| Test | Objetivo | Resultado |
| :--- | :--- | :---: |
| `test_inheritance_base_estimator_and_transformer_mixin` | Valida que la clase herede formalmente de `BaseEstimator` y `TransformerMixin`. | **PASÓ** |
| `test_fit_returns_self` | Valida que `.fit()` devuelva `self` y aprenda atributos con sufijo `_`. | **PASÓ** |
| `test_fit_transform_creates_at_least_three_new_features` | Valida la creación de al menos 3 variables derivadas (se generan 8 variables). | **PASÓ** |
| `test_transform_prevents_dataframe_mutation_error_de_copia` | Valida que `transform()` use `X.copy()` para no mutar el DataFrame original. | **PASÓ** |
| `test_no_data_leakage_between_train_and_test` | Valida que el conjunto de prueba nunca sea utilizado en `.fit()` ni altere estadísticas de entrenamiento. | **PASÓ** |
| `test_fit_transform_consistency` | Valida que `.fit_transform(X)` sea idéntico a `.fit(X).transform(X)`. | **PASÓ** |
| `test_integration_in_sklearn_pipeline` | Valida integración directa en un `sklearn.pipeline.Pipeline`. | **PASÓ** |
| `test_missing_required_column_raises_key_error` | Valida control de errores defensivo ante columnas ausentes. | **PASÓ** |
