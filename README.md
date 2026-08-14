# Proyecto ML ESAN — Grupo 4

## Predicción de conversión y Optimización Bayesiana de campañas digitales y WhatsApp

Proyecto final del curso **Machine Learning & Deep Learning — ESAN (2026)**. El caso es académico y ficticio, inspirado en un contexto de telecomunicaciones, y utiliza exclusivamente **datos sintéticos reproducibles**.

La solución combina dos capas:

1. **Predictiva:** estima la probabilidad de conversión de una interacción cliente–campaña–canal digital.
2. **Prescriptiva:** utiliza Optimización Bayesiana para priorizar qué configuración conviene experimentar después con un presupuesto limitado de pruebas.

## Archivo principal

El entregable técnico principal del repositorio es:

`Proyecto_Final_ML_Optimizacion_Bayesiana_ESAN_FINAL.ipynb`

El notebook contiene el flujo completo y ejecutado del proyecto: generación del dataset sintético, auditoría de calidad, EDA, preparación, partición temporal, baseline, modelos alternativos, selección en Validation, evaluación final en Test, interpretabilidad, Optimización Bayesiana, comparación contra Random Search y recomendación final.

## Resultados reproducibles

La selección del modelo se realiza únicamente con **Validation**; **Test** queda reservado para la evaluación final.

- Modelo ganador: **Logistic Regression**
- Validation AUC: **0.615**
- Validation PR-AUC: **0.250**
- Threshold seleccionado en Validation: **0.16**
- Test AUC: **0.627**
- Test PR-AUC: **0.250**
- Test Recall: **0.631**
- Test F1: **0.344**
- Test Brier: **0.141**

Benchmark prescriptivo, con 20 repeticiones y 30 pruebas por método:

- BO score final medio: **0.334**
- Random Search score final medio: **0.264**
- Tasa de éxito para alcanzar al menos 95% del óptimo: **85% BO vs 10% Random Search**

Configuración candidata encontrada por BO:

- Segmento: **Jóvenes**
- Hora: **19:00**
- Canal: **Chatbot Web**
- Wording: **Digital**
- Beneficio: **Descuento 20%**
- Flujo: **Corto**
- Probabilidad estimada: **40.0%**
- Score ajustado: **0.339**

> La recomendación de BO prioriza una prueba; no sustituye una validación mediante A/B testing con datos reales.

## ¿Por qué Machine Learning y no Deep Learning?

El caso utiliza datos **tabulares y estructurados**: 50,000 interacciones y 19 predictores, sin imágenes, audio, texto libre ni secuencias de alta dimensionalidad. Por ello se comparan Logistic Regression, Random Forest y XGBoost. La complejidad adicional solo se justifica si mejora la generalización; en esta ejecución el baseline Logistic Regression obtuvo el mejor resultado en Validation.

El target `conversion_exitosa` es binario. Logistic Regression trabaja con una pérdida probabilística tipo log-loss; XGBoost usa un objetivo de clasificación binaria y Random Forest combina probabilidades de múltiples árboles. La selección final se basa en métricas fuera de muestra: PR-AUC, AUC, F1 y Brier Score.

## Calidad de datos, outliers y leakage

Se auditan faltantes, duplicados, rangos numéricos, distribución del target y estabilidad temporal.

No se eliminan ni winsorizan valores extremos plausibles porque las variables sintéticas tienen distribuciones y límites explícitos y la auditoría no encuentra valores imposibles fuera de dichos rangos. Modificarlos sin evidencia de error alteraría las hipótesis del generador sintético original.

`ctr`, `contencion` y `csat` se utilizan únicamente para análisis descriptivo porque son resultados del journey y podrían introducir leakage si se utilizaran como predictores de conversión.

## Partición temporal

- **Train:** meses 1 a 8
- **Validation:** meses 9 y 10
- **Test:** meses 11 y 12

El ganador y el threshold se eligen en Validation. Test se utiliza únicamente para la evaluación final.

## Estructura del repositorio

```text
Proyecto-ML-ESAN-Grupo4/
├── Proyecto_Final_ML_Optimizacion_Bayesiana_ESAN_FINAL.ipynb
├── Presentacion_Final_ESAN_15min.pdf
├── README.md
├── requirements.txt
└── outputs/
    ├── metricas_modelos_validation.csv
    ├── metricas_ganador_test.csv
    ├── estabilidad_test_por_mes.csv
    ├── importancia_permutada.csv
    ├── benchmark_bo_vs_random.csv
    ├── recomendacion_bo.csv
    └── resumen_resultados_ppt.json
```

## Cómo reproducir

1. Clonar o descargar el repositorio.
2. Crear un entorno de Python (recomendado Python 3.13 o Google Colab).
3. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

4. Abrir `Proyecto_Final_ML_Optimizacion_Bayesiana_ESAN_FINAL.ipynb` en Jupyter o Google Colab.
5. Ejecutar las celdas en orden mediante **Run all / Ejecutar todas**.
6. Contrastar las salidas con los archivos de `outputs/`.

## Reproducibilidad

- Semilla principal: `SEED = 42`.
- El dataset se genera dentro del notebook; no requiere una fuente de datos externa.
- Las hipótesis, coeficientes, distribuciones e interacciones del generador sintético original se preservan.
- El preprocesamiento está encapsulado en pipelines de scikit-learn.
- Train, Validation y Test están separados temporalmente.
- Los archivos de `outputs/` permiten verificar las cifras principales reportadas en la presentación.

## Referencias

- Brochu, E., Cora, V. M., & de Freitas, N. (2010). *A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning*.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- ESAN (2026). *Rúbrica del trabajo final — Machine Learning & Deep Learning*.

## Transparencia sobre herramientas de apoyo

OpenAI ChatGPT (2026) se utilizó como apoyo para revisión editorial, documentación del código y verificación de consistencia entre notebook, resultados y presentación. **Los resultados numéricos reportados provienen de la ejecución del código del proyecto.**