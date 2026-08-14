# Proyecto ML ESAN - Grupo 4

## Predicción de conversión y Optimización Bayesiana de campañas digitales y WhatsApp

Proyecto final del curso **Machine Learning & Deep Learning - ESAN (2026)**. Se desarrolla un caso académico ficticio inspirado en telecomunicaciones, utilizando exclusivamente datos sintéticos.

El objetivo es construir una solución analítica de dos capas:

1. **Predictiva:** estimar la probabilidad de conversión de una interacción cliente-campaña-canal digital.
2. **Prescriptiva:** utilizar Optimización Bayesiana para priorizar qué configuración de campaña conviene experimentar a continuación con un presupuesto limitado de pruebas.

## Resultado principal

La selección del modelo se realiza únicamente con el conjunto de **Validation**, manteniendo **Test** intacto hasta la evaluación final.

- Modelo ganador: **Logistic Regression**
- Validation AUC: **0.615**
- Validation PR-AUC: **0.250**
- Threshold seleccionado en Validation: **0.16**
- Test AUC: **0.627**
- Test PR-AUC: **0.250**
- Test Recall: **0.631**
- Test F1: **0.344**
- Test Brier: **0.141**

En la capa prescriptiva, con 20 repeticiones y un presupuesto de 30 pruebas:

- Score final medio BO: **0.334**
- Score final medio Random Search: **0.264**
- Éxito para alcanzar al menos 95% del óptimo: **85% BO vs 10% Random Search**

Configuración candidata recomendada para validación experimental:

- Segmento: **Jóvenes**
- Hora: **19:00**
- Canal: **Chatbot Web**
- Wording: **Digital**
- Beneficio: **Descuento 20%**
- Flujo: **Corto**
- Probabilidad estimada de conversión: **40.0%**
- Score de negocio ajustado: **0.339**

> La recomendación es una **prioridad de experimento**, no una decisión automática de producción. Debe validarse con A/B testing y datos reales anonimizados antes de un despliegue.

## ¿Por qué Machine Learning y no Deep Learning?

Los datos son **tabulares y estructurados**: 50,000 interacciones y 19 predictores utilizados por el modelo. No hay imágenes, audio, texto libre ni secuencias de alta dimensionalidad que justifiquen una red neuronal profunda.

Por ello se comparan tres modelos apropiados para el problema:

- Logistic Regression como baseline interpretable.
- Random Forest como alternativa no lineal.
- XGBoost como alternativa de boosting.

La mayor complejidad solo se acepta si mejora la generalización. En este caso, Logistic Regression obtuvo el mejor desempeño en Validation y menor brecha Train-Validation.

El target `conversion_exitosa` es binario. Logistic Regression estima probabilidades mediante log-loss; XGBoost utiliza un objetivo probabilístico de clasificación binaria y `logloss` como métrica de evaluación; Random Forest combina probabilidades de múltiples árboles. La selección final se basa en métricas fuera de muestra: PR-AUC, AUC, F1 y Brier Score.

## Calidad de datos y valores atípicos

El notebook audita valores faltantes, duplicados, rangos numéricos, distribución del target y estabilidad temporal.

No se eliminan ni winsorizan valores extremos plausibles porque:

1. el dataset es sintético y cada variable tiene distribuciones y límites explícitos definidos en el generador;
2. la auditoría no detecta valores imposibles fuera de esos límites;
3. modificar extremos sin evidencia de error alteraría artificialmente las hipótesis del generador original.

También se evita leakage: `ctr`, `contencion` y `csat` son resultados del journey y se utilizan únicamente para análisis descriptivo, no como predictores de conversión.

## Partición temporal

Para simular un escenario de predicción futura:

- **Train:** meses 1 a 8
- **Validation:** meses 9 y 10
- **Test:** meses 11 y 12

El modelo ganador y el threshold se eligen en Validation. Test se utiliza una sola vez para la evaluación final.

## Estructura del repositorio

```text
Proyecto-ML-ESAN-Grupo4/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── Proyecto_Final_ML_Optimizacion_Bayesiana_ESAN.ipynb
├── outputs/
│   ├── metricas_modelos_validation.csv
│   ├── metricas_ganador_test.csv
│   ├── estabilidad_test_por_mes.csv
│   ├── importancia_permutada.csv
│   ├── benchmark_bo_vs_random.csv
│   ├── recomendacion_bo.csv
│   ├── resumen_resultados_ppt.json
│   └── graficos/
│       ├── comparacion_modelos_validation.png
│       ├── matriz_confusion_test.png
│       ├── importancia_permutada.png
│       └── bo_vs_random.png
└── docs/
    ├── Presentacion_Final_ESAN_15min.pptx
    └── Presentacion_Final_ESAN_15min.pdf
```

## Cómo reproducir el proyecto

### Opción recomendada: Google Colab

1. Abrir `notebooks/Proyecto_Final_ML_Optimizacion_Bayesiana_ESAN.ipynb` en Google Colab.
2. Seleccionar **Entorno de ejecución > Ejecutar todas**.
3. Esperar a que finalicen todas las celdas.
4. Verificar las métricas de Validation y Test.
5. Revisar la comparación de Optimización Bayesiana vs Random Search y la recomendación final.

El notebook contiene comentarios y explicaciones paso a paso pensados para que una persona sin experiencia avanzada en programación pueda seguir la lógica.

### Ejecución local

Se recomienda Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
jupyter notebook
```

Luego abrir el notebook y ejecutar las celdas en orden.

## Reproducibilidad

- Semilla principal: `SEED = 42`.
- El dataset se genera dentro del notebook.
- El generador sintético mantiene los supuestos originales del proyecto.
- El preprocesamiento se encapsula en pipelines de scikit-learn.
- Train, Validation y Test se separan temporalmente.
- Las cifras de la presentación se contrastan con `outputs/resumen_resultados_ppt.json`.
- Los archivos de `outputs/` se generan a partir de la ejecución del notebook.

## Contenido académico cubierto

El proyecto documenta:

- problema de negocio y target analítico;
- generación y comprensión de datos sintéticos;
- auditoría de calidad, outliers y leakage;
- EDA;
- preprocesamiento;
- baseline y modelos alternativos;
- justificación de ML frente a DL;
- funciones objetivo/pérdida y métricas;
- estrategia de validación temporal;
- selección en Validation;
- evaluación final en Test;
- análisis de generalización y calibración;
- matriz de confusión e interpretabilidad;
- Proceso Gaussiano y función UCB;
- benchmark BO vs Random Search;
- recomendación de negocio, riesgos, limitaciones y próximos pasos.

## Referencias

- Brochu, E., Cora, V. M., & de Freitas, N. (2010). *A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning*.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- ESAN (2026). *Rúbrica del trabajo final - Machine Learning & Deep Learning*.

## Transparencia sobre herramientas de apoyo

OpenAI ChatGPT (2026) se utilizó como herramienta de apoyo para **revisión editorial, documentación del código y verificación de consistencia entre notebook, resultados y presentación**. Los resultados numéricos del proyecto provienen de la ejecución del código del notebook.
