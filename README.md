# Proyecto ML ESAN - Grupo 4

## Predicción de conversión y Optimización Bayesiana de campañas digitales y WhatsApp

Proyecto final del curso **Machine Learning & Deep Learning - ESAN (2026)**. Caso académico ficticio inspirado en telecomunicaciones, desarrollado exclusivamente con datos sintéticos.

La solución tiene dos capas:

1. **Predictiva:** estima la probabilidad de conversión de una interacción cliente-campaña-canal digital.
2. **Prescriptiva:** usa Optimización Bayesiana para priorizar qué configuración conviene experimentar después con un presupuesto limitado de pruebas.

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

Benchmark prescriptivo, 20 repeticiones y 30 pruebas:

- BO score final medio: **0.334**
- Random Search score final medio: **0.264**
- Éxito para alcanzar >=95% del óptimo: **85% BO vs 10% Random Search**

Configuración candidata encontrada por BO:

- Segmento: **Jóvenes**
- Hora: **19:00**
- Canal: **Chatbot Web**
- Wording: **Digital**
- Beneficio: **Descuento 20%**
- Flujo: **Corto**
- Probabilidad estimada: **40.0%**
- Score ajustado: **0.339**

> La salida es una prioridad de experimento. No sustituye un A/B test con datos reales.

## ¿Por qué Machine Learning y no Deep Learning?

El caso utiliza datos **tabulares y estructurados**: 50,000 interacciones y 19 predictores, sin imágenes, audio, texto libre ni secuencias de alta dimensionalidad. Por ello se comparan Logistic Regression, Random Forest y XGBoost. La complejidad solo se justifica si mejora la generalización; en esta ejecución el baseline Logistic Regression fue superior en Validation.

El target `conversion_exitosa` es binario. Logistic Regression trabaja con una pérdida probabilística tipo log-loss; XGBoost usa un objetivo de clasificación binaria con `logloss`; Random Forest combina probabilidades de múltiples árboles. La selección final se basa en métricas fuera de muestra: PR-AUC, AUC, F1 y Brier Score.

## Calidad de datos, outliers y leakage

Se auditan faltantes, duplicados, rangos numéricos, distribución del target y estabilidad temporal.

No se eliminan ni winsorizan valores extremos plausibles porque las variables sintéticas tienen distribuciones y límites explícitos y la auditoría no encuentra valores imposibles fuera de esos rangos. Modificarlos sin evidencia de error alteraría las hipótesis del generador original.

`ctr`, `contencion` y `csat` se utilizan solo para análisis descriptivo porque son resultados del journey y podrían introducir leakage si se usaran como predictores.

## Partición temporal

- **Train:** meses 1 a 8
- **Validation:** meses 9 y 10
- **Test:** meses 11 y 12

El ganador y el threshold se eligen en Validation. Test se utiliza únicamente para la evaluación final.

## Estructura del repositorio

```text
Proyecto-ML-ESAN-Grupo4/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── src/
│   └── run_project.py
└── outputs/
    ├── metricas_modelos_validation.csv
    ├── metricas_ganador_test.csv
    ├── estabilidad_test_por_mes.csv
    ├── importancia_permutada.csv
    ├── benchmark_bo_vs_random.csv
    ├── recomendacion_bo.csv
    └── resumen_resultados_ppt.json
```

El repositorio contiene **código fuente ejecutable**, dependencias, instrucciones y salidas verificables, cumpliendo el requisito de reproducibilidad. El notebook final ejecutado y la presentación final se entregan además como archivos de la entrega académica.

## Cómo reproducir

Se recomienda Python 3.11+.

```bash
git clone https://github.com/bquintanillag/Proyecto-ML-ESAN-Grupo4.git
cd Proyecto-ML-ESAN-Grupo4
python -m venv .venv
```

Activar el entorno:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Instalar dependencias y ejecutar:

```bash
pip install -r requirements.txt
python src/run_project.py
```

La ejecución regenera los resultados centrales en `outputs/`.

## Reproducibilidad

- Semilla principal: `SEED = 42`.
- El dataset se genera desde cero; no requiere una fuente externa.
- Las hipótesis y coeficientes del generador sintético original se preservan.
- El preprocesamiento está encapsulado en pipelines de scikit-learn.
- Train, Validation y Test están separados temporalmente.
- `outputs/resumen_resultados_ppt.json` permite contrastar las cifras principales de la presentación.
- `src/run_project.py` fue validado contra la ejecución del notebook final: reproduce exactamente el modelo ganador, métricas de Test, matriz de confusión y resultados de BO/Random Search.

## Cobertura metodológica

El proyecto cubre problema y target de negocio, generación y comprensión de datos, auditoría de calidad y outliers, prevención de leakage, preprocesamiento, baseline y alternativas, justificación ML vs DL, funciones objetivo/pérdida, validación temporal, selección en Validation, evaluación final en Test, generalización, interpretabilidad, Proceso Gaussiano + UCB, benchmark BO vs Random Search, recomendación, riesgos y próximos pasos.

## Referencias

- Brochu, E., Cora, V. M., & de Freitas, N. (2010). *A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning*.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- ESAN (2026). *Rúbrica del trabajo final - Machine Learning & Deep Learning*.

## Transparencia sobre herramientas de apoyo

OpenAI ChatGPT (2026) se utilizó como apoyo para revisión editorial, documentación del código y verificación de consistencia entre notebook, resultados y presentación. Los resultados numéricos provienen de la ejecución del código del proyecto.
