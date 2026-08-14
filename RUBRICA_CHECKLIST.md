# Checklist de cumplimiento de la rúbrica ESAN

## 1. Problema, objetivos y valor para el negocio
- [x] Contexto y decisión de negocio definidos.
- [x] Target analítico: `conversion_exitosa`.
- [x] Capa predictiva y capa prescriptiva claramente diferenciadas.
- [x] Criterios de éxito técnicos y de negocio.

## 2. Datos, comprensión y preparación
- [x] Unidad de análisis y variables documentadas.
- [x] Dataset sintético reproducible con `SEED = 42`.
- [x] Auditoría de faltantes y duplicados.
- [x] Revisión de rangos y valores atípicos.
- [x] Justificación explícita para conservar extremos plausibles sin alterar el generador.
- [x] Prevención de leakage: CTR, contención y CSAT no se usan como predictores.
- [x] Split temporal Train / Validation / Test.

## 3. Metodología y selección de modelos
- [x] Baseline: Logistic Regression.
- [x] Alternativas: Random Forest y XGBoost.
- [x] Justificación explícita de ML frente a Deep Learning.
- [x] Funciones objetivo/pérdida explicadas.
- [x] Hiperparámetros explícitos y semillas fijas.
- [x] Selección del ganador únicamente con Validation.

## 4. Implementación y reproducibilidad
- [x] `requirements.txt` con versiones.
- [x] Código fuente ejecutable en `src/run_project.py`.
- [x] Instrucciones completas en `README.md`.
- [x] Datos sintéticos generados desde código; no dependen de archivos externos.
- [x] Salidas de validación disponibles en `outputs/`.

## 5. Evaluación, comparación e interpretación técnica
- [x] AUC y PR-AUC.
- [x] Precision, Recall y F1 con threshold optimizado en Validation.
- [x] Brier Score para calidad probabilística.
- [x] Matriz de confusión del ganador en Test.
- [x] Comparación Train vs Validation para revisar generalización.
- [x] Estabilidad temporal en los meses de Test.
- [x] Importancia permutada.

## 6. Impacto, recomendaciones, riesgos y limitaciones
- [x] Recomendación accionable de configuración de campaña.
- [x] BO comparada con Random Search bajo el mismo presupuesto.
- [x] La salida se plantea como prioridad de experimento, no decisión automática.
- [x] Se reconocen limitaciones de datos sintéticos, drift, sesgo y sobrecontacto.
- [x] Validación futura mediante A/B testing y data real anonimizada.

## 7. Presentación y sustentación
- [x] PPT alineado con los resultados reproducibles.
- [x] Notas de exposición distribuidas entre cinco integrantes.
- [x] Tiempo total planificado: 15 minutos.
- [x] Referencias y uso de herramientas de apoyo reconocidos.
- [x] Presentación final preparada también en PDF para la entrega académica.

> Este checklist es una guía de trazabilidad. La calificación final depende de la evaluación y sustentación del equipo.
