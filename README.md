# Predicción de Fuga de Clientes (Churn) en Telecomunicaciones 
Este proyecto fue desarrollado como el desafío final de clasificación para el Diplomado en Data Science, Machine Learning, Inteligencia Artificial, Deep Learning Versión N°23 de la Pontificia Universidad Católica de Valparaíso.

## Objetivos del Proyecto
* Implementar un modelo de clasificación capaz de predecir la fuga de clientes (*churn*)
* Medir la capacidad predictiva mediante métricas de validación técnica y de negocio
* Identificar patrones de comportamiento para gestionar mecanismos de retención y fidelización

## Metodología de Trabajo
* **Análisis Exploratorio (EDA):** Visualización de la distribución de las variables y su impacto en la fuga
* **Evaluación de Datos:** Revisión de valores faltantes y tratamiento de datos atípicos (*outliers*) detectados en variables como minutos diarios y llamadas a soporte
* **Ingeniería de Características:** Selección estratégica de variables, eliminando redundancias correlacionadas (como cargos vs. minutos)
* **Modelamiento:** Comparativa entre **Árboles de Decisión** (con poda/pruning para evitar sobreajuste) y **Random Forest**

## Conclusión
El análisis permitió determinar que los clientes con planes internacionales y aquellos con alta frecuencia de llamadas a soporte técnico tienen una propensión significativamente mayor a la fuga. El modelo de Random Forest resultó ser la herramienta más robusta para orientar las estrategias de fidelización de la empresa.
