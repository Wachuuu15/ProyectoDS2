# Detección de Fracturas Cervicales en Imágenes de Tomografía Computarizada (TC) 🩻

Este proyecto tiene como objetivo desarrollar un modelo de aprendizaje automático para detectar y localizar fracturas en las vértebras cervicales utilizando imágenes de tomografía computarizada (TC). El modelo debe ser capaz de identificar fracturas en cada una de las siete vértebras cervicales (C1-C7) y proporcionar una probabilidad general de que el paciente tenga alguna fractura.

## Objetivos

- Desarrollar un modelo de machine learning que prediga la presencia de fracturas en las vértebras cervicales.
- Determinar la localización de la fractura a nivel de las vértebras C1-C7.
- Optimizar el rendimiento del modelo utilizando una métrica de **pérdida logarítmica ponderada (log loss)**.

## Estructura del Proyecto

- **Exploratory Data Analysis (EDA)**: Análisis exploratorio de los datos para comprender la distribución de las fracturas, las características de las imágenes y las relaciones entre las variables.

## Datos

Los datos de este proyecto provienen de un conjunto de imágenes TC y sus correspondientes etiquetas de fractura. El dataset incluye:

- **train.csv**: Metadatos del conjunto de entrenamiento que contienen:
  - `StudyInstanceUID`: ID del estudio de imágenes para cada paciente.
  - `patient_overall`: Indica si el paciente tiene alguna fractura.
  - `C1, C2, C3, C4, C5, C6, C7`: Etiquetas binarias para la presencia de fracturas en cada vértebra cervical.

- **test.csv**: Estructura para las predicciones de prueba con las mismas columnas que el conjunto de entrenamiento, pero sin etiquetas.

- **train_bounding_boxes.csv**: Coordenadas de las cajas delimitadoras para las fracturas en un subconjunto del conjunto de entrenamiento.

- **sample_submission.csv**: Un archivo de ejemplo que muestra el formato esperado para las predicciones.

## Evaluación

El desempeño del modelo se evalúa utilizando una métrica de **pérdida logarítmica ponderada**. La fórmula utilizada es:

$$
L_{ij} = - w_j \left( y_{ij} \log(p_{ij}) + (1 - y_{ij}) \log(1 - p_{ij}) \right)
$$

Donde:
- \( y_{ij} \) es la etiqueta verdadera (1 si hay fractura, 0 si no).
- \( p_{ij} \) es la probabilidad predicha de fractura.
- \( w_j \) es el peso asignado a cada tipo de predicción.

Los pesos son los siguientes:
- **Fractura negativa en una vértebra**: Peso de 1.
- **Fractura positiva en una vértebra**: Peso de 2.
- **Sin fractura en el paciente (global)**: Peso de 7.
- **Fractura positiva en el paciente (global)**: Peso de 14.

## Requisitos

Para ejecutar este proyecto, necesitarás instalar las siguientes librerías:

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow 

