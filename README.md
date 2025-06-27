# Evaluador de Perfil de Inversión

Este proyecto es una aplicación que evalúa el perfil de riesgo de un usuario mediante un árbol de decisión (`decision_tree_model.joblib`) y recomienda acciones basadas en clusters predefinidos obtenidos mediante K-means. Los clusters se generaron utilizando un dataset que incluye métricas como el CAGR (Compound Annual Growth Rate) y el Sharpe Index.

## Características

- **Evaluación de riesgo**: El sistema determina el nivel de riesgo del usuario (del 1 al 10) mediante un cuestionario interactivo.
- **Recomendaciones personalizadas**: Para cada nivel de riesgo, se ofrece una lista de acciones recomendadas basadas en clusters predefinidos.
- **Interfaz gráfica**: La aplicación utiliza `tkinter` para una experiencia de usuario intuitiva.

## Requisitos

- Python 3.x
- Bibliotecas necesarias:
  - `tkinter`
  - `pandas`
  - `scikit-learn` (para el modelo de árbol de decisión y K-means)
  - `joblib` (para cargar el modelo entrenado)

## Instalación

1. Clona el repositorio o descarga los archivos.
2. Instala las dependencias con el siguiente comando:
   ```bash
   pip install pandas scikit-learn joblib
