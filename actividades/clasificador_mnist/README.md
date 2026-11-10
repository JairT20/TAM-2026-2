# Clasificación de Dígitos MNIST con Red Neuronal y OpenCV

Actividad práctica de reconocimiento e inferencia de dígitos manuscritos utilizando visión por computadora y Deep Learning.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jairt20-tam-2026-2-streamlit-app-82vlb5.streamlit.app)

---

## 👥 Integrantes

- **Jair Telpiz**
- **Nathalia Mora**

*Materia: Teoría de Aprendizaje de Máquinas (TAM)*

---

## 🌐 Demo en Vivo (Web)

Podés probar la aplicación desplegada en la nube sin instalar nada:

👉 **[Abrir Clasificador MNIST en Streamlit Cloud](https://jairt20-tam-2026-2-streamlit-app-82vlb5.streamlit.app)**

Permite subir fotos de cuadernos o capturar trazos manuscritos en vivo usando la cámara de tu celular o laptop.

---

## 🧠 Arquitectura y Funcionamiento

La solución combina procesamiento digital de imágenes con una red neuronal entrenada:

1. **Preprocesamiento con OpenCV (`preprocess_opencv_single_digit`):**
   - **Detección de iluminación:** identifica si el fondo es oscuro o papel con sombras/cuadrícula.
   - **Binarización adaptativa:** elimina el ruido de las hojas de cuaderno y resalta el trazo del lápiz/lapicero.
   - **Segmentación por componentes conectados:** aísla el dígito principal y agrupa trazos desconectados del mismo número (como las barras de un 4 o un 7).
   - **Normalización y Centrado:** redimensiona manteniendo la relación de aspecto a un cuadro de 20×20 y lo centra por **Centro de Masa (momentos estadísticos)** en un lienzo de 28×28, igual que el formato de entrenamiento de MNIST.

2. **Modelo de Red Neuronal (`mnist_model.keras`):**
   - Inferencia con backend PyTorch sobre la entrada normalizada de 784 píxeles (28×28).
   - Salida Softmax con distribución de probabilidad para cada dígito del 0 al 9.

3. **Interfaz Interactiva (Streamlit):**
   - Comparación visual: Imagen original con bounding box verde vs. imagen procesada de 28×28 que recibe la red.
   - Gráfico de barras interactivo con los porcentajes de certeza para las 10 clases.

---

## 📁 Archivos de la Actividad

- `app_streamlit.py`: Código de la interfaz web en Streamlit.
- `probar_digito.py`: Versión de escritorio con interfaz gráfica en Tkinter.
- `ministcode.ipynb`: Cuaderno Jupyter con el entrenamiento y evaluación de la red.
- `mnist_model.keras`: Pesos y arquitectura del modelo entrenado.

---

## 🚀 Ejecución Local

Para correr esta actividad en tu entorno local:

```bash
# 1. Instalar dependencias
pip install -r ../../requirements.txt

# 2. Ejecutar la aplicación web
streamlit run app_streamlit.py

# 3. O ejecutar la aplicación de escritorio
python probar_digito.py
```
