# Clasificación de Dígitos MNIST con Redes Neuronales (Densa y CNN) y OpenCV

Actividad práctica de reconocimiento e inferencia de dígitos manuscritos utilizando visión por computadora y Deep Learning, comparando modelos densos y convolucionales.

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

La solución combina procesamiento digital de imágenes con modelos de aprendizaje profundo:

1. **Preprocesamiento con OpenCV (`preprocess_opencv_single_digit`):**
   - **Detección de iluminación:** identifica si el fondo es oscuro o papel con sombras/cuadrícula.
   - **Binarización adaptativa:** elimina el ruido de las hojas de cuaderno y resalta el trazo del lápiz/lapicero.
   - **Segmentación por componentes conectados:** aísla el dígito principal y agrupa trazos desconectados del mismo número (como las barras de un 4 o un 7).
   - **Normalización y Centrado:** redimensiona manteniendo la relación de aspecto a un cuadro de 20×20 y lo centra por **Centro de Masa (momentos estadísticos)** en un lienzo de 28×28, igual que el formato de entrenamiento de MNIST.

2. **Modelos de Red Neuronal Disponibles:**
   - **Red Convolucional (CNN - `mnist_model_conv.keras`):**
     - **Capa Conv2D 1:** 32 filtros, kernel de 3×3, activación ReLU (salida: 26×26×32).
     - **Capa Conv2D 2:** 64 filtros, kernel de 3×3, activación ReLU (salida: 24×24×64).
     - **Aplanamiento (`Flatten`):** transformación de los mapas de características a un vector denso (36,864 dimensiones).
     - **Capa Densa de Salida:** 10 neuronas con activación Softmax para clasificar los dígitos del 0 al 9.
     - **Entrenamiento:** 30 épocas sobre el conjunto de entrenamiento de MNIST.
     - **Tensor de entrada:** Formato espacial `(1, 28, 28, 1)` con píxeles en rango `[0, 255]`.
   - **Red Neuronal Densa (MLP - `mnist_model.keras`):**
     - 2 capas densas intermedias de 64 neuronas cada una con activación ReLU (`Dense(64) -> Dense(64)`).
     - Capa densa de salida de 10 neuronas con activación Softmax.
     - **Tensor de entrada:** Vector aplanado `(1, 784)` con píxeles en rango `[0, 255]`.

3. **Interfaz Interactiva (Streamlit):**
   - **Selector de arquitectura:** Permite alternar en tiempo real entre la **Red Convolucional (CNN)** y la **Red Densa (MLP)** para contrastar su comportamiento y certeza sobre la misma captura.
   - **Comparación visual:** Muestra la imagen original con el bounding box detectado junto a la imagen preprocesada de 28×28 que recibe la red.
   - **Gráfico de probabilidades:** Gráfico de barras interactivo con la distribución de probabilidades (0 al 9) calculada por la arquitectura seleccionada.

---

## 📁 Archivos de la Actividad

- `app_streamlit.py`: Código de la interfaz web en Streamlit con selector de arquitectura.
- `probar_digito.py`: Versión de escritorio con interfaz gráfica en Tkinter.
- `ministcode.ipynb`: Cuaderno Jupyter con el entrenamiento y evaluación de los modelos.
- `mnist_model_conv.keras`: Pesos de la Red Convolucional (CNN - 2 Conv [32, 64], 30 épocas).
- `mnist_model.keras`: Pesos de la Red Densa (MLP - 2 capas ocultas de 64).

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
