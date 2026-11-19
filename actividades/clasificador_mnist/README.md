# Clasificación de Dígitos MNIST: Red Densa (MLP) vs. Red Convolucional (CNN) con OpenCV

Actividad práctica de reconocimiento e inferencia de dígitos manuscritos utilizando visión por computadora y Deep Learning, comparando el comportamiento, la representación interna y la eficiencia entre arquitecturas densas y convolucionales.

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

## 👁️ 1. Pipeline de Preprocesamiento con OpenCV

Antes de pasar la imagen a cualquiera de las redes, la captura cruda pasa por la función `preprocess_opencv_single_digit`:

1. **Detección de iluminación y fondo:** Analiza los píxeles perimetrales para distinguir fondos oscuros de papel con sombras o cuadrículas.
2. **Binarización morfológica adaptativa:** Elimina el ruido de las hojas de cuaderno y resalta el trazo del lápiz o lapicero.
3. **Segmentación por componentes conectados:** Aísla el trazo principal del dígito y fusiona trazos desconectados pertenecientes al mismo número (como las barras transversales en un 4 o un 7).
4. **Normalización y Centrado por Momentos (Centro de Masa):** Escala el dígito conservando su relación de aspecto a una caja de $20 \times 20$ píxeles y lo posiciona exactamente en el centro de masa de un lienzo negro de $28 \times 28$, emulando el formato estándar de MNIST.

---

## 🧠 2. Arquitectura de la Red Densa (MLP)

El modelo **Perceptrón Multicapa (MLP)** procesa la imagen aplanándola en un vector unidimensional sin conservar la topología espacial bidimensional.

* **Archivo de pesos:** `mnist_model.keras` (~673 KB)
* **Tensor de entrada:** Vector aplanado `(1, 784)` con valores de píxeles en rango $[0, 255]$.

### Estructura capa por capa:
| Capa | Tipo | Configuración / Unidades | Función de Activación | Salida / Forma |
| :--- | :--- | :--- | :--- | :--- |
| **Entrada** | `Input` | 784 valores escalares | — | `(None, 784)` |
| **Oculta 1** | `Dense` | 64 neuronas | ReLU | `(None, 64)` |
| **Oculta 2** | `Dense` | 64 neuronas | ReLU | `(None, 64)` |
| **Salida** | `Dense` | 10 neuronas (dígitos 0 al 9) | Softmax | `(None, 10)` |

### Características y Limitaciones:
- **Pérdida de coherencia espacial:** Al aplanar la matriz de $28 \times 28$ a un vector de 784 elementos, la red pierde la noción de adyacencia vertical y horizontal entre píxeles.
- **Sensibilidad a traslaciones:** Si el dígito está ligeramente desplazado del centro geométrico, las neuronas densas no activan los mismos pesos, degradando la precisión.

---

## 🔬 3. Arquitectura de la Red Convolucional (CNN)

La **Red Convolucional** procesa la imagen conservando explícitamente su estructura espacial 2D, aplicando filtros deslizantes (*kernels*) que aprenden jerarquías de patrones visuales.

* **Archivo de pesos:** `mnist_model_conv.keras` (~637 KB)
* **Tensor de entrada:** Tensor tridimensional espacial `(1, 28, 28, 1)`.

### Estructura capa por capa:
| Capa | Tipo | Configuración | Activación | Dimensión de Salida |
| :--- | :--- | :--- | :--- | :--- |
| **`input`** | `InputLayer` | Entrada monocromática | — | `(None, 28, 28, 1)` |
| **`conv1`** | `Conv2D` | 32 filtros, kernel $3 \times 3$, `padding="same"` | ReLU | `(None, 28, 28, 32)` |
| **`maxP1`** | `MaxPooling2D` | Ventana $2 \times 2$, stride 2 | — | `(None, 14, 14, 32)` |
| **`conv2`** | `Conv2D` | 64 filtros, kernel $3 \times 3$, `padding="same"` | ReLU | `(None, 14, 14, 64)` |
| **`maxP2`** | `MaxPooling2D` | Ventana $2 \times 2$, stride 2 | — | `(None, 7, 7, 64)` |
| **`Flatten`** | `Flatten` | Aplanado de mapas comprimidos | — | `(None, 3136)` |
| **`Densa`** | `Dense` | 10 neuronas de clasificación | Softmax | `(None, 10)` |

### Entrenamiento:
- **Función de pérdida:** Sparse Categorical Cross-Entropy.
- **Optimizador:** Adam.
- **Épocas:** 30 épocas sobre el conjunto de entrenamiento de MNIST.

### Inspección de Filtros y Feature Maps en la App:
1. **Filtros (Kernels) de Capa 1:** 32 matrices de $3 \times 3$ aprendidas durante el entrenamiento que detectan bordes, curvas y orientaciones básicas.
2. **Feature Maps (Capa 1 - `conv1`):** 32 representaciones de $28 \times 28$ píxeles que muestran dónde y con qué intensidad reaccionó la primera capa convolucional ante el trazo del dígito.
3. **Feature Maps (Capa 2 - `conv2`):** 64 representaciones de $14 \times 14$ píxeles obtenidas tras el primer Max-Pooling, enfocadas en características de mayor nivel y combinaciones de trazos.

---

## 📊 4. Comparativa Técnica: MLP vs. CNN

| Criterio | Red Neuronal Densa (MLP) | Red Neuronal Convolucional (CNN) |
| :--- | :--- | :--- |
| **Entrada del modelo** | Vector aplanado `(1, 784)` | Tensor espacial `(1, 28, 28, 1)` |
| **Manejo de la geometría** | Ignora relaciones espaciales 2D | Preserva la localidad espacial mediante convoluciones |
| **Invariancia a traslación** | Muy baja (depende estrictamente del centrado) | Alta (gracias al submuestreo de `MaxPooling2D`) |
| **Conexiones hacia capa densa** | 64 entradas por neurona intermedia | $7 \times 7 \times 64 = 3,136$ entradas en `Flatten` |
| **Tamaño del archivo `.keras`** | ~673 KB | ~637 KB |
| **Capacidad de interpretabilidad** | Caja negra (pesos matriciales densos) | Alta: inspección directa de filtros y mapas de activación |

---

## 💻 5. Interfaz en Streamlit

La interfaz web en [app_streamlit.py](app_streamlit.py) cuenta con:
1. **Selector dinámico de modelo:** Alterna entre MLP y CNN para comparar predicciones y nivel de certeza sobre la misma imagen.
2. **Módulo interactivo de inspección CNN:** Pestañas dedicadas para visualizar los filtros aprendidos y los mapas de características resultantes en tiempo real.
3. **Distribución de probabilidades:** Gráfico de barras interactivo con los porcentajes de certeza para las 10 clases (dígitos 0 al 9).
4. **Múltiples métodos de entrada:** Admite subida de archivos de imagen (JPG, PNG, WEBP) y captura en vivo mediante la cámara web.

---

## 📁 6. Archivos de la Actividad

- `app_streamlit.py`: Aplicación web interactiva en Streamlit.
- `probar_digito.py`: Interfaz de escritorio en Tkinter.
- `ministcodes.ipynb`: Cuaderno Jupyter con el diseño, entrenamiento y extracción de activaciones de los modelos.
- `mnist_model_conv.keras`: Pesos de la Red Convolucional (CNN con MaxPooling).
- `mnist_model.keras`: Pesos de la Red Densa (MLP).

---

## 🚀 7. Ejecución Local

```bash
# 1. Instalar dependencias
pip install -r ../../requirements.txt

# 2. Ejecutar la aplicación web
streamlit run app_streamlit.py

# 3. O ejecutar la aplicación de escritorio
python probar_digito.py
```
