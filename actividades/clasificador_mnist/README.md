# Proyecto TAM: Deep Learning, Series de Tiempo y Agente RAG

Aplicación integral de **Teoría de Aprendizaje de Máquinas (TAM)** que reúne 5 módulos interactivos en Streamlit: clasificación de dígitos manuscritos con redes densas y convolucionales, regresión no lineal, pronóstico de series temporales complejas y un Agente RAG especializado en circuitos y dispositivos electrónicos.

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

## 📈 5. Regresión No Lineal: Aproximación de Función Coseno

Aproximación continua de la señal $y = \cos(2\pi f x)$ ($f=2$) mediante una red neuronal profunda con activaciones mixtas (`swish`, `relu`, `tanh`) y capa de salida lineal.

* **Archivo de pesos:** `modelo_coseno_regresion.keras` (~109 KB)
* **Entrenamiento:** 1000 épocas, optimizador Adam (`lr=0.01`), función de pérdida `mse` y métrica `mae`.
* **En Streamlit:** Slider interactivo para evaluar cualquier valor $x \in [-2, 2]$, comparando el valor real contra la predicción puntual y la curva completa.

---

## 🌊 6. Series de Tiempo Complejas (Tendencia + Estacionalidad + Ruido)

Pronóstico multi-paso ($p = 200$ puntos a futuro) a partir de una ventana de $w = 1000$ puntos pasados sobre una señal sintética realista con tendencia lineal, dos componentes estacionales y ruido gaussiano.

### Modelos Comparados:
* **DNN** (`dnn_compleja.keras`): Red densa de 128 → 64 → 200 neuronas.
* **CNN 1D** (`cnn1d_compleja.keras`): Convoluciones temporales 1D (`Conv1D` + `MaxPooling1D`).
* **Simple RNN** (`rnn_compleja.keras`): Capa recurrente simple de 64 unidades.
* **LSTM** (`lstm_compleja.keras`): Red de memoria a corto y largo plazo (64 unidades).
* **GRU** (`gru_compleja.keras`): Unidad recurrente cerrada (64 unidades).

En Streamlit se puede variar la semilla del ruido, elegir ventanas aleatorias y visualizar en tiempo real la tabla comparativa de **MSE** y **MAE** entre todos los modelos.

---

## ⚡ 7. Agente RAG: Circuitos y Dispositivos Electrónicos

Asistente conversacional especializado construido con **LangChain**, **ChromaDB** y **Google Gemini (`gemini-3.5-flash-lite`)** sobre el documento `Manual_Maestro_Circuitos_y_Dispositivos.pdf`.

* **Restricción estricta de dominio:** Utiliza recuperación vectorial (*embeddings* `models/gemini-embedding-001`) y un *System Prompt* blindado para responder **únicamente** sobre leyes eléctricas, análisis nodal/mallas, Thévenin/Norton, circuitos RC/RL/RLC, diodos y transistores, citando las páginas consultadas y rechazando temas externos.
* **Configuración de API Key:** Lee `GEMINI_API_KEY` automáticamente desde `st.secrets`, variables de entorno o mediante campo seguro en la barra lateral.

---

## 💻 8. Interfaz en Streamlit

La interfaz web en [app_streamlit.py](app_streamlit.py) organiza los 5 módulos desde un menú de navegación lateral:
1. **Clasificador MNIST (Densa & CNN):** Selector de modelo, carga/cámara con OpenCV, barras de probabilidad, filtros aprendidos y *feature maps*.
2. **Regresión No Lineal (Coseno):** Evaluación interactiva y gráfica real vs. predicha.
3. **Series de Tiempo Complejas:** Selector de ventana aleatoria, filtros de arquitectura y tabla comparativa de error.
4. **Agente RAG (Circuitos y Dispositivos):** Chat interactivo con citas por página del manual.

---

## 📁 9. Archivos de la Actividad

- `app_streamlit.py`: Aplicación web principal en Streamlit con los 5 módulos.
- `ministcode.ipynb`: Cuaderno Jupyter organizado en 5 secciones para el entrenamiento y prueba en Google Colab.
- `Manual_Maestro_Circuitos_y_Dispositivos.pdf`: Documento técnico base para el Agente RAG.
- `mnist_model.keras` y `mnist_model_conv.keras`: Pesos de clasificación MNIST (Densa y Convolucional).
- `modelo_coseno_regresion.keras`: Pesos de la regresión no lineal (1000 épocas).
- `dnn_compleja.keras`, `cnn1d_compleja.keras`, `rnn_compleja.keras`, `lstm_compleja.keras`, `gru_compleja.keras`: Pesos de los modelos de series de tiempo.
- `probar_digito.py`: Interfaz de escritorio en Tkinter para MNIST.

---

## 🚀 10. Ejecución Local y Despliegue

```bash
# 1. Instalar dependencias (incluyendo LangChain, Chroma y Google GenAI)
pip install streamlit keras torch opencv-python-headless numpy pandas pillow matplotlib langchain langchain-community langchain-core langchain-chroma langchain-google-genai chromadb pypdf

# 2. Configurar tu clave de Google AI Studio (opcional por entorno o en Streamlit Secrets)
export GEMINI_API_KEY="tu_api_key"

# 3. Ejecutar la aplicación web
streamlit run app_streamlit.py
```
