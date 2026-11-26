# Proyecto TAM: Deep Learning, Series de Tiempo y Agente RAG

Aplicacion web desarrollada para **Teoria de Aprendizaje de Maquinas (TAM)** que integra cinco modulos de inteligencia artificial: clasificacion de digitos manuscritos con redes densas y convolucionales, regresion no lineal, pronostico de series temporales complejas y un agente RAG especializado en circuitos y dispositivos electronicos.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jairt20-tam-2026-2-streamlit-app-82vlb5.streamlit.app)

---

## Integrantes

- **Jair Telpiz**
- **Nathalia Mora**

Materia: Teoria de Aprendizaje de Maquinas (TAM)

## Demo en Vivo

Aplicacion desplegada en Streamlit Cloud:

- [Abrir Proyecto TAM en Streamlit Cloud](https://jairt20-tam-2026-2-streamlit-app-82vlb5.streamlit.app)

---

## 1. Clasificacion MNIST: Red Densa (MLP)

### 1.a. Pipeline de Preprocesamiento con OpenCV

Antes de realizar inferencia en cualquiera de las redes de clasificacion, la imagen capturada pasa por `preprocess_opencv_single_digit`:

1. **Deteccion de iluminacion y fondo:** Analiza los pixeles perimetrales para distinguir fondos oscuros de papel con sombras o cuadriculas.
2. **Binarizacion morfologica adaptativa:** Elimina el ruido de hojas de cuaderno y resalta el trazo del digito.
3. **Segmentacion por componentes conectados:** Aisla el trazo principal y fusiona trazos cercanos pertenecientes al mismo digito (como barras transversales en un 4 o un 7).
4. **Normalizacion y centrado por momentos:** Escala el digito conservando la relacion de aspecto a 20x20 pixeles y lo centra por centro de masa en un lienzo de 28x28.

### 1.b. Arquitectura y Entrenamiento

El Perceptron Multicapa (MLP) aplana la imagen en un vector unidimensional sin conservar la topologia espacial 2D.

- **Archivo de pesos:** `mnist_model.keras`
- **Tensor de entrada:** `(1, 784)`

| Capa | Tipo | Configuracion | Activacion | Salida |
| :--- | :--- | :--- | :--- | :--- |
| Entrada | `Input` | 784 valores escalares | — | `(None, 784)` |
| Oculta 1 | `Dense` | 128 / 64 neuronas | ReLU | `(None, 64)` |
| Oculta 2 | `Dense` | 64 neuronas | ReLU | `(None, 64)` |
| Salida | `Dense` | 10 neuronas (0 al 9) | Softmax | `(None, 10)` |

---

## 2. Clasificacion MNIST: Red Convolucional (CNN 2D)

### 2.a. Arquitectura Capa por Capa

La Red Convolucional procesa la imagen conservando su estructura espacial 2D mediante filtros deslizantes (*kernels*) y submuestreo.

- **Archivo de pesos:** `mnist_model_conv.keras`
- **Tensor de entrada:** `(1, 28, 28, 1)`
- **Entrenamiento:** Optimizador Adam, perdida `sparse_categorical_crossentropy`, 30 epocas.

| Capa | Tipo | Configuracion | Activacion | Salida |
| :--- | :--- | :--- | :--- | :--- |
| `input` | `InputLayer` | Entrada monocromatica | — | `(None, 28, 28, 1)` |
| `conv1` | `Conv2D` | 32 filtros, kernel 3x3, `padding="same"` | ReLU | `(None, 28, 28, 32)` |
| `maxP1` | `MaxPooling2D` | Ventana 2x2, stride 2 | — | `(None, 14, 14, 32)` |
| `conv2` | `Conv2D` | 64 filtros, kernel 3x3, `padding="same"` | ReLU | `(None, 14, 14, 64)` |
| `maxP2` | `MaxPooling2D` | Ventana 2x2, stride 2 | — | `(None, 7, 7, 64)` |
| `Flatten` | `Flatten` | Aplanado de mapas | — | `(None, 3136)` |
| `Densa` | `Dense` | 10 neuronas de clasificacion | Softmax | `(None, 10)` |

### 2.b. Inspeccion de Filtros y Feature Maps

La aplicacion incluye pestanas dedicadas para visualizar:
1. **Filtros de Capa 1:** 32 matrices de 3x3 aprendidas para detectar bordes y curvaturas.
2. **Feature Maps (`conv1`):** 32 mapas de 28x28 pixeles mostrando la activacion inicial.
3. **Feature Maps (`conv2`):** 64 mapas de 14x14 pixeles obtenidos tras el primer Max-Pooling.

### 2.c. Comparativa Tecnica: MLP vs. CNN

| Criterio | Red Neuronal Densa (MLP) | Red Neuronal Convolucional (CNN) |
| :--- | :--- | :--- |
| Entrada del modelo | Vector aplanado `(1, 784)` | Tensor espacial `(1, 28, 28, 1)` |
| Manejo de geometria | Ignora relaciones espaciales 2D | Preserva localidad espacial mediante convoluciones |
| Invariancia a traslacion | Baja (depende del centrado) | Alta (gracias a `MaxPooling2D`) |
| Interpretabilidad | Caja negra (pesos matriciales) | Alta (inspeccion de filtros y feature maps) |

---

## 3. Regresion No Lineal: Funcion Coseno

### 3.a. Arquitectura y Entrenamiento

Aproximacion continua de la funcion $y = \cos(2\pi f x)$ ($f = 2$) mediante una red neuronal profunda con activaciones mixtas (`swish`, `relu`, `tanh`) y salida lineal.

- **Archivo de pesos:** `modelo_coseno_regresion.keras`
- **Entrenamiento:** 1000 epocas, optimizador Adam (`lr=0.01`), funcion de perdida `mse` y metrica `mae`.

### 3.b. Evaluacion en Streamlit

Permite evaluar cualquier valor $x \in [-2, 2]$ mediante un control deslizante interactivo, mostrando el valor real, el valor predicho, el error absoluto y la grafica completa de ajuste.

---

## 4. Series de Tiempo Complejas

### 4.a. Senal Sintetica y Ventaneo Temporal

Se genera una serie temporal con tendencia lineal, dos componentes estacionales y ruido gaussiano normalizado. Mediante ventaneo deslizante ($w = 1000$ pasos de entrada), los modelos predicen los siguientes $p = 200$ pasos a futuro.

### 4.b. Modelos Comparados

- **DNN** (`dnn_compleja.keras`): Red densa `128 -> 64 -> 200`.
- **CNN 1D** (`cnn1d_compleja.keras`): Convoluciones temporales 1D (`Conv1D` + `MaxPooling1D`).
- **Simple RNN** (`rnn_compleja.keras`): Capa recurrente simple de 64 unidades.
- **LSTM** (`lstm_compleja.keras`): Red de memoria a corto y largo plazo (64 unidades).
- **GRU** (`gru_compleja.keras`): Unidad recurrente cerrada (64 unidades).

En Streamlit se pueden variar la semilla de ruido y la ventana temporal para comparar en vivo las curvas de prediccion y la tabla de errores **MSE** y **MAE**.

---

## 5. Agente RAG: Circuitos y Dispositivos Electronicos

### 5.a. Recuperacion Vectorial y Restriccion de Dominio

Asistente conversacional construido con **LangChain**, **ChromaDB** y **Google Gemini (`gemini-3.5-flash-lite`)** sobre el documento `Manual_Maestro_Circuitos_y_Dispositivos.pdf`.

- Utiliza embeddings `models/gemini-embedding-001` y un *System Prompt* estricto para responder unicamente sobre leyes electricas, analisis nodal/mallas, Thevenin/Norton, circuitos RC/RL/RLC, diodos y transistores, citando las paginas consultadas y rechazando consultas externas.

### 5.b. Configuracion de API Key

La aplicacion obtiene `GEMINI_API_KEY` automaticamente desde `st.secrets`, variables de entorno o mediante campo seguro en la barra lateral.

---

## Archivos del Proyecto

- `app_streamlit.py`: Aplicacion web en Streamlit con los 5 modulos.
- `ministcode.ipynb`: Cuaderno Jupyter organizado en 5 secciones para entrenamiento en Colab.
- `Manual_Maestro_Circuitos_y_Dispositivos.pdf`: Documento base del Agente RAG.
- `mnist_model.keras` y `mnist_model_conv.keras`: Modelos de clasificacion MNIST.
- `modelo_coseno_regresion.keras`: Modelo de regresion no lineal (1000 epocas).
- `dnn_compleja.keras`, `cnn1d_compleja.keras`, `rnn_compleja.keras`, `lstm_compleja.keras`, `gru_compleja.keras`: Modelos de series de tiempo.
- `probar_digito.py`: Aplicacion de escritorio en Tkinter para MNIST.

## Ejecucion Local

```bash
# 1. Instalar dependencias
pip install -r ../../requirements.txt

# 2. Ejecutar la aplicacion web
streamlit run app_streamlit.py
```
