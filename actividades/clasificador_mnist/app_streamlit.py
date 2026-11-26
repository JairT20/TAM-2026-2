import os
import sys

# Forzar backend PyTorch para Keras 3
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DENSE_PATH = os.path.join(BASE_DIR, "mnist_model.keras")
MODEL_CONV_PATH = os.path.join(BASE_DIR, "mnist_model_conv.keras")
MODEL_COS_PATH = os.path.join(BASE_DIR, "modelo_coseno_regresion.keras")
MANUAL_PDF_PATH = os.path.join(BASE_DIR, "Manual_Maestro_Circuitos_y_Dispositivos.pdf")

SERIES_MODELS = {
    "DNN": os.path.join(BASE_DIR, "dnn_compleja.keras"),
    "CNN 1D": os.path.join(BASE_DIR, "cnn1d_compleja.keras"),
    "Simple RNN": os.path.join(BASE_DIR, "rnn_compleja.keras"),
    "LSTM": os.path.join(BASE_DIR, "lstm_compleja.keras"),
    "GRU": os.path.join(BASE_DIR, "gru_compleja.keras"),
}

st.set_page_config(
    page_title="Proyecto TAM - Deep Learning & Agente RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos visuales modernos para tarjetas, encabezados e imágenes
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 1rem 1.2rem;
        border-radius: 0.75rem;
    }
    .hero-card {
        padding: 1.25rem 1.5rem;
        border-radius: 0.85rem;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(16, 185, 129, 0.08));
        border: 1px solid rgba(59, 130, 246, 0.25);
        margin-bottom: 1.5rem;
    }
    .hero-card h1 {
        margin: 0 0 0.35rem 0;
        font-size: 1.9rem;
    }
    .hero-card p {
        margin: 0;
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_keras_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return keras.saving.load_model(model_path)

def preprocess_opencv_single_digit(cv_img):
    """
    OpenCV preprocessing for a SINGLE handwritten digit:
    1. Background detection and Otsu / adaptive background subtraction.
    2. Connected components analysis to isolate the main digit and merge its strokes.
    3. Normalization and centering by Center of Mass (moments) on a 28x28 canvas.
    """
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img

    # 1. Background detection and binarization
    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_median = float(np.median(border_pixels))
    max_val = float(np.max(gray))
    bg_is_dark = (bg_median < 50) and (max_val - bg_median > 80)

    if bg_is_dark:
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        k_size = max(25, int(min(gray.shape[:2]) * 0.1) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        bg = cv2.medianBlur(bg, k_size)

        norm = cv2.divide(gray, bg, scale=255)
        p_paper = float(np.median(norm))
        p_dark = float(np.percentile(norm, 1))
        T = p_dark + 0.35 * (p_paper - p_dark)
        bin_img = (norm < T).astype(np.uint8) * 255

    # 2. Connected components to find the single main digit
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img)
    if num_labels <= 1:
        return None, bin_img

    total_pixels = bin_img.shape[0] * bin_img.shape[1]
    min_area = max(40, int(total_pixels * 0.0003))

    valid_indices = [
        i for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]

    if not valid_indices:
        valid_indices = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]

    # Find the largest stroke of the digit
    main_idx = max(valid_indices, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    mx, my, mw, mh, _ = stats[main_idx]

    # Merge nearby disconnected strokes belonging to the same digit (e.g. crossbars in 4, 5, 7)
    x1, y1, x2, y2 = mx, my, mx + mw, my + mh
    pad = max(mw, mh) * 0.4
    for i in valid_indices:
        if i == main_idx:
            continue
        ix, iy, iw, ih, _ = stats[i]
        if not (ix > x2 + pad or ix + iw < x1 - pad or iy > y2 + pad or iy + ih < y1 - pad):
            x1 = min(x1, ix)
            y1 = min(y1, iy)
            x2 = max(x2, ix + iw)
            y2 = max(y2, iy + ih)

    w = x2 - x1
    h = y2 - y1

    # 3. Crop and resize maintaining aspect ratio into 20x20
    digit_crop = bin_img[y1:y2, x1:x2]
    dh, dw = digit_crop.shape
    if dh == 0 or dw == 0:
        return None, bin_img

    factor = 20.0 / max(dh, dw)
    new_w = max(1, int(round(dw * factor)))
    new_h = max(1, int(round(dh * factor)))
    resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    dilated = cv2.dilate(resized, np.ones((2, 2), np.uint8), iterations=1)

    # 4. Center by Center of Mass on a 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[(28 - new_h) // 2:(28 - new_h) // 2 + new_h, (28 - new_w) // 2:(28 - new_w) // 2 + new_w] = dilated

    M = cv2.moments(canvas)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        shift_x = 14.0 - cx
        shift_y = 14.0 - cy
        T_mat = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, T_mat, (28, 28), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    input_vector = canvas.astype(np.float32).reshape(1, 784)
    return {
        "box": (int(x1), int(y1), int(w), int(h)),
        "canvas": canvas,
        "vector": input_vector
    }, bin_img


# Sidebar Navigation across the 5 Project Sections
st.sidebar.title("🧭 Navegación TAM")
modulo = st.sidebar.radio(
    "Seleccionar Módulo:",
    [
        "1 & 2. Clasificador MNIST (Densa & CNN)",
        "3. Regresión No Lineal (Coseno)",
        "4. Series de Tiempo Complejas",
        "5. Agente RAG (Circuitos y Dispositivos)"
    ]
)

st.sidebar.markdown("---")


# ==============================================================================
# MÓDULO 1 & 2: CLASIFICADOR MNIST (RED DENSA Y CONVOLUCIONAL 2D)
# ==============================================================================
if modulo == "1 & 2. Clasificador MNIST (Densa & CNN)":
    st.markdown(
        """
        <div class="hero-card">
            <h1>Clasificador de Dígitos MNIST</h1>
            <p>Reconocimiento de un único dígito manuscrito mediante <b>OpenCV</b> y redes neuronales <b>MLP / CNN (Keras & PyTorch)</b>.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar: Model Architecture Selection
    st.sidebar.header("Arquitectura del Modelo")
    model_choice = st.sidebar.selectbox(
        "Seleccionar Red:",
        [
            "Convolucional (CNN - 2 Conv [32, 64], 30 épocas)",
            "Densa (MLP - Capas Ocultas [64, 64])"
        ],
        index=0
    )
    is_cnn = "CNN" in model_choice
    selected_model_path = MODEL_CONV_PATH if is_cnn else MODEL_DENSE_PATH

    # Load model
    try:
        model = load_keras_model(selected_model_path)
    except Exception as e:
        st.error(f"Error al cargar el modelo ({selected_model_path}): {e}")
        st.stop()

    # Input mode selection
    st.sidebar.header("Opciones de Entrada")
    input_mode = st.sidebar.radio("Fuente de Imagen:", ["Subir Foto de Cuaderno", "Tomar Foto con Cámara"])

    uploaded_file = None
    if input_mode == "Subir Foto de Cuaderno":
        uploaded_file = st.sidebar.file_uploader(
            "Elegí una imagen (JPG, PNG, JPEG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"]
        )
    else:
        uploaded_file = st.sidebar.camera_input("Capturá el dígito con tu cámara")

    if uploaded_file is not None:
        # Read image into OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if cv_img is None:
            st.error("No se pudo decodificar la imagen.")
            st.stop()

        with st.spinner("Procesando dígito con OpenCV y ejecutando inferencia..."):
            digit_item, bin_img = preprocess_opencv_single_digit(cv_img)

        if digit_item is None:
            st.warning("No se detectó un dígito claro en la imagen. Probá con mayor contraste o mejor iluminación.")
        else:
            canvas_float = digit_item["canvas"].astype(np.float32)
            if is_cnn:
                model_input = canvas_float.reshape(1, 28, 28, 1)
            else:
                model_input = canvas_float.reshape(1, 784)

            pred_probs = model.predict(model_input, verbose=0)[0]
            pred_digit = int(np.argmax(pred_probs))
            conf = float(pred_probs[pred_digit]) * 100

            # Annotate image
            annotated_img = cv_img.copy()
            x, y, w, h = digit_item["box"]
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 220, 0), 3)
            cv2.putText(
                annotated_img,
                f"{pred_digit} ({conf:.1f}%)",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 220, 0),
                2
            )
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            # Main result metrics row
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(
                    label="Dígito Detectado",
                    value=f"{pred_digit}"
                )
            with col_m2:
                st.metric(
                    label="Confianza de Predicción",
                    value=f"{conf:.1f}%"
                )
            with col_m3:
                st.metric(
                    label="Arquitectura Activa",
                    value="CNN 2D" if is_cnn else "MLP Densa"
                )

            st.markdown("---")

            # 3-column OpenCV pipeline visualizer
            col1, col2, col3 = st.columns([1.2, 1, 1])
            with col1:
                st.subheader("1. Detección Original")
                st.image(annotated_rgb, use_container_width=True)

            with col2:
                st.subheader("2. Máscara OpenCV")
                st.image(bin_img, use_container_width=True, caption="Binarización morfológica")

            with col3:
                st.subheader("3. Entrada 28×28")
                st.image(digit_item["canvas"], width=220, caption=f"Centrado por masa (Dígito {pred_digit})")

            # Probability distribution bar chart
            st.subheader("Distribución de Probabilidades (0 al 9)")
            df_probs = pd.DataFrame({
                "Dígito": [str(d) for d in range(10)],
                "Probabilidad (%)": pred_probs * 100
            })
            st.bar_chart(df_probs.set_index("Dígito"))

            # Inspección de Filtros y Feature Maps para la Red Convolucional
            if is_cnn:
                st.markdown("---")
                st.subheader("🔬 Inspección Convolucional: Filtros y Feature Maps")
                st.caption("Visualización de las representaciones y patrones internos aprendidos por la CNN para el dígito actual.")

                with st.expander("Ver Filtros (Kernels) y Mapas de Características", expanded=True):
                    conv_layers = [l for l in model.layers if isinstance(l, keras.layers.Conv2D) or "conv" in l.name.lower()]

                    if conv_layers:
                        tab_filt, tab_c1, tab_c2 = st.tabs([
                            "Filtros Aprendidos (Kernels C1)",
                            "Feature Maps - Capa 1 (32 mapas)",
                            "Feature Maps - Capa 2 (64 mapas)"
                        ])

                        with tab_filt:
                            st.markdown("**32 Filtros de 3×3 aprendidos por la primera capa (`conv1`):**")
                            try:
                                wc1 = conv_layers[0].get_weights()[0]
                                fig_filt, axes = plt.subplots(4, 8, figsize=(8, 4))
                                for idx, ax in enumerate(axes.flat):
                                    if idx < wc1.shape[-1]:
                                        ax.imshow(wc1[:, :, 0, idx], cmap="viridis")
                                    ax.axis("off")
                                fig_filt.tight_layout()
                                st.pyplot(fig_filt)
                                plt.close(fig_filt)
                            except Exception as e:
                                st.warning(f"No se pudieron visualizar los filtros: {e}")

                        with tab_c1:
                            try:
                                model_conv1 = keras.Model(inputs=model.inputs, outputs=conv_layers[0].output)
                                y_conv1 = model_conv1.predict(model_input, verbose=0)
                                st.markdown(f"**Activaciones generadas por `{conv_layers[0].name}`:**")
                                st.caption(f"32 mapas de características de {y_conv1.shape[1]}×{y_conv1.shape[2]} píxeles")
                                fig_c1, axes = plt.subplots(4, 8, figsize=(10, 5))
                                for idx, ax in enumerate(axes.flat):
                                    if idx < y_conv1.shape[-1]:
                                        ax.imshow(y_conv1[0, :, :, idx], cmap="gray")
                                    ax.axis("off")
                                fig_c1.tight_layout()
                                st.pyplot(fig_c1)
                                plt.close(fig_c1)
                            except Exception as e:
                                st.warning(f"No se pudieron generar los feature maps de la capa 1: {e}")

                        if len(conv_layers) > 1:
                            with tab_c2:
                                try:
                                    model_conv2 = keras.Model(inputs=model.inputs, outputs=conv_layers[1].output)
                                    y_conv2 = model_conv2.predict(model_input, verbose=0)
                                    st.markdown(f"**Activaciones generadas por `{conv_layers[1].name}`:**")
                                    st.caption(f"64 mapas de características de {y_conv2.shape[1]}×{y_conv2.shape[2]} píxeles (tras Max-Pooling)")
                                    fig_c2, axes = plt.subplots(8, 8, figsize=(10, 10))
                                    for idx, ax in enumerate(axes.flat):
                                        if idx < y_conv2.shape[-1]:
                                            ax.imshow(y_conv2[0, :, :, idx], cmap="gray")
                                        ax.axis("off")
                                    fig_c2.tight_layout()
                                    st.pyplot(fig_c2)
                                    plt.close(fig_c2)
                                except Exception as e:
                                    st.warning(f"No se pudieron generar los feature maps de la capa 2: {e}")
    else:
        st.info("Subí una foto de un único dígito en tu cuaderno o capturalo con la cámara.")


# ==============================================================================
# MÓDULO 3: REGRESIÓN NO LINEAL (APROXIMACIÓN DEL COSENO - 1000 ÉPOCAS)
# ==============================================================================
elif modulo == "3. Regresión No Lineal (Coseno)":
    st.title("📈 Regresión No Lineal: Función Coseno")
    st.markdown(r"Aproximación continua de $y = \cos(2\pi f x)$ ($f=2$) mediante red neuronal profunda (`swish`, `relu`, `tanh`) entrenada con **MSE** durante **1000 épocas**.")

    try:
        model_cos = load_keras_model(MODEL_COS_PATH)
    except Exception as e:
        st.error(f"No se encontró el modelo de regresión ({MODEL_COS_PATH}): {e}")
        st.stop()

    x_val = st.slider("Seleccioná un valor de entrada $x$ para evaluar:", min_value=-2.0, max_value=2.0, value=0.25, step=0.01)

    y_real_pt = float(np.cos(2 * np.pi * 2 * x_val))
    y_pred_pt = float(model_cos.predict(np.array([[x_val]], dtype=np.float32), verbose=0)[0][0])
    err_abs = abs(y_real_pt - y_pred_pt)

    c1, c2, c3 = st.columns(3)
    c1.metric("Valor Real cos(4πx)", f"{y_real_pt:.4f}")
    c2.metric("Predicción del Modelo", f"{y_pred_pt:.4f}")
    c3.metric("Error Absoluto (MAE)", f"{err_abs:.4f}")

    # Full curve plot
    x_grid = np.arange(-2, 2, 0.01).reshape(-1, 1).astype(np.float32)
    y_real_curve = np.cos(2 * np.pi * 2 * x_grid)
    y_pred_curve = model_cos.predict(x_grid, verbose=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_grid, y_real_curve, label="Señal Coseno Real", color="green", linewidth=2, alpha=0.7)
    ax.plot(x_grid, y_pred_curve, label="Predicción Red Neuronal", color="red", linestyle="--", linewidth=2)
    ax.scatter([x_val], [y_pred_pt], color="blue", s=80, zorder=5, label=f"Punto evaluado (x={x_val:.2f})")
    ax.set_title("Regresión Continua: Señal Real vs Predicción (1000 épocas)")
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitud")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


# ==============================================================================
# MÓDULO 4: SERIES DE TIEMPO COMPLEJAS (VENTANEO W=1000 -> P=200)
# ==============================================================================
elif modulo == "4. Series de Tiempo Complejas":
    st.title("🌊 Series de Tiempo Complejas (Tendencia + Estacionalidad + Ruido)")
    st.markdown("Comparación de **5 arquitecturas** (**DNN, CNN 1D, SimpleRNN, LSTM, GRU**) prediciendo los próximos **200 pasos** a partir de una ventana de **1000 puntos**.")

    st.sidebar.header("Configuración de Señal")
    seed = st.sidebar.number_input("Semilla aleatoria (Ruido):", min_value=0, max_value=9999, value=42, step=1)
    selected_models = st.sidebar.multiselect(
        "Modelos a comparar:",
        list(SERIES_MODELS.keys()),
        default=list(SERIES_MODELS.keys())
    )

    # Generate the complex signal (matching ministcode Section 4)
    np.random.seed(int(seed))
    t_complex = np.arange(0, 400, 0.01)
    trend = 0.005 * t_complex
    seasonality_1 = np.sin(2 * np.pi * 0.1 * t_complex)
    seasonality_2 = 0.5 * np.cos(2 * np.pi * 0.5 * t_complex)
    noise = np.random.normal(0, 0.15, size=t_complex.shape)
    y_complex = trend + seasonality_1 + seasonality_2 + noise
    y_complex = (y_complex - np.mean(y_complex)) / np.std(y_complex)

    w_comp, p_comp = 1000, 200
    max_start = len(y_complex) - w_comp - p_comp - 1

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        start_idx = st.slider("Índice de inicio de la ventana temporal:", min_value=0, max_value=max_start, value=2000, step=200)
    with col_ctrl2:
        if st.button("🎲 Ventana Aleatoria"):
            start_idx = int(np.random.randint(0, max_start))

    input_seq = y_complex[start_idx : start_idx + w_comp].astype(np.float32)
    target_seq = y_complex[start_idx + w_comp : start_idx + w_comp + p_comp].astype(np.float32)

    if not selected_models:
        st.warning("Seleccioná al menos un modelo en la barra lateral para evaluar.")
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        t_in = np.arange(0, w_comp)
        t_pred = np.arange(w_comp, w_comp + p_comp)

        ax.plot(t_in, input_seq, label="Ventana de Entrada (1000 puntos)", color="black", alpha=0.65)
        ax.plot(t_pred, target_seq, label="Señal Real Esperada (200 puntos)", color="green", linewidth=2.5, linestyle="--")

        metrics_rows = []
        for m_name in selected_models:
            m_path = SERIES_MODELS[m_name]
            try:
                m_obj = load_keras_model(m_path)
                if m_name == "DNN":
                    pred = m_obj.predict(input_seq.reshape(1, w_comp), verbose=0)[0]
                else:
                    pred = m_obj.predict(input_seq.reshape(1, w_comp, 1), verbose=0)[0]

                mse_val = float(np.mean((target_seq - pred) ** 2))
                mae_val = float(np.mean(np.abs(target_seq - pred)))
                metrics_rows.append({"Modelo": m_name, "MSE": round(mse_val, 5), "MAE": round(mae_val, 5)})
                ax.plot(t_pred, pred, label=f"{m_name} (MAE: {mae_val:.3f})", alpha=0.85)
            except Exception as e:
                st.error(f"Error cargando {m_name} ({m_path}): {e}")

        ax.set_title("Pronóstico Multi-Paso: Entrada vs Real vs Modelos")
        ax.set_xlabel("Pasos Temporales")
        ax.set_ylabel("Amplitud Normalizada")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        if metrics_rows:
            st.subheader("📊 Comparativa de Desempeño en esta Ventana")
            df_metrics = pd.DataFrame(metrics_rows).sort_values("MAE")
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)


# ==============================================================================
# MÓDULO 5: AGENTE RAG DE CIRCUITOS Y DISPOSITIVOS
# ==============================================================================
elif modulo == "5. Agente RAG (Circuitos y Dispositivos)":
    st.title("⚡ Agente RAG: Circuitos y Dispositivos Electrónicos")
    st.markdown("Asistente inteligente basado en **LangChain + Gemini + ChromaDB** entrenado exclusivamente con el **Manual Maestro de Circuitos y Dispositivos**.")

    # Resolve API Key securely from Streamlit Secrets, Env Var, or Sidebar input
    api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif os.environ.get("GEMINI_API_KEY"):
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        api_key = st.sidebar.text_input("Google AI Studio API Key:", type="password", help="Guardala en st.secrets['GEMINI_API_KEY'] para dejarla fija.")

    if not os.path.exists(MANUAL_PDF_PATH):
        st.error(f"No se encontró el archivo PDF del manual en: {MANUAL_PDF_PATH}")
        st.stop()

    if not api_key:
        st.info("Ingresá tu API Key de Google AI Studio en la barra lateral (o configurala en `st.secrets`) para activar el Agente RAG.")
        st.stop()

    @st.cache_resource
    def build_rag_chain(pdf_path, g_api_key):
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        split_docs = splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=g_api_key)
        vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=g_api_key,
            temperature=0.1
        )

        system_prompt = (
            "Eres un asistente experto exclusivamente en Circuitos Eléctricos y Dispositivos Electrónicos.\n"
            "Instrucciones obligatorias:\n"
            "1. Responde de forma técnica, precisa y didáctica utilizando ÚNICAMENTE el contexto provisto del Manual Maestro.\n"
            "2. Si la pregunta NO está relacionada con circuitos eléctricos, análisis nodal/mallas, leyes eléctricas, "
            "o dispositivos electrónicos (diodos, transistores, RC/RL/RLC, etc.), responde exactamente: "
            "'Lo siento, solo puedo responder preguntas relacionadas con Circuitos y Dispositivos Electrónicos.'\n"
            "3. Si la respuesta no se encuentra en el contexto del manual, indica que esa información no está en el documento.\n\n"
            "Contexto:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        chain = prompt | llm | StrOutputParser()
        return retriever, chain

    try:
        with st.spinner("Cargando base vectorial del Manual Maestro..."):
            retriever, rag_chain = build_rag_chain(MANUAL_PDF_PATH, api_key)
    except ImportError as ie:
        st.error(f"Faltan librerías de LangChain/Chroma/Gemini en el entorno: {ie}")
        st.code("pip install langchain langchain-community langchain-core langchain-chroma langchain-google-genai chromadb pypdf")
        st.stop()
    except Exception as e:
        st.error(f"Error inicializando el Agente RAG: {e}")
        st.stop()

    # Chat UI
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = [
            {"role": "assistant", "content": "¡Hola! Soy tu asistente técnico de Circuitos y Dispositivos. Preguntame sobre Ley de Ohm, Kirchhoff, Thévenin, circuitos RC/RL/RLC, diodos o transistores."}
        ]

    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_q := st.chat_input("Escribí tu consulta sobre circuitos o dispositivos..."):
        st.session_state.rag_messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Consultando el Manual Maestro..."):
                docs = retriever.invoke(user_q)
                context_text = "\n\n".join(d.page_content for d in docs)
                answer = rag_chain.invoke({"context": context_text, "input": user_q})
                st.markdown(answer)

                # Mostrar páginas consultadas
                pages = sorted(set(str(d.metadata.get("page", 0) + 1) for d in docs))
                if pages:
                    st.caption(f"📄 Páginas del manual consultadas: {', '.join(pages)}")

        st.session_state.rag_messages.append({"role": "assistant", "content": answer})


# Footer in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Integrantes:**")
st.sidebar.markdown("• **Jair Telpiz**  \n• **Nathalia Mora**")
st.sidebar.caption("Teoría de Aprendizaje de Máquinas (TAM)")
