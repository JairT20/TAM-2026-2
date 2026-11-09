import os
import sys

# Forzar backend PyTorch para Keras 3
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import keras
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_model.keras")

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

@st.cache_resource
def load_keras_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return keras.saving.load_model(MODEL_PATH)

def preprocess_opencv_image(cv_img):
    """
    OpenCV segmentation and centering for one or multiple handwritten digits.
    Matches the exact notebook/desktop pipeline:
    1. Background detection and Otsu / adaptive background subtraction.
    2. Connected components segmentation.
    3. Left-to-right sorting.
    4. Grouping of nearby components of the same digit.
    5. Centering by Center of Mass (moments) on a 28x28 canvas.
    """
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img

    # 1. Background detection
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

    # 2. Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img)
    if num_labels <= 1:
        return [], bin_img

    total_pixels = bin_img.shape[0] * bin_img.shape[1]
    min_area = max(40, int(total_pixels * 0.0003))

    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area and h >= 12 and w >= 4:
            boxes.append([x, y, w, h, area])

    if not boxes:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        boxes = [stats[largest][:5].tolist()]

    # 3. Sort left-to-right
    boxes = sorted(boxes, key=lambda b: b[0])

    # 4. Group adjacent strokes
    merged_boxes = []
    for b in boxes:
        if not merged_boxes:
            merged_boxes.append(b)
        else:
            prev = merged_boxes[-1]
            prev_x2 = prev[0] + prev[2]
            prev_y2 = prev[1] + prev[3]
            curr_x2 = b[0] + b[2]
            curr_y2 = b[1] + b[3]

            overlap_x = min(prev_x2, curr_x2) - max(prev[0], b[0])
            gap_x = b[0] - prev_x2
            min_w = min(prev[2], b[2])

            if overlap_x > 0 or gap_x < min_w * 0.15:
                new_x = min(prev[0], b[0])
                new_y = min(prev[1], b[1])
                new_w = max(prev_x2, curr_x2) - new_x
                new_h = max(prev_y2, curr_y2) - new_y
                merged_boxes[-1] = [new_x, new_y, new_w, new_h, prev[4] + b[4]]
            else:
                merged_boxes.append(b)

    # 5. Normalize and center each digit by Center of Mass
    digit_items = []
    for x, y, w, h, _ in merged_boxes:
        digit_crop = bin_img[y:y+h, x:x+w]
        dh, dw = digit_crop.shape
        factor = 20.0 / max(dh, dw)
        new_w = max(1, int(round(dw * factor)))
        new_h = max(1, int(round(dh * factor)))
        resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        dilated = cv2.dilate(resized, np.ones((2, 2), np.uint8), iterations=1)

        canvas = np.zeros((28, 28), dtype=np.uint8)
        canvas[(28-new_h)//2:(28-new_h)//2+new_h, (28-new_w)//2:(28-new_w)//2+new_w] = dilated

        M = cv2.moments(canvas)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            shift_x = 14.0 - cx
            shift_y = 14.0 - cy
            T_mat = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            canvas = cv2.warpAffine(canvas, T_mat, (28, 28), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        input_vector = canvas.astype(np.float32).reshape(1, 784)
        digit_items.append({
            "box": (int(x), int(y), int(w), int(h)),
            "canvas": canvas,
            "vector": input_vector
        })

    return digit_items, bin_img

# UI Header
st.title("🔢 Clasificador de Dígitos MNIST")
st.markdown("Reconocimiento de dígitos manuscritos en fotos de cuaderno o cámara mediante **OpenCV** y **Red Neuronal (Keras/PyTorch)**.")

# Load model
try:
    model = load_keras_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
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

    with st.spinner("Procesando imagen con OpenCV y ejecutando inferencia..."):
        digit_items, bin_img = preprocess_opencv_image(cv_img)

    if not digit_items:
        st.warning("No se detectaron dígitos claros en la imagen. Probá con mayor contraste o mejor iluminación.")
    else:
        # Run inference for all digits
        annotated_img = cv_img.copy()
        predictions = []

        for item in digit_items:
            pred_probs = model.predict(item["vector"], verbose=0)[0]
            pred_digit = int(np.argmax(pred_probs))
            conf = float(pred_probs[pred_digit]) * 100
            item["digit"] = pred_digit
            item["conf"] = conf
            item["probs"] = pred_probs
            predictions.append((pred_digit, conf))

            # Draw green bounding box and text
            x, y, w, h = item["box"]
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(
                annotated_img,
                f"{pred_digit} ({conf:.0f}%)",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 0),
                2
            )

        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Overview metric
        seq_text = " ".join([str(p[0]) for p in predictions])
        avg_conf = np.mean([p[1] for p in predictions])

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Secuencia Detectada", seq_text)
        col_m2.metric("Confianza Promedio", f"{avg_conf:.1f}%")

        # Side by side visualizer
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Detección en Imagen")
            st.image(annotated_rgb, use_container_width=True)

        with col2:
            st.subheader("2. Visión 28x28 (Keras)")
            if len(digit_items) == 1:
                st.image(digit_items[0]["canvas"], width=200, caption=f"Dígito: {digit_items[0]['digit']}")
            else:
                sub_cols = st.columns(min(len(digit_items), 5))
                for idx, item in enumerate(digit_items):
                    with sub_cols[idx % 5]:
                        st.image(item["canvas"], width=90, caption=f"#{idx+1}: {item['digit']}")

        # Probability breakdown per digit
        st.subheader("Distribución de Probabilidades")
        for idx, item in enumerate(digit_items):
            with st.expander(f"Dígito #{idx+1} - Predicción: {item['digit']} ({item['conf']:.1f}%)", expanded=(len(digit_items) == 1)):
                df_probs = pd.DataFrame({
                    "Dígito": [str(d) for d in range(10)],
                    "Probabilidad (%)": item["probs"] * 100
                })
                st.bar_chart(df_probs.set_index("Dígito"))
else:
    st.info("Subí una foto de cuaderno o tomá una captura con tu cámara desde el panel lateral para clasificarla.")
