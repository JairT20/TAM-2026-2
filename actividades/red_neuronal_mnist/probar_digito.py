#!/home/jair/venv_modernas/bin/python
import os
import sys

# Forzar backend PyTorch para Keras 3
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import keras
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_model.keras")

class DigitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector MNIST - Pruebas con Cuaderno")
        self.root.geometry("720x650")
        self.root.minsize(680, 600)

        # Cargar modelo
        try:
            self.model = keras.saving.load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Error al cargar modelo", f"No se pudo cargar '{MODEL_PATH}':\n{e}")
            sys.exit(1)

        self._crear_interfaz()

    def _crear_interfaz(self):
        # Header
        header = ttk.Frame(self.root, padding=12)
        header.pack(fill=tk.X)
        
        ttk.Label(
            header,
            text="Prueba de Red Neuronal MNIST con Fotos de Cuaderno",
            font=("Helvetica", 15, "bold")
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="OpenCV binariza, recorta y centra el dígito en 28x28 tal como espera la red.",
            font=("Helvetica", 10),
            foreground="#555555"
        ).pack(anchor="w", pady=(2, 0))

        # Botón selección
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill=tk.X)
        ttk.Button(
            btn_frame,
            text="📂 Seleccionar Foto de Cuaderno...",
            command=self.cargar_imagen
        ).pack(side=tk.LEFT)

        self.lbl_path = ttk.Label(btn_frame, text="Ningún archivo seleccionado", foreground="#777777")
        self.lbl_path.pack(side=tk.LEFT, padx=12)

        # Contenedor de visualización
        vis_frame = ttk.Frame(self.root, padding=10)
        vis_frame.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo: Imagen Original
        left_box = ttk.LabelFrame(vis_frame, text=" 1. Imagen Original / Recortada ", padding=10)
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.lbl_orig = ttk.Label(left_box, text="Cargá una foto\npara comenzar", anchor="center")
        self.lbl_orig.pack(fill=tk.BOTH, expand=True)

        # Panel derecho: Visión OpenCV (28x28)
        right_box = ttk.LabelFrame(vis_frame, text=" 2. Lo que 've' la Red Neuronal (28x28 OpenCV) ", padding=10)
        right_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.lbl_processed = ttk.Label(right_box, text="Acá verás el dígito procesado\n(blanco sobre negro)", anchor="center")
        self.lbl_processed.pack(fill=tk.BOTH, expand=True)

        # Panel inferior: Resultados
        res_frame = ttk.LabelFrame(self.root, text=" Resultado de la Inferencia ", padding=12)
        res_frame.pack(fill=tk.X, padx=15, pady=10)

        self.lbl_result = ttk.Label(
            res_frame,
            text="Predicción: -",
            font=("Helvetica", 18, "bold"),
            foreground="#0066cc"
        )
        self.lbl_result.pack(anchor="center")

        self.lbl_conf = ttk.Label(
            res_frame,
            text="Esperando imagen...",
            font=("Helvetica", 11)
        )
        self.lbl_conf.pack(anchor="center", pady=(4, 6))

    def preprocesar_con_opencv(self, cv_img):
        """
        Segmentación y normalización OpenCV para uno o múltiples dígitos:
        1. Detección automática de tipo de fondo (papel con sombras vs imagen invertida).
        2. Binarización limpia eliminando sombras y cuadrícula.
        3. Segmentación por componentes conectados y unión de trazos cercanos.
        4. Ordenamiento de izquierda a derecha.
        5. Centrado por centro de masa (moments) a 28x28 para cada dígito.
        """
        if len(cv_img.shape) == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_img

        # 1. Detectar tipo de fondo
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

        # 2. Segmentación de componentes conectados
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

        # 3. Ordenar cajas de izquierda a derecha (coordenada X)
        boxes = sorted(boxes, key=lambda b: b[0])

        # 4. Agrupar trazos cercanos o superpuestos del mismo número (ej: barras de un '4')
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

        # 5. Normalizar cada dígito a 28x28 centrado por Centro de Masa
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

    def cargar_imagen(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar foto de cuaderno",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos los archivos", "*.*")]
        )
        if not file_path:
            return

        self.lbl_path.config(text=os.path.basename(file_path))

        cv_img = cv2.imread(file_path)
        if cv_img is None:
            messagebox.showerror("Error", "No se pudo leer la imagen seleccionada.")
            return

        digit_items, _ = self.preprocesar_con_opencv(cv_img)

        if not digit_items:
            self.lbl_result.config(text="No se detectaron dígitos")
            self.lbl_conf.config(text="Intentá con otra imagen más nítida")
            return

        # Inferencia con Keras para cada dígito encontrado
        predicted_sequence = []
        confidences = []
        annotated_img = cv_img.copy()

        for item in digit_items:
            pred = self.model.predict(item["vector"], verbose=0)[0]
            d = int(np.argmax(pred))
            conf = float(pred[d]) * 100
            item["digit"] = d
            item["conf"] = conf
            predicted_sequence.append(str(d))
            confidences.append(f"{d} ({conf:.1f}%)")

            # Bounding box verde y etiqueta con el dígito detectado
            x, y, w, h = item["box"]
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(
                annotated_img,
                str(d),
                (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 190, 0),
                2,
                cv2.LINE_AA
            )

        full_number = "".join(predicted_sequence)
        self.lbl_result.config(text=f"Secuencia detectada:  {full_number}")
        self.lbl_conf.config(text=f"Dígitos: " + ", ".join(confidences))

        # Renderizar Imagen con Bounding Boxes en panel izquierdo
        h, w = annotated_img.shape[:2]
        max_size = 280
        scale = min(max_size / w, max_size / h)
        preview_orig = cv2.resize(annotated_img, (int(w * scale), int(h * scale)))
        preview_orig_rgb = cv2.cvtColor(preview_orig, cv2.COLOR_BGR2RGB)
        im_orig = Image.fromarray(preview_orig_rgb)
        tk_orig = ImageTk.PhotoImage(im_orig)
        self.lbl_orig.config(image=tk_orig, text="")
        self.lbl_orig.image = tk_orig

        # Renderizar tira horizontal de dígitos 28x28 en panel derecho
        canvases = [item["canvas"] for item in digit_items]
        separator = np.full((28, 3), 60, dtype=np.uint8)
        combined_strip = canvases[0]
        for c in canvases[1:]:
            combined_strip = np.hstack([combined_strip, separator, c])

        ch, cw = combined_strip.shape
        target_h = 180
        scale_proc = target_h / ch
        target_w = max(1, int(cw * scale_proc))
        preview_proc = cv2.resize(combined_strip, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        im_proc = Image.fromarray(preview_proc)
        tk_proc = ImageTk.PhotoImage(im_proc)
        self.lbl_processed.config(image=tk_proc, text="")
        self.lbl_processed.image = tk_proc


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitClassifierApp(root)
    root.mainloop()
