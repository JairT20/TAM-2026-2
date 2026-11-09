# Teoría de Aprendizaje de Máquinas (TAM)

Repositorio de entregas, evaluaciones y proyectos prácticos de la materia Teoría de Aprendizaje de Máquinas.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tam-2026-2-d4nq6ymhmc4vrsetgeo8dn.streamlit.app/)

---

## 🌐 Demo en Vivo (Streamlit Cloud)

Podés probar el clasificador de dígitos manuscritos directamente en tu navegador o celular sin instalar nada:

👉 **[Abrir Aplicación Web en Streamlit Cloud](https://tam-2026-2-d4nq6ymhmc4vrsetgeo8dn.streamlit.app/)**

Permite subir fotos de cuadernos o tomar capturas con la cámara para clasificar dígitos en tiempo real con OpenCV y una Red Neuronal entrenada en MNIST.

---

## 📁 Estructura del Repositorio

```text
TAM/
├── actividades/
│   └── red_neuronal_mnist/        # Red neuronal entrenada con MNIST y clasificación de dígitos
│       ├── ministcode.ipynb       # Entrenamiento del modelo
│       ├── mnist_model.keras      # Pesos y arquitectura del modelo Keras
│       ├── probar_digito.py       # Aplicación interactiva de escritorio con OpenCV
│       └── app_streamlit.py       # Aplicación web interactiva para Streamlit Cloud
├── parciales/
│   └── parcial_1/                 # Entregables, análisis teóricos, figuras y visualizaciones interactivas
└── talleres/                      # Espacio para talleres prácticos del semestre
```

---

## 🛠️ Requisitos de Entorno

- **Python:** 3.10+
- **Librerías principales:**
  - `torch`
  - `keras` (v3+)
  - `opencv-python`
  - `numpy`, `matplotlib`
  - `pillow`

---

## 🚀 Ejecución Local

1. Clonar el repositorio e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Iniciar la aplicación web en Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```
3. O iniciar la aplicación de escritorio (Tkinter):
   ```bash
   python actividades/red_neuronal_mnist/probar_digito.py
   ```
