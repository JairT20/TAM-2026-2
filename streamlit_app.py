import os
import runpy

target_script = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "actividades",
    "red_neuronal_mnist",
    "app_streamlit.py"
)

if __name__ == "__main__":
    runpy.run_path(target_script, run_name="__main__")
