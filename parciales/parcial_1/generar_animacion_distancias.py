import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

fig, ax = plt.subplots(figsize=(11, 5.5), dpi=100)

# Energías de juguete para 3 fotos
e = np.array([[10], [20], [30]])  # norm_comb (3x1)
ones = np.ones((3, 1))            # On_comb (3x1)

# Matrices intermedias
mat_A = e @ ones.T                # (3x3) proyecta horizontal
mat_B = ones @ e.T                # (3x3) proyecta vertical
mat_sum = mat_A + mat_B           # ||x_i||^2 + ||x_j||^2

# Simulación de producto cruzado 2 * X @ X.T
dot_prod = np.array([
    [10.0, 12.0, 14.0],
    [12.0, 20.0, 21.0],
    [14.0, 21.0, 30.0]
])
sq_dists = mat_sum - 2 * dot_prod

def draw_frame(frame):
    ax.clear()
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.0, 6.0)
    ax.axis('off')
    
    # 4 fases de animación
    # 0-14: Término 1 (norm_comb @ On.T)
    # 15-29: Término 2 (+ On @ norm_comb.T)
    # 30-44: Término 3 (- 2 * X @ X.T)
    # 45-59: Resultado final con np.maximum (||x_i - x_j||^2)
    phase = (frame // 15) % 4
    
    ax.text(6.0, 5.4, r"$\mathbf{sq\_dists = (norm \ @ \ On^\top) + (On \ @ \ norm^\top) - 2(X @ X^\top)}$", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#1A365D')
    
    # Dibujar bloque 1 (A): norm_comb @ On.T
    ax.text(2.0, 4.4, r"$\mathbf{1.\ (norm \ @ \ On^\top)}$", ha='center', fontsize=10, fontweight='bold', 
            color='#2B6CB0' if phase == 0 else '#4A5568')
    for i in range(3):
        for j in range(3):
            val = mat_A[i, j]
            bg = '#BEE3F8' if phase == 0 else '#EDF2F7'
            edge = '#3182CE' if phase == 0 else '#CBD5E0'
            rect = patches.Rectangle((j*0.9 + 0.7, (2-i)*0.9 + 1.2), 0.85, 0.85, 
                                     facecolor=bg, edgecolor=edge, lw=1.5 if phase == 0 else 1.0)
            ax.add_patch(rect)
            ax.text(j*0.9 + 1.12, (2-i)*0.9 + 1.62, f"{int(val)}", ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Signo +
    ax.text(3.9, 2.5, "+", ha='center', va='center', fontsize=18, fontweight='bold', color='#4A5568')
    
    # Dibujar bloque 2 (B): On @ norm_comb.T
    ax.text(5.5, 4.4, r"$\mathbf{2.\ (On \ @ \ norm^\top)}$", ha='center', fontsize=10, fontweight='bold', 
            color='#2F855A' if phase == 1 else '#4A5568')
    for i in range(3):
        for j in range(3):
            val = mat_B[i, j]
            bg = '#C6F6D5' if phase == 1 else '#EDF2F7'
            edge = '#38A169' if phase == 1 else '#CBD5E0'
            rect = patches.Rectangle((j*0.9 + 4.2, (2-i)*0.9 + 1.2), 0.85, 0.85, 
                                     facecolor=bg, edgecolor=edge, lw=1.5 if phase == 1 else 1.0)
            ax.add_patch(rect)
            ax.text(j*0.9 + 4.62, (2-i)*0.9 + 1.62, f"{int(val)}", ha='center', va='center', fontsize=9, fontweight='bold')
            
    # Signo - 2(X @ X.T)
    ax.text(7.4, 2.5, "-", ha='center', va='center', fontsize=18, fontweight='bold', color='#4A5568')
    ax.text(8.0, 2.5, r"$2(X X^\top)$", ha='center', va='center', fontsize=11, fontweight='bold', color='#D69E2E' if phase == 2 else '#718096')

    # Signo =
    ax.text(9.2, 2.5, "=", ha='center', va='center', fontsize=20, fontweight='bold', color='#4A5568')

    # Dibujar Matriz Resultado Final sq_dists
    ax.text(10.7, 4.4, r"$\mathbf{sq\_dists\ (||x_i - x_j||^2)}$", ha='center', fontsize=10, fontweight='bold', 
            color='#9B2C2C' if phase == 3 else '#4A5568')
    for i in range(3):
        for j in range(3):
            is_diag = (i == j)
            val = sq_dists[i, j]
            if phase < 3:
                txt = "?"
                bg = '#F7FAFC'
                edge = '#CBD5E0'
            else:
                txt = f"{val:.1f}" if not is_diag else "0.0"
                bg = '#FED7D7' if not is_diag else '#FEFCBF'
                edge = '#E53E3E' if not is_diag else '#D69E2E'
            
            rect = patches.Rectangle((j*0.9 + 9.4, (2-i)*0.9 + 1.2), 0.85, 0.85, 
                                     facecolor=bg, edgecolor=edge, lw=1.5 if phase == 3 else 1.0)
            ax.add_patch(rect)
            ax.text(j*0.9 + 9.82, (2-i)*0.9 + 1.62, txt, ha='center', va='center', fontsize=8.5, fontweight='bold')

    # Cuadro explicativo inferior
    box_rect = patches.FancyBboxPatch((0.5, -0.6), 11.0, 1.1, boxstyle="round,pad=0.2", 
                                      facecolor='#EDF2F7', edgecolor='#CBD5E0', linewidth=1)
    ax.add_patch(box_rect)
    
    if phase == 0:
        msg1 = "Paso 1: norm_comb @ On_comb.T (Columna de energías × Fila de unos)"
        msg2 = "Copia la energía ||x_i||² de cada foto horizontalmente en todas las columnas."
    elif phase == 1:
        msg1 = "Paso 2: On_comb @ norm_comb.T (Columna de unos × Fila de energías)"
        msg2 = "Copia la energía ||x_j||² de cada foto verticalmente en todas las filas."
    elif phase == 2:
        msg1 = "Paso 3: Resta 2 * (X @ X.T)"
        msg2 = "Resta el doble producto interno de píxeles: -2 * <x_i, x_j> para completar (a - b)²."
    else:
        msg1 = "Paso 4: ¡Matriz completa de distancias cuadráticas ||x_i - x_j||²!"
        msg2 = "np.maximum(..., 0.0) asegura que la diagonal principal sea exactamente 0.0 sin NaN."
        
    ax.text(6.0, 0.05, msg1, ha='center', va='center', fontsize=11, fontweight='bold', color='#1A202C')
    ax.text(6.0, -0.35, msg2, ha='center', va='center', fontsize=9.5, color='#4A5568')

anim = FuncAnimation(fig, draw_frame, frames=60, interval=200)
anim.save('/home/jair/Documents/U/TAM/animacion_distancias.gif', writer=PillowWriter(fps=5))
print("Animacion guardada en /home/jair/Documents/U/TAM/animacion_distancias.gif")
