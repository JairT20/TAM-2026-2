import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

# Datos de juguete para ilustrar
np.random.seed(42)
X_sq = np.array([
    [0.4, 0.9, 0.1, 0.6],
    [0.8, 0.2, 0.5, 0.3],
    [0.1, 0.7, 0.8, 0.2]
])
n_rows, n_cols = X_sq.shape
ones = np.ones((n_cols, 1))
norms = X_sq @ ones

def draw_frame(frame):
    ax.clear()
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 6)
    ax.axis('off')
    
    # Determinar qué fila está activa
    active_row = (frame // 15) % n_rows
    sub_step = frame % 15  # 0-4: resalta fila, 5-9: multiplica por unos, 10-14: proyecta resultado
    
    # Título
    ax.text(6, 5.3, r"$\mathbf{norm\_comb = (Muestra\_combinada^2) \ @ \ On\_d}$", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#1A365D')
    
    # 1. Dibujar Matriz (X**2)
    ax.text(1.5, 4.3, r"$(X^2) \in \mathbb{R}^{1000 \times 784}$", ha='center', fontsize=11, fontweight='bold', color='#2B6CB0')
    for i in range(n_rows):
        is_active = (i == active_row)
        bg_col = '#BEE3F8' if is_active else '#EDF2F7'
        edge_col = '#3182CE' if is_active else '#CBD5E0'
        lw = 2.5 if is_active else 1.0
        
        for j in range(n_cols):
            rect = patches.Rectangle((j*0.8, (2 - i)*0.9 + 1), 0.75, 0.75, 
                                     facecolor=bg_col, edgecolor=edge_col, linewidth=lw)
            ax.add_patch(rect)
            ax.text(j*0.8 + 0.375, (2 - i)*0.9 + 1.375, f"{X_sq[i, j]:.1f}", 
                    ha='center', va='center', fontsize=9, 
                    fontweight='bold' if is_active else 'normal')
            
    # Símbolo @
    ax.text(3.7, 2.2, r"$\mathbf{@}$", ha='center', va='center', fontsize=18, color='#4A5568')
    
    # 2. Dibujar Vector de Unos On_d
    ax.text(4.7, 4.3, r"$On\_d \in \mathbb{R}^{784 \times 1}$", ha='center', fontsize=11, fontweight='bold', color='#2F855A')
    for j in range(n_cols):
        rect = patches.Rectangle((4.7, (3 - j)*0.7 + 0.8), 0.75, 0.6, 
                                 facecolor='#C6F6D5', edgecolor='#38A169', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(4.7 + 0.375, (3 - j)*0.7 + 1.1, "1", ha='center', va='center', fontsize=11, fontweight='bold', color='#22543D')
        
    # Símbolo =
    ax.text(6.0, 2.2, r"$\mathbf{=}$", ha='center', va='center', fontsize=20, color='#4A5568')
    
    # 3. Dibujar Vector Resultado norm_comb
    ax.text(7.5, 4.3, r"$norm\_comb \in \mathbb{R}^{1000 \times 1}$", ha='center', fontsize=11, fontweight='bold', color='#9B2C2C')
    for i in range(n_rows):
        has_result = (i < active_row) or (i == active_row and sub_step >= 10)
        is_cur = (i == active_row and sub_step >= 10)
        
        bg_col = '#FED7D7' if is_cur else ('#FFF5F5' if has_result else '#F7FAFC')
        edge_col = '#E53E3E' if is_cur else ('#FEB2B2' if has_result else '#E2E8F0')
        lw = 2.5 if is_cur else 1.0
        
        rect = patches.Rectangle((7.5, (2 - i)*0.9 + 1), 1.6, 0.75, 
                                 facecolor=bg_col, edgecolor=edge_col, linewidth=lw)
        ax.add_patch(rect)
        
        if has_result:
            val_str = f"||x_{i+1}||² = {norms[i, 0]:.2f}"
            ax.text(7.5 + 0.8, (2 - i)*0.9 + 1.375, val_str, ha='center', va='center', fontsize=9, fontweight='bold', color='#742A2A')
        else:
            ax.text(7.5 + 0.8, (2 - i)*0.9 + 1.375, "?", ha='center', va='center', fontsize=10, color='#A0AEC0')

    # Explicación del cálculo activo abajo
    row_vals = X_sq[active_row]
    terms = [f"({v:.1f} × 1)" for v in row_vals]
    sum_str = " + ".join(terms) + f" = {norms[active_row, 0]:.2f}"
    
    box_rect = patches.FancyBboxPatch((0.5, -0.3), 11, 0.9, boxstyle="round,pad=0.2", 
                                      facecolor='#EDF2F7', edgecolor='#CBD5E0', linewidth=1)
    ax.add_patch(box_rect)
    
    if sub_step < 5:
        msg = f"Paso 1: Seleccionando Fila {active_row + 1} (Foto {active_row + 1})"
    elif sub_step < 10:
        msg = f"Paso 2: Multiplicando píxeles por 1: {sum_str}"
    else:
        msg = f"Paso 3: ¡Sumado e insertado en norm_comb[{active_row}] = {norms[active_row, 0]:.2f}!"
        
    ax.text(6, 0.15, msg, ha='center', va='center', fontsize=11, fontweight='bold', color='#2D3748')

anim = FuncAnimation(fig, draw_frame, frames=45, interval=200)
anim.save('/home/jair/Documents/U/TAM/animacion_norm_comb.gif', writer=PillowWriter(fps=5))
print("Animacion guardada en /home/jair/Documents/U/TAM/animacion_norm_comb.gif")
