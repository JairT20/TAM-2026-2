import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

# Configuración de figura con estilo profesional
fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
fig.patch.set_facecolor('#F8FAFC')

def draw_frame(frame):
    ax.clear()
    ax.set_facecolor('#F8FAFC')
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 7)
    ax.axis('off')
    
    # 4 fases:
    # 0 a 24: Traje 1 (Vectores Euclidianos - Bloques 1 y 3)
    # 25 a 49: Traje 2 (Espacio RKHS y Funciones - Bloque 2)
    # 50 a 74: Traje 3 (Matrices y Frobenius - Bloque 3)
    # 75 a 95: Síntesis / Mapa de los 3 Trajes
    phase = frame // 25
    t = (frame % 25) / 24.0  # de 0.0 a 1.0 en cada fase
    
    if phase == 0:
        # ==========================================
        # TRAJE 1: VECTORES EUCLIDIANOS
        # ==========================================
        ax.text(6, 6.4, "TRAJE 1: VECTORES EUCLIDIANOS (Bloques 1 y 3)", 
                ha='center', va='center', fontsize=14, fontweight='bold', color='#1E293B')
        ax.text(6, 5.85, r"$\mathbf{a^\top b = \sum_{i=1}^N a_i b_i}$  (Producto punto clásico fila $\times$ columna)", 
                ha='center', va='center', fontsize=12, color='#2563EB')
        
        # Panel izquierdo: Norma al cuadrado (Error en OLS / Ridge)
        p1 = patches.FancyBboxPatch((0.2, 0.5), 5.8, 4.8, boxstyle="round,pad=0.2",
                                    facecolor='#FFFFFF', edgecolor='#93C5FD', linewidth=2)
        ax.add_patch(p1)
        ax.text(3.1, 4.8, "1. Longitud al cuadrado (Bloque 1)", ha='center', fontsize=11, fontweight='bold', color='#1D4ED8')
        ax.text(3.1, 4.3, r"$e^\top e = \|e\|_2^2 = \sum e_i^2$", ha='center', fontsize=12, color='#1E40AF')
        
        # Dibujar vector de error
        ang = np.pi/6 + 0.1 * np.sin(t * np.pi)
        vx, vy = 2.2 * np.cos(ang), 2.2 * np.sin(ang)
        ax.arrow(1.5, 1.8, vx, vy, head_width=0.15, head_length=0.2, fc='#DC2626', ec='#DC2626', lw=2.5, length_includes_head=True)
        ax.text(1.5 + vx/2 - 0.2, 1.8 + vy/2 + 0.2, r"$\vec{e} = t - \Phi w$", fontsize=10, fontweight='bold', color='#DC2626')
        ax.plot([1.5, 1.5 + vx], [1.8, 1.8], '--', color='#94A3B8')
        ax.plot([1.5 + vx, 1.5 + vx], [1.8, 1.8 + vy], '--', color='#94A3B8')
        ax.text(3.1, 1.1, "Multiplicar un vector por sí mismo\nes su distancia o error al cuadrado", ha='center', fontsize=9, color='#475569')

        # Panel derecho: Correlación / Covarianza (Pearson en Bloque 3)
        p2 = patches.FancyBboxPatch((6.4, 0.5), 5.8, 4.8, boxstyle="round,pad=0.2",
                                    facecolor='#FFFFFF', edgecolor='#86EFAC', linewidth=2)
        ax.add_patch(p2)
        ax.text(9.3, 4.8, "2. Covarianza Centrada (Bloque 3)", ha='center', fontsize=11, fontweight='bold', color='#15803D')
        ax.text(9.3, 4.3, r"$x^\top H y = \sum (x_i - \bar{x})(y_i - \bar{y})$", ha='center', fontsize=12, color='#166534')
        
        # Puntos de correlación
        np.random.seed(10)
        xs = np.linspace(7.2, 11.2, 7)
        ys = 1.6 + 0.45 * (xs - 7.2) + 0.15 * np.sin(xs * 4 + t * 2 * np.pi)
        ax.scatter(xs, ys, color='#16A34A', s=60, zorder=4)
        ax.plot([7.2, 11.2], [1.6, 1.6 + 0.45 * 4], '-', color='#22C55E', lw=2)
        ax.text(9.3, 1.1, "Mide si dos variables van en la misma\ndirección o en direcciones opuestas", ha='center', fontsize=9, color='#475569')

    elif phase == 1:
        # ==========================================
        # TRAJE 2: ESPACIO DE FUNCIONES RKHS
        # ==========================================
        ax.text(6, 6.4, "TRAJE 2: ESPACIO DE FUNCIONES RKHS (Bloque 2)", 
                ha='center', va='center', fontsize=14, fontweight='bold', color='#1E293B')
        ax.text(6, 5.85, r"$\mathbf{\langle f, g \rangle_{\mathcal{H}_k}}$  (Producto interno entre funciones de dimensión infinita)", 
                ha='center', va='center', fontsize=12, color='#7C3AED')
        
        # Panel izquierdo: Propiedad Reproductora (Máquina de evaluar)
        p1 = patches.FancyBboxPatch((0.2, 0.5), 5.8, 4.8, boxstyle="round,pad=0.2",
                                    facecolor='#FFFFFF', edgecolor='#DDD6FE', linewidth=2)
        ax.add_patch(p1)
        ax.text(3.1, 4.8, "1. La Propiedad Reproductora", ha='center', fontsize=11, fontweight='bold', color='#6D28D9')
        ax.text(3.1, 4.3, r"$\langle f, k(x, \cdot) \rangle_{\mathcal{H}_k} = f(x)$", ha='center', fontsize=13, color='#5B21B6')
        
        # Dibujar función f(t) y kernel k(x, .)
        xx = np.linspace(0.8, 5.4, 60)
        f_curve = 2.4 + 0.6 * np.sin(xx * 2)
        x_eval = 2.5 + 1.2 * np.sin(t * np.pi)
        y_eval = 2.4 + 0.6 * np.sin(x_eval * 2)
        k_bump = y_eval * np.exp(-((xx - x_eval)**2) / 0.3) + 1.6
        
        ax.plot(xx, f_curve, color='#7C3AED', lw=2.5, label='f(·)')
        ax.plot(xx, k_bump, '--', color='#EC4899', lw=1.5, label='k(x, ·)')
        ax.scatter([x_eval], [y_eval], color='#DC2626', s=80, zorder=5)
        ax.plot([x_eval, x_eval], [1.5, y_eval], ':', color='#DC2626', lw=1.5)
        ax.text(x_eval, y_eval + 0.3, f"f({x_eval:.1f})", ha='center', fontsize=9, fontweight='bold', color='#DC2626')
        ax.text(3.1, 1.1, "¡No hace falta integrar!\nEl producto interno evalúa la función directamente", ha='center', fontsize=9, color='#475569')

        # Panel derecho: MMD entre poblaciones
        p2 = patches.FancyBboxPatch((6.4, 0.5), 5.8, 4.8, boxstyle="round,pad=0.2",
                                    facecolor='#FFFFFF', edgecolor='#FBCFE8', linewidth=2)
        ax.add_patch(p2)
        ax.text(9.3, 4.8, "2. Distancia MMD (Embeddings)", ha='center', fontsize=11, fontweight='bold', color='#BE185D')
        ax.text(9.3, 4.3, r"$\|\mu_P - \mu_Q\|^2 = \langle \mu_P, \mu_P \rangle + \langle \mu_Q, \mu_Q \rangle - 2\langle \mu_P, \mu_Q \rangle$", ha='center', fontsize=9, color='#9D174D')
        
        # Puntos MNIST vs Fashion y sus baricentros
        ax.scatter([7.5, 7.8, 7.4, 8.0], [2.8, 3.2, 3.5, 2.6], color='#3B82F6', s=40, alpha=0.6, label='MNIST')
        ax.scatter([10.2, 10.8, 10.5, 11.0], [2.5, 3.1, 3.4, 2.7], color='#F97316', s=40, alpha=0.6, label='Fashion')
        mu_p = np.array([7.675, 3.025])
        mu_q = np.array([10.625, 2.925])
        ax.scatter([mu_p[0]], [mu_p[1]], color='#1D4ED8', s=120, marker='*', zorder=5)
        ax.scatter([mu_q[0]], [mu_q[1]], color='#C2410C', s=120, marker='*', zorder=5)
        ax.annotate('', xy=mu_q, xytext=mu_p, arrowprops=dict(arrowstyle="<->", color="#BE185D", lw=2.5))
        ax.text(9.15, 3.25, r"$\mathbf{MMD}$", ha='center', fontsize=11, fontweight='bold', color='#BE185D')
        ax.text(9.3, 1.1, "Mide la distancia entre los centros\nde gravedad de dos distribuciones enteras", ha='center', fontsize=9, color='#475569')

    elif phase == 2:
        # ==========================================
        # TRAJE 3: MATRICES Y FROBENIUS
        # ==========================================
        ax.text(6, 6.4, "TRAJE 3: MATRICES 2D / FROBENIUS (Bloque 3)", 
                ha='center', va='center', fontsize=14, fontweight='bold', color='#1E293B')
        ax.text(6, 5.85, r"$\mathbf{\langle A, B \rangle_F = \mathrm{tr}(A^\top B) = \sum A_{ij} B_{ij}}$  (Producto interno entre matrices)", 
                ha='center', va='center', fontsize=12, color='#D97706')
        
        # Panel izquierdo: Alineamiento de Kernels
        p1 = patches.FancyBboxPatch((0.2, 0.5), 5.8, 4.8, boxstyle="round,pad=0.2",
                                    facecolor='#FFFFFF', edgecolor='#FDE68A', linewidth=2)
        ax.add_patch(p1)
        ax.text(3.1, 4.8, "1. Alineamiento de Gram Centrado", ha='center', fontsize=11, fontweight='bold', color='#B45309')
        ax.text(3.1, 4.3, r"$\langle K_c, L_c \rangle_F = \mathrm{tr}(K_X H K_Y H)$", ha='center', fontsize=12, color='#92400E')
        
        # Representación de 2 matrices pequeñas
        kx_grid = np.array([[1.0, 0.8], [0.8, 1.0]])
        ky_grid = np.array([[1.0, 0.7], [0.7, 1.0]])
        for r in range(2):
            for c in range(2):
                ax.add_patch(patches.Rectangle((1.2 + c*0.8, 3.0 - r*0.8), 0.75, 0.75, facecolor='#FEF3C7', edgecolor='#D97706'))
                ax.text(1.2 + c*0.8 + 0.375, 3.0 - r*0.8 + 0.375, f"{kx_grid[r, c]:.1f}", ha='center', va='center', fontsize=9)
        ax.text(3.1, 2.6, "*", fontsize=20, ha='center', va='center', color='#B45309', fontweight='bold')
        for r in range(2):
            for c in range(2):
                ax.add_patch(patches.Rectangle((3.6 + c*0.8, 3.0 - r*0.8), 0.75, 0.75, facecolor='#ECFDF5', edgecolor='#059669'))
                ax.text(3.6 + c*0.8 + 0.375, 3.0 - r*0.8 + 0.375, f"{ky_grid[r, c]:.1f}", ha='center', va='center', fontsize=9)
        ax.text(3.1, 1.1, "Multiplica similitud con similitud:\n¿Los pares cercanos en X son cercanos en Y?", ha='center', fontsize=9, color='#475569')

        # Panel derecho: Pérdida del signo
        p2 = patches.FancyBboxPatch((6.4, 0.5), 5.8, 4.8, boxstyle="round,pad=0.2",
                                    facecolor='#FFFFFF', edgecolor='#FED7AA', linewidth=2)
        ax.add_patch(p2)
        ax.text(9.3, 4.8, "2. ¿Por qué pierde el signo? (R²)", ha='center', fontsize=11, fontweight='bold', color='#C2410C')
        ax.text(9.3, 4.3, r"$\rho_{\mathrm{RKHS}} = \hat{\rho}_{XY}^2 \in [0, 1]$", ha='center', fontsize=13, color='#9A3412')
        
        # Animación de flecha que se invierte pero el kernel no cambia
        flip = -1 if (t > 0.5) else 1
        ax.arrow(8.5, 2.5, 1.5, 0.9 * flip, head_width=0.15, head_length=0.2, fc='#EA580C', ec='#EA580C', lw=2.5, length_includes_head=True)
        sign_str = "Pendiente Positiva (+)" if flip > 0 else "Pendiente Negativa (-)"
        ax.text(9.3, 3.8, sign_str, ha='center', fontsize=10, fontweight='bold', color='#EA580C')
        ax.text(9.3, 2.0, r"$K_{-y} = (-y)(-y)^\top = yy^\top = K_y$", ha='center', fontsize=10, fontweight='bold', color='#7C2D12')
        ax.text(9.3, 1.1, "Al kernel no le importa si sube o baja:\nRegistra distancias relativas, perdiendo el signo", ha='center', fontsize=9, color='#475569')

    else:
        # ==========================================
        # FASE 4: SÍNTESIS / MAPA MENTAL UNIFICADO
        # ==========================================
        ax.text(6, 6.4, "SÍNTESIS: EL PRODUCTO INTERNO ES EL MISMO CONCEPTO", 
                ha='center', va='center', fontsize=14, fontweight='bold', color='#0F172A')
        ax.text(6, 5.85, "Cambia de 'ropa' según el espacio, pero siempre mide PROYECCIÓN Y SIMILITUD", 
                ha='center', va='center', fontsize=11, color='#475569')
        
        cols = [
            ("Vectores (1D)", "#2563EB", "#DBEAFE", r"$a^\top b = \sum a_i b_i$", "• Errores al cuadrado: ||e||²\n• Desvíos de Pearson: xᵀHy\n• Espacio Euclidiano común"),
            ("Funciones RKHS", "#7C3AED", "#EDE9FE", r"$\langle f, k(x, \cdot) \rangle = f(x)$", "• Máquina de evaluar funciones\n• Similitud entre poblaciones\n• Distancia MMD (baricentros)"),
            ("Matrices 2D", "#D97706", "#FEF3C7", r"$\mathrm{tr}(A^\top B) = \sum A_{ij} B_{ij}$", "• Alineamiento de similitudes\n• Frobenius en matrices de Gram\n• Colapsa a Pearson² (pierde signo)")
        ]
        
        for idx, (title, stroke, fill, formula, desc) in enumerate(cols):
            x_pos = 0.2 + idx * 4.2
            panel = patches.FancyBboxPatch((x_pos, 0.5), 3.9, 4.8, boxstyle="round,pad=0.25",
                                           facecolor=fill, edgecolor=stroke, linewidth=2.5)
            ax.add_patch(panel)
            ax.text(x_pos + 1.95, 4.8, title, ha='center', fontsize=12, fontweight='bold', color=stroke)
            
            # Badge fórmula
            badge = patches.FancyBboxPatch((x_pos + 0.2, 3.8), 3.5, 0.7, boxstyle="round,pad=0.1",
                                           facecolor='#FFFFFF', edgecolor=stroke, linewidth=1)
            ax.add_patch(badge)
            ax.text(x_pos + 1.95, 4.15, formula, ha='center', va='center', fontsize=10, fontweight='bold', color=stroke)
            
            ax.text(x_pos + 0.4, 2.3, desc, ha='left', va='center', fontsize=9.5, color='#1E293B', linespacing=1.6)

anim = FuncAnimation(fig, draw_frame, frames=96, interval=150)
output_path = '/home/jair/Documents/U/TAM/animacion_producto_interno.gif'
anim.save(output_path, writer=PillowWriter(fps=6))
print(f"Animación guardada exitosamente en {output_path}")
