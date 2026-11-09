import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

fig = plt.figure(figsize=(11, 5.5), dpi=100)

# Datos sintéticos para representar dos distribuciones
np.random.seed(42)
n_pts = 35

# Distribución P (círculo / anillo)
theta = np.linspace(0, 2*np.pi, n_pts)
r_P = 1.0 + 0.15 * np.random.randn(n_pts)
X_P = np.stack([r_P * np.cos(theta), r_P * np.sin(theta)], axis=1)

# Distribución Q (grupo desplazado)
X_Q = np.random.randn(n_pts, 2) * 0.4 + np.array([2.0, 0.8])

# Mapeo a RKHS simulado (espacio 3D levantado)
def phi_map(X, mode='P'):
    if mode == 'P':
        z = np.sin(X[:, 0] * 1.5) + np.cos(X[:, 1] * 1.5) + 2.0
        return np.column_stack([X[:, 0]*0.8, X[:, 1]*0.8, z])
    else:
        z = 0.5 * (X[:, 0]**2 + X[:, 1]**2) - 0.5
        return np.column_stack([X[:, 0]*0.7 + 0.5, X[:, 1]*0.7, z + 2.0])

Phi_P = phi_map(X_P, 'P')
Phi_Q = phi_map(X_Q, 'Q')

mu_P = np.mean(Phi_P, axis=0)
mu_Q = np.mean(Phi_Q, axis=0)

def draw_frame(frame):
    fig.clear()
    
    # 4 fases de animación (60 frames total, 15 por fase)
    phase = (frame // 15) % 4
    sub = (frame % 15) / 15.0
    
    # Subplot 1: Espacio Original X (2D)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(-2.0, 3.5)
    ax1.set_ylim(-2.0, 2.5)
    ax1.set_title(r"1. Espacio Original $\mathcal{X}$ ($\mathbb{R}^{784}$)", fontsize=12, fontweight='bold', color='#1A365D')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_xlabel("Caracteristica 1")
    ax1.set_ylabel("Caracteristica 2")
    
    # Dibujar puntos en espacio original
    ax1.scatter(X_P[:, 0], X_P[:, 1], color='#3182CE', alpha=0.7, s=40, label=r'Distribucion $P$ (MNIST)')
    ax1.scatter(X_Q[:, 0], X_Q[:, 1], color='#E53E3E', alpha=0.7, s=40, label=r'Distribucion $Q$ (Fashion)')
    ax1.legend(loc='upper left', fontsize=8.5)
    
    # Subplot 2: Espacio RKHS H_k (3D simulando dimensión infinita)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_xlim(-2, 3)
    ax2.set_ylim(-2, 3)
    ax2.set_zlim(0, 4)
    ax2.set_title(r"2. Espacio RKHS $\mathcal{H}_k$ (Dimension $\infty$)", fontsize=12, fontweight='bold', color='#2F855A')
    ax2.view_init(elev=20, azim=45 + frame*0.5)
    
    # En fase 0: solo espacio original activo
    # En fase >= 1: proyectar puntos al RKHS
    if phase >= 1:
        alpha_pts = min(1.0, sub * 2) if phase == 1 else 0.5
        ax2.scatter(Phi_P[:, 0], Phi_P[:, 1], Phi_P[:, 2], color='#3182CE', alpha=alpha_pts, s=25)
        ax2.scatter(Phi_Q[:, 0], Phi_Q[:, 1], Phi_Q[:, 2], color='#E53E3E', alpha=alpha_pts, s=25)
        
    # En fase >= 2: calcular y mostrar baricentros mu_P y mu_Q
    if phase >= 2:
        s_bar = 120 + int(20 * np.sin(sub * np.pi))
        ax2.scatter(mu_P[0], mu_P[1], mu_P[2], color='#2B6CB0', s=s_bar, edgecolor='black', lw=1.5, depthshade=False)
        ax2.text(mu_P[0], mu_P[1], mu_P[2]+0.35, r"$\mu_P$ (Baricentro)", fontsize=10, fontweight='bold', color='#1A365D')
        
        ax2.scatter(mu_Q[0], mu_Q[1], mu_Q[2], color='#C53030', s=s_bar, edgecolor='black', lw=1.5, depthshade=False)
        ax2.text(mu_Q[0], mu_Q[1], mu_Q[2]+0.35, r"$\mu_Q$ (Baricentro)", fontsize=10, fontweight='bold', color='#742A2A')
        
    # En fase == 3: trazar distancia MMD entre baricentros
    if phase == 3:
        ax2.plot([mu_P[0], mu_Q[0]], [mu_P[1], mu_Q[1]], [mu_P[2], mu_Q[2]], 
                 color='#805AD5', linestyle='-', linewidth=3.5)
        
        mid = (mu_P + mu_Q) / 2.0
        ax2.text(mid[0], mid[1], mid[2]+0.45, 
                 r"MMD = $\|\mu_P - \mu_Q\|_{\mathcal{H}}$", 
                 fontsize=11, fontweight='bold', color='#553C9A', ha='center')

    # Cuadro de texto explicativo en la parte inferior
    if phase == 0:
        desc = "Paso 1: Dos distribuciones complejas P y Q en el espacio original de pixeles."
    elif phase == 1:
        desc = "Paso 2: El kernel k(x, .) 'levanta' cada dato como una funcion en el RKHS de dimension infinita."
    elif phase == 2:
        desc = "Paso 3: Calculamos el Baricentro (centro de gravedad) de cada distribucion: mu_P y mu_Q."
    else:
        desc = "Paso 4: MMD es la distancia euclidiana en linea recta entre ambos baricentros: ||mu_P - mu_Q||."

    fig.text(0.5, 0.03, desc, ha='center', va='center', fontsize=11, fontweight='bold', 
             color='#1A202C', bbox=dict(boxstyle="round,pad=0.4", facecolor='#EDF2F7', edgecolor='#CBD5E0'))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

anim = FuncAnimation(fig, draw_frame, frames=60, interval=200)
anim.save('/home/jair/Documents/U/TAM/animacion_rkhs_mmd.gif', writer=PillowWriter(fps=5))
print("Animacion guardada en /home/jair/Documents/U/TAM/animacion_rkhs_mmd.gif")
