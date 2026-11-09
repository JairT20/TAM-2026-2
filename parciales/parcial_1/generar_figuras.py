import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# -------------------------------------------------------------
# FIGURA 1.5: REGRESIÓN REGULARIZADA Y NO REGULARIZADA
# -------------------------------------------------------------
np.random.seed(42)
N, P, Q = 300, 5, 20
sigma_eta, lamda = 0.5, 1.0

X = np.random.normal(0, 1, size=(N, P))
X_linear = X
X_sq = X ** 2
cross_terms = [X[:, i:i+1] * X[:, j:j+1] for i in range(P) for j in range(i+1, P)]
Phi = np.hstack([X_linear, X_sq] + cross_terms)

w_star = np.random.normal(0, 1, size=(Q, 1))
eta = np.random.normal(0, sigma_eta, size=(N, 1))
t = Phi @ w_star + eta

indices = np.random.permutation(N)
N_train = int(0.8 * N)
idx_tr, idx_te = indices[:N_train], indices[N_train:]
Phi_tr, t_tr = Phi[idx_tr], t[idx_tr]
Phi_te, t_te = Phi[idx_te], t[idx_te]

w_ols = np.linalg.inv(Phi_tr.T @ Phi_tr) @ (Phi_tr.T @ t_tr)
w_mle = w_ols.copy()
I_Q = np.eye(Q)
w_ridge = np.linalg.inv(Phi_tr.T @ Phi_tr + lamda * I_Q) @ (Phi_tr.T @ t_tr)
w_map = w_ridge.copy()

mse_tr_ols = np.mean((t_tr - Phi_tr @ w_ols)**2)
mse_te_ols = np.mean((t_te - Phi_te @ w_ols)**2)
mse_tr_ridge = np.mean((t_tr - Phi_tr @ w_ridge)**2)
mse_te_ridge = np.mean((t_te - Phi_te @ w_ridge)**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)

# Panel 1: Comparación de Coeficientes w_hat vs w*
dims = np.arange(1, Q + 1)
ax1.plot(dims, w_star, 'k--', label=r'$w^*$ (Verdadero)', marker='o', markersize=4, alpha=0.8)
ax1.plot(dims, w_ols, color='#0d6efd', label=r'$\hat{w}_{\mathrm{OLS}} = \hat{w}_{\mathrm{MLE}}$', marker='s', markersize=4, linestyle='-')
ax1.plot(dims, w_ridge, color='#dc3545', label=r'$\hat{w}_{\mathrm{Ridge}} = \hat{w}_{\mathrm{MAP}}$', marker='^', markersize=4, linestyle=':')
ax1.set_title(r"Estimación de Coeficientes $\hat{w}$ vs. Vector Real $w^*$", fontsize=11, fontweight='bold')
ax1.set_xlabel("Índice de la Característica (1 a 20)")
ax1.set_ylabel("Valor del Coeficiente")
ax1.set_xticks(dims)
ax1.legend(frameon=True, fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel 2: Comparación de Errores MSE
models = ['OLS', 'MLE', 'Ridge (λ=1)', 'MAP']
x_pos = np.arange(len(models))
width = 0.35
ax2.bar(x_pos - width/2, [mse_tr_ols, mse_tr_ols, mse_tr_ridge, mse_tr_ridge], width, label='MSE Train', color='#4dabf7')
ax2.bar(x_pos + width/2, [mse_te_ols, mse_te_ols, mse_te_ridge, mse_te_ridge], width, label='MSE Test', color='#ff6b6b')
ax2.set_title("Error Cuadrático Medio (MSE): Train vs. Test", fontsize=11, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models)
ax2.set_ylabel("MSE")
ax2.set_ylim(0.20, 0.28)
for i in x_pos:
    ax2.text(i - width/2, [mse_tr_ols, mse_tr_ols, mse_tr_ridge, mse_tr_ridge][i] + 0.002, f"{[mse_tr_ols, mse_tr_ols, mse_tr_ridge, mse_tr_ridge][i]:.4f}", ha='center', fontsize=8)
    ax2.text(i + width/2, [mse_te_ols, mse_te_ols, mse_te_ridge, mse_te_ridge][i] + 0.002, f"{[mse_te_ols, mse_te_ols, mse_te_ridge, mse_te_ridge][i]:.4f}", ha='center', fontsize=8)
ax2.legend(frameon=True, fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig("/home/jair/Documents/U/TAM/figuras/fig_1_5_comparacion.png")
plt.close(fig)
print("Figura 1.5 guardada con éxito.")

# -------------------------------------------------------------
# FIGURA 2.4: MMD^2 Y MATRICES DE GRAM
# -------------------------------------------------------------
path_mnist = "/home/jair/Documents/U/TAM/data/mnist_train.gz"
path_fmnist = "/home/jair/Documents/U/TAM/data/fashion_mnist_train.gz"

def load_data(path):
    with gzip.open(path, 'rb') as f:
        f.read(16)
        buf = f.read(1000 * 28 * 28)
        return np.frombuffer(buf, dtype=np.uint8).astype(np.float64).reshape(1000, 784) / 255.0

X_m = load_data(path_mnist)
X_f = load_data(path_fmnist)

np.random.seed(42)
p_m = np.random.permutation(1000)
p_f = np.random.permutation(1000)
M1, M2 = X_m[p_m[:500]], X_m[p_m[500:]]
F1, F2 = X_f[p_f[:500]], X_f[p_f[500:]]

sigma = 11.0550

def rbf(A, B):
    sq = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1) - 2 * (A @ B.T)
    return np.exp(-np.maximum(sq, 0.0) / (2 * sigma**2))

K_XX = rbf(M1, M1)
K_YY = rbf(F1, F1)
K_XY = rbf(M1, F1)

fig2 = plt.figure(figsize=(11, 4.5), dpi=200)
gs = fig2.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])

# Subplot 1: Barras MMD^2
ax_bar = fig2.add_subplot(gs[0])
casos = ['MNIST_1 vs.\nFashion_1', 'MNIST_1 vs.\nMNIST_2\n(Control)', 'Fashion_1 vs.\nFashion_2\n(Control)']
valores = [0.122343, 0.001486, 0.001328]
colores = ['#dc3545', '#198754', '#0d6efd']
barras = ax_bar.bar(casos, valores, color=colores, width=0.55, edgecolor='black', linewidth=0.8)
ax_bar.set_title(r"Valores Empíricos de $\widehat{\mathrm{MMD}}_b^2$", fontsize=11, fontweight='bold')
ax_bar.set_ylabel(r"$\widehat{\mathrm{MMD}}_b^2$")
ax_bar.grid(True, linestyle='--', alpha=0.5)
for bar in barras:
    h = bar.get_height()
    ax_bar.annotate(f"{h:.6f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

# Subplot 2: Matriz K_XX (MNIST vs MNIST)
ax_kxx = fig2.add_subplot(gs[1])
im1 = ax_kxx.imshow(K_XX[:100, :100], cmap='viridis', vmin=0, vmax=1)
ax_kxx.set_title(r"$K_{XX}$ (MNIST vs. MNIST)", fontsize=10, fontweight='bold')
ax_kxx.set_xlabel("Muestras (primeras 100)")
ax_kxx.set_ylabel("Muestras (primeras 100)")
plt.colorbar(im1, ax=ax_kxx, fraction=0.046, pad=0.04)

# Subplot 3: Matriz K_XY (MNIST vs Fashion)
ax_kxy = fig2.add_subplot(gs[2])
im2 = ax_kxy.imshow(K_XY[:100, :100], cmap='viridis', vmin=0, vmax=1)
ax_kxy.set_title(r"$K_{XY}$ (MNIST vs. Fashion)", fontsize=10, fontweight='bold')
ax_kxy.set_xlabel("Fashion (primeras 100)")
ax_kxy.set_ylabel("MNIST (primeras 100)")
plt.colorbar(im2, ax=ax_kxy, fraction=0.046, pad=0.04)

plt.tight_layout()
fig2.savefig("/home/jair/Documents/U/TAM/figuras/fig_2_4_mmd.png")
plt.close(fig2)
print("Figura 2.4 guardada con éxito.")
