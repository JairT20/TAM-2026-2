import numpy as np
import matplotlib.pyplot as plt

# Fijamos semilla para reproducibilidad absoluta
np.random.seed(42)

print("=" * 65)
print(" PUNTO 1.5: REGRESIÓN OLS, MLE, RIDGE Y MAP (N=300, P=5, Q=20)")
print("=" * 65)

# 1. Configuración de parámetros
N = 300
P = 5
Q = 20
sigma_eta = 0.5
lamda = 1.0

# 2. Generación de matriz X (300 x 5)
X = np.random.normal(0, 1, size=(N, P))

# 3. Construcción reproducible de Phi(X) (Q = 20 características)
# - 5 características lineales: x_0, ..., x_4
# - 5 características cuadráticas: x_0^2, ..., x_4^2
# - 10 interacciones por pares (i < j): x_i * x_j
X_linear = X
X_sq = X ** 2
cross_terms = [X[:, i:i+1] * X[:, j:j+1] for i in range(P) for j in range(i+1, P)]
Phi = np.hstack([X_linear, X_sq] + cross_terms)

print(f"Dimensiones de Phi(X): {Phi.shape}")

# 4. Parámetro real w* y target t con ruido gaussiano
w_star = np.random.normal(0, 1, size=(Q, 1))
eta = np.random.normal(0, sigma_eta, size=(N, 1))
t = Phi @ w_star + eta

# 5. Partición aleatoria: 80% train (240) y 20% test (60)
indices = np.random.permutation(N)
N_train = int(0.8 * N)
idx_tr, idx_te = indices[:N_train], indices[N_train:]

Phi_tr, t_tr = Phi[idx_tr], t[idx_tr]
Phi_te, t_te = Phi[idx_te], t[idx_te]

# 6. Ajuste algebraico de los cuatro estimadores
I_Q = np.eye(Q)

# (a) OLS
w_ols = np.linalg.inv(Phi_tr.T @ Phi_tr) @ (Phi_tr.T @ t_tr)

# (b) MLE (idéntico a OLS con ruido gaussiano)
w_mle = np.linalg.inv(Phi_tr.T @ Phi_tr) @ (Phi_tr.T @ t_tr)

# (c) Ridge (lambda = 1.0)
w_ridge = np.linalg.inv(Phi_tr.T @ Phi_tr + lamda * I_Q) @ (Phi_tr.T @ t_tr)

# (d) MAP (prior compatible: sigma_w^2 = sigma_eta^2 / lambda = 0.25 => ratio = 1.0)
sigma_w2 = (sigma_eta ** 2) / lamda
alpha_map = (sigma_eta ** 2) / sigma_w2
w_map = np.linalg.inv(Phi_tr.T @ Phi_tr + alpha_map * I_Q) @ (Phi_tr.T @ t_tr)

# 7. Evaluación y reporte detallado de resultados
def evaluar(w_hat, nombre):
    mse_tr = np.mean((t_tr - Phi_tr @ w_hat) ** 2)
    mse_te = np.mean((t_te - Phi_te @ w_hat) ** 2)
    norm_diff = np.linalg.norm(w_hat - w_star)
    print(f"{nombre:22s} | MSE Train: {mse_tr:.6f} | MSE Test: {mse_te:.6f} | ||w - w*||: {norm_diff:.6f}")
    return mse_tr, mse_te, norm_diff

print("\n--- RESULTADOS NUMÉRICOS ---")
m_ols = evaluar(w_ols, "Mínimos Cuadrados (OLS)")
m_mle = evaluar(w_mle, "Máx. Verosimilitud (MLE)")
m_rdg = evaluar(w_ridge, "Regularización Ridge")
m_map = evaluar(w_map, "Máx. a Posteriori (MAP)")

print("\n--- COMPROBACIÓN DE COINCIDENCIAS TEÓRICAS ---")
print(f"Discrepancia máx |w_ols - w_mle|:     {np.max(np.abs(w_ols - w_mle)):.2e}")
print(f"Discrepancia máx |w_ridge - w_map|: {np.max(np.abs(w_ridge - w_map)):.2e}")

# 8. Visualización gráfica de resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Comparación de coeficientes estimados vs reales
dims = np.arange(1, Q + 1)
ax1.plot(dims, w_star, 'k--', marker='o', label=r'$w^*$ Real', linewidth=1.5)
ax1.plot(dims, w_ols, 'b-', marker='s', label=r'$\hat{w}_{\mathrm{OLS}} = \hat{w}_{\mathrm{MLE}}$', alpha=0.8)
ax1.plot(dims, w_ridge, 'r:', marker='^', label=r'$\hat{w}_{\mathrm{Ridge}} = \hat{w}_{\mathrm{MAP}}$', alpha=0.8)
ax1.set_title("Comparación de Coeficientes Estimados", fontsize=11, fontweight='bold')
ax1.set_xlabel("Índice de la Característica (1 a 20)")
ax1.set_ylabel("Valor del Coeficiente")
ax1.set_xticks(dims)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Comparación de MSE
modelos = ['OLS', 'MLE', 'Ridge (λ=1)', 'MAP']
x = np.arange(len(modelos))
width = 0.35
ax2.bar(x - width/2, [m_ols[0], m_mle[0], m_rdg[0], m_map[0]], width, label='MSE Train', color='skyblue')
ax2.bar(x + width/2, [m_ols[1], m_mle[1], m_rdg[1], m_map[1]], width, label='MSE Test', color='salmon')
ax2.set_title("MSE: Train vs. Test", fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(modelos)
ax2.set_ylabel("Error Cuadrático Medio")
ax2.set_ylim(0.20, 0.28)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
print("\nMostrando gráficos de resultados...")
plt.show()
