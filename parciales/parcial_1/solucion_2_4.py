import numpy as np
import matplotlib.pyplot as plt

# Semilla fija para reproducibilidad absoluta
np.random.seed(42)

print("=" * 65)
print(" PUNTO 2.4: MMD^2 EN MNIST Y FASHION-MNIST (KERNEL GAUSSIANO)")
print("=" * 65)

# 1. Carga de datos
# En Google Colab se usa directamente tensorflow.keras.datasets.
# Incluimos fallback transparente a la caché local por si se corre en una terminal sin TensorFlow.
try:
    from tensorflow.keras.datasets import mnist, fashion_mnist
    print("Cargando datasets desde tensorflow.keras.datasets...")
    (X_m_tr, _), _ = mnist.load_data()
    (X_f_tr, _), _ = fashion_mnist.load_data()
    X_mnist_all = X_m_tr[:1000].reshape(-1, 28 * 28) / 255.0
    X_fmnist_all = X_f_tr[:1000].reshape(-1, 28 * 28) / 255.0
except Exception:
    import gzip, os
    print("Cargando datasets desde caché local...")
    def cargar_local(ruta):
        with gzip.open(ruta, 'rb') as f:
            f.read(16)
            buf = f.read(1000 * 28 * 28)
            return np.frombuffer(buf, dtype=np.uint8).astype(np.float64).reshape(1000, 784) / 255.0
    X_mnist_all = cargar_local("/home/jair/Documents/U/TAM/data/mnist_train.gz")
    X_fmnist_all = cargar_local("/home/jair/Documents/U/TAM/data/fashion_mnist_train.gz")

print(f"Muestras totales cargadas: MNIST={X_mnist_all.shape}, Fashion-MNIST={X_fmnist_all.shape}")

# 2. Partición aleatoria en dos grupos disjuntos de 500 imágenes (n = m = 500)
perm_m = np.random.permutation(1000)
MNIST_1, MNIST_2 = X_mnist_all[perm_m[:500]], X_mnist_all[perm_m[500:]]

perm_f = np.random.permutation(1000)
FASHION_1, FASHION_2 = X_fmnist_all[perm_f[:500]], X_fmnist_all[perm_f[500:]]

print(f"Particiones listas: n = {len(MNIST_1)}, m = {len(FASHION_1)} imágenes de 784 píxeles.")

# 3. Heurística de la mediana sobre muestra combinada (1000 imágenes)
Muestra_combinada = np.vstack([MNIST_1, FASHION_1])
# Vector de unos explícito para la muestra combinada (1000 x 1)
N_comb = len(Muestra_combinada)
On_comb = np.ones((N_comb, 1))

# Norma al cuadrado de cada imagen: vector columna (1000 x 1)
norm_comb = np.sum(Muestra_combinada**2, axis=1, keepdims=True)

# Matriz de distancias cuadráticas (1000 x 1000) por expansión matricial: (G @ 1.T) + (1 @ G.T) - 2 * (M @ M.T)
sq_dists_comb = (norm_comb @ On_comb.T) + (On_comb @ norm_comb.T) - 2 * (Muestra_combinada @ Muestra_combinada.T)
sq_dists_comb = np.maximum(sq_dists_comb, 0.0)

indices_triu = np.triu_indices(N_comb, k=1)
distancias_pares = np.sqrt(sq_dists_comb[indices_triu])
sigma = np.median(distancias_pares)

print(f"\n-> Ancho de banda calibrado por mediana (sigma): {sigma:.4f}")

# 4. Función de Kernel Gaussiano (RBF) con vectores de unos explícitos
def rbf_kernel(X1, X2, sigma):
    n = X1.shape[0]
    m = X2.shape[0]
    
    On_n = np.ones((n, 1))
    On_m = np.ones((m, 1))
    
    norm_X1 = np.sum(X1**2, axis=1, keepdims=True)  # (n, 1)
    norm_X2 = np.sum(X2**2, axis=1, keepdims=True)  # (m, 1)
    
    # Expansión euclidiana cuadrática: (norm_X1 @ 1_m.T) + (1_n @ norm_X2.T) - 2 * (X1 @ X2.T)
    sq_dist = (norm_X1 @ On_m.T) + (On_n @ norm_X2.T) - 2 * (X1 @ X2.T)
    return np.exp(-np.maximum(sq_dist, 0.0) / (2 * (sigma ** 2)))

# 5. Estimador Empírico Sesgado Matricial de MMD^2
def calcular_mmd2(X, Y, sigma):
    n = X.shape[0]
    m = Y.shape[0]
    
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)
    
    On_n = np.ones((n, 1))
    On_m = np.ones((m, 1))
    
    term_XX = (1.0 / (n**2)) * (On_n.T @ K_XX @ On_n)[0, 0]
    term_YY = (1.0 / (m**2)) * (On_m.T @ K_YY @ On_m)[0, 0]
    term_XY = (2.0 / (n * m)) * (On_n.T @ K_XY @ On_m)[0, 0]
    
    mmd2 = term_XX + term_YY - term_XY
    return mmd2, term_XX, term_YY, term_XY, K_XX, K_YY, K_XY

# 6. Cálculo y reporte de las 3 comparaciones con el mismo sigma
mmd_cross, tXX_c, tYY_c, tXY_c, K_XX, K_YY, K_XY = calcular_mmd2(MNIST_1, FASHION_1, sigma)
mmd_null_m, tXX_m, tYY_m, tXY_m, _, _, _ = calcular_mmd2(MNIST_1, MNIST_2, sigma)
mmd_null_f, tXX_f, tYY_f, tXY_f, _, _, _ = calcular_mmd2(FASHION_1, FASHION_2, sigma)

print("\n" + "=" * 65)
print(" RESULTADOS NUMÉRICOS DE MMD^2")
print("=" * 65)
print(f"1. MNIST_1 vs. Fashion_1:       MMD^2 = {mmd_cross:.6f}")
print(f"   (Term XX: {tXX_c:.6f}, Term YY: {tYY_c:.6f}, Term XY: {tXY_c:.6f})")
print(f"2. MNIST_1 vs. MNIST_2 (Nulo):  MMD^2 = {mmd_null_m:.6f}")
print(f"   (Term XX: {tXX_m:.6f}, Term YY: {tYY_m:.6f}, Term XY: {tXY_m:.6f})")
print(f"3. Fashion_1 vs. Fashion_2 (Nulo): MMD^2 = {mmd_null_f:.6f}")
print(f"   (Term XX: {tXX_f:.6f}, Term YY: {tYY_f:.6f}, Term XY: {tXY_f:.6f})")
print(f"\nDimensiones matrices de Gram: K_XX={K_XX.shape}, K_YY={K_YY.shape}, K_XY={K_XY.shape}")

# 7. Visualización gráfica (Estilo de clase del profesor)
fig = plt.figure(figsize=(13, 4.5))

# Subplot 1: Gráfico de barras de MMD^2
ax1 = fig.add_subplot(1, 3, 1)
casos = ['MNIST_1 vs.\nFashion_1', 'MNIST_1 vs.\nMNIST_2\n(Nulo)', 'Fashion_1 vs.\nFashion_2\n(Nulo)']
valores = [mmd_cross, mmd_null_m, mmd_null_f]
colores = ['firebrick', 'forestgreen', 'royalblue']
barras = ax1.bar(casos, valores, color=colores, width=0.55, edgecolor='black')
ax1.set_title(r"Valores de $\widehat{\mathrm{MMD}}_b^2$", fontsize=11, fontweight='bold')
ax1.set_ylabel(r"$\widehat{\mathrm{MMD}}_b^2$")
ax1.grid(True, linestyle='--', alpha=0.5)
for bar in barras:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.6f}", ha='center', fontsize=9, fontweight='bold')

# Subplot 2: Matriz K_XX (primeras 100 muestras)
ax2 = fig.add_subplot(1, 3, 2)
im1 = ax2.imshow(K_XX[:100, :100], cmap='viridis', vmin=0, vmax=1)
ax2.set_title(r"Matriz $K_{XX}$ (MNIST vs. MNIST)", fontsize=10, fontweight='bold')
ax2.set_xlabel("Muestras (100)")
ax2.set_ylabel("Muestras (100)")
plt.colorbar(im1, ax=ax2, fraction=0.046, pad=0.04)

# Subplot 3: Matriz K_XY cruzada (primeras 100 muestras)
ax3 = fig.add_subplot(1, 3, 3)
im2 = ax3.imshow(K_XY[:100, :100], cmap='viridis', vmin=0, vmax=1)
ax3.set_title(r"Matriz $K_{XY}$ (MNIST vs. Fashion)", fontsize=10, fontweight='bold')
ax3.set_xlabel("Fashion-MNIST (100)")
ax3.set_ylabel("MNIST (100)")
plt.colorbar(im2, ax=ax3, fraction=0.046, pad=0.04)

plt.tight_layout()
print("\nMostrando gráficos de resultados...")
plt.show()
