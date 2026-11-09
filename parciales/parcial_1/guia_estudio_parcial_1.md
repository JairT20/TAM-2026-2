# Guía de Estudio Oficial: Parcial 1 TAM (2026-2)
**Materia:** Teoría de Aprendizaje de Máquina  
**Profesor:** Diego Armando Pérez Rosero --- Universidad Nacional de Colombia  
**Documento base:** `Parcial_1_TAM_2026_2_Pearson_RKHS.pdf`

---

## 1. Regresión en Espacios de Características

**Definición general del problema (enunciado oficial):**  
Sea un conjunto de datos i.i.d. $\{(x_n, t_n)\}_{n=1}^N$ con $x_n \in \mathbb{R}^P$ y $t_n \in \mathbb{R}$. Considere un mapeo explícito $\phi: \mathbb{R}^P \to \mathbb{R}^Q$ y el modelo de regresión:
$$t_n = \phi(x_n)^\top \mathbf{w} + \eta_n, \quad \mathbf{w} \in \mathbb{R}^Q, \quad \eta_n \sim \mathcal{N}(0, \sigma_\eta^2) \text{ (i.i.d.)}$$
Sea $\mathbf{\Phi} \in \mathbb{R}^{N \times Q}$ la matriz de diseño cuyas filas son $\phi(x_n)^\top$, y $\mathbf{t} \in \mathbb{R}^{N \times 1}$ el vector de respuestas.

---

### 1.1 Mínimos Cuadrados (OLS)

> **Enunciado Oficial:**  
> *Plantee la función objetivo y deduzca, paso a paso, el estimador por mínimos cuadrados, incluyendo las condiciones necesarias para la existencia y unicidad de la solución.*

#### A. Notación y Tabla Dimensional
| Objeto | Dimensión | Rango | Descripción |
| :--- | :---: | :---: | :--- |
| $N, P, Q$ | Escalares | $n \in \{1,\dots,N\}$, $j \in \{1,\dots,Q\}$ | Muestras, dimensión original y dimensión mapeada ($Q > P$). |
| $x_n$ | $\mathbb{R}^{P \times 1}$ | Entrada original de la muestra $n$. | |
| $\phi(x_n)$ | $\mathbb{R}^{Q \times 1}$ | Vector de características no lineales evaluado en $x_n$. | |
| $\mathbf{\Phi}$ | $\mathbb{R}^{N \times Q}$ | Fila $n$ es $\phi(x_n)^\top$. Matriz de diseño completa. | |
| $\mathbf{t}$ | $\mathbb{R}^{N \times 1}$ | $[t_1, \dots, t_N]^\top$. Vector de mediciones observadas. | |
| $\mathbf{w}$ | $\mathbb{R}^{Q \times 1}$ | $[w_1, \dots, w_Q]^\top$. Vector de pesos a estimar. | |
| $\mathbf{e}$ | $\mathbb{R}^{N \times 1}$ | $\mathbf{t} - \mathbf{\Phi}\mathbf{w}$. Vector de residuos o errores empíricos. | |

#### B. Planteamiento de la Función Objetivo
El error de cada muestra es $e_n = t_n - \phi(x_n)^\top \mathbf{w}$. La suma de errores al cuadrado matricial es:
$$J(\mathbf{w}) = \sum_{n=1}^N e_n^2 = \mathbf{e}^\top \mathbf{e} = \|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|_2^2 = (\mathbf{t} - \mathbf{\Phi}\mathbf{w})^\top (\mathbf{t} - \mathbf{\Phi}\mathbf{w})$$

Expandiendo el producto:
$$J(\mathbf{w}) = \mathbf{t}^\top \mathbf{t} - \mathbf{t}^\top \mathbf{\Phi}\mathbf{w} - \mathbf{w}^\top \mathbf{\Phi}^\top \mathbf{t} + \mathbf{w}^\top \mathbf{\Phi}^\top \mathbf{\Phi}\mathbf{w}$$
Como $\mathbf{t}^\top \mathbf{\Phi}\mathbf{w}$ es un escalar ($1 \times 1$), es idéntico a su transpuesta $\mathbf{w}^\top \mathbf{\Phi}^\top \mathbf{t}$:
$$J(\mathbf{w}) = \mathbf{t}^\top \mathbf{t} - 2\mathbf{w}^\top \mathbf{\Phi}^\top \mathbf{t} + \mathbf{w}^\top \mathbf{\Phi}^\top \mathbf{\Phi}\mathbf{w}$$

#### C. Deducción Analítica del Estimador
Derivamos respecto a $\mathbf{w} \in \mathbb{R}^{Q \times 1}$:
$$\nabla_{\mathbf{w}} J(\mathbf{w}) = \mathbf{0} - 2\mathbf{\Phi}^\top \mathbf{t} + 2\mathbf{\Phi}^\top \mathbf{\Phi}\mathbf{w} = -2\mathbf{\Phi}^\top \mathbf{t} + 2\mathbf{\Phi}^\top \mathbf{\Phi}\mathbf{w}$$

* **Convexidad:** El Hessiano $\nabla^2 J = 2\mathbf{\Phi}^\top \mathbf{\Phi} \succeq 0$ es semidefinido positivo, garantizando que el punto estacionario es mínimo global.
* **Ecuaciones Normales:** Igualando a cero y dividiendo entre 2:
  $$\mathbf{\Phi}^\top \mathbf{\Phi}\mathbf{w} = \mathbf{\Phi}^\top \mathbf{t}$$

#### D. Existencia y Unicidad de la Solución
- **Existencia:** La solución **siempre existe**. Por el Teorema de Rouché-Capelli, $\mathbf{\Phi}^\top \mathbf{t} \in \operatorname{Col}(\mathbf{\Phi}^\top) = \operatorname{Col}(\mathbf{\Phi}^\top \mathbf{\Phi})$, por lo que el sistema nunca es inconsistente.
- **Unicidad:** La solución es **única** si y solo si $\mathbf{\Phi}^\top \mathbf{\Phi}$ es invertible, lo que exige rango de columnas completo:
   $$\operatorname{rango}(\mathbf{\Phi}) = Q \quad (\text{requiere } N \ge Q)$$
   En tal caso: $\mathbf{w}^* = (\mathbf{\Phi}^\top \mathbf{\Phi})^{-1}\mathbf{\Phi}^\top \mathbf{t}$.
- **Caso degenerado ($N < Q$ o multicolinealidad):** Existen infinitas soluciones. La de mínima norma $\|\mathbf{w}\|_2$ se obtiene con la pseudoinversa: $\mathbf{w}^* = \mathbf{\Phi}^+ \mathbf{t}$.

---

### 1.2 Mínimos Cuadrados Regularizados (Ridge Regression)

> **Enunciado Oficial:**  
> *Formule el problema con penalización cuadrática $\lambda \|\mathbf{w}\|_2^2$, con $\lambda > 0$, y derive su solución cerrada. Explique el efecto del parámetro de regularización.*

#### A. Formulación de la Función Objetivo
$$J_{\text{reg}}(\mathbf{w}) = \|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_2^2 = (\mathbf{t} - \mathbf{\Phi}\mathbf{w})^\top (\mathbf{t} - \mathbf{\Phi}\mathbf{w}) + \lambda \mathbf{w}^\top \mathbf{w}$$
$$J_{\text{reg}}(\mathbf{w}) = \mathbf{t}^\top \mathbf{t} - 2\mathbf{w}^\top \mathbf{\Phi}^\top \mathbf{t} + \mathbf{w}^\top \mathbf{\Phi}^\top \mathbf{\Phi}\mathbf{w} + \lambda \mathbf{w}^\top \mathbf{w}$$

#### B. Deducción de la Solución Cerrada
Derivamos respecto a $\mathbf{w} \in \mathbb{R}^{Q \times 1}$:
$$\nabla_{\mathbf{w}} J_{\text{reg}}(\mathbf{w}) = -2\mathbf{\Phi}^\top \mathbf{t} + 2\mathbf{\Phi}^\top \mathbf{\Phi}\mathbf{w} + 2\lambda \mathbf{w} = \mathbf{0}$$

Agrupando con la matriz identidad $\mathbf{I} \in \mathbb{R}^{Q \times Q}$ y dividiendo entre 2:
$$(\mathbf{\Phi}^\top \mathbf{\Phi} + \lambda \mathbf{I})\mathbf{w} = \mathbf{\Phi}^\top \mathbf{t}$$

Multiplicando por la izquierda por la inversa:
$$\mathbf{w}^*_{\text{reg}} = (\mathbf{\Phi}^\top \mathbf{\Phi} + \lambda \mathbf{I})^{-1} \mathbf{\Phi}^\top \mathbf{t}$$

*Chequeo dimensional:* $\underbrace{\mathbf{w}^*}_{Q \times 1} = \underbrace{(\mathbf{\Phi}^\top \mathbf{\Phi} + \lambda \mathbf{I})^{-1}}_{Q \times Q} \underbrace{\mathbf{\Phi}^\top \mathbf{t}}_{Q \times 1}$.

#### C. Efecto del Parámetro de Regularización $\lambda$ (Los 4 Pilares)
- **Espectral / Invertibilidad:** Al sumar $\lambda \mathbf{I}$, los autovalores de la matriz pasan de $\mu_j \ge 0$ a $\mu_j + \lambda > 0$. La matriz es **siempre invertible y bien condicionada**, incluso si $N < Q$.
- **Encogimiento (Weight Shrinkage):** Atenúa los pesos hacia cero con el factor $\frac{\mu_j}{\mu_j + \lambda} < 1$, eliminando oscilaciones espurias del modelo.
- **Compromiso Sesgo-Varianza:** Reduce drásticamente la varianza del estimador a cambio de un sesgo pequeño y controlado.
- **Límites asintóticos:**
   * $\lim_{\lambda \to 0^+} \mathbf{w}^* = (\mathbf{\Phi}^\top \mathbf{\Phi})^{-1}\mathbf{\Phi}^\top \mathbf{t} = \mathbf{w}^*_{\text{OLS}}$.
   * $\lim_{\lambda \to \infty} \mathbf{w}^* = \mathbf{0}$ (modelo plano).

---

### 1.3 Máxima Verosimilitud (MLE)

> **Enunciado Oficial:**  
> *Bajo el supuesto de ruido gaussiano i.i.d., construya la función de verosimilitud y su logaritmo, y obtenga el estimador de $\mathbf{w}$ por máxima verosimilitud. Explicite su relación con mínimos cuadrados.*

**Densidad individual y verosimilitud conjunta:**
Cada residuo es $\eta_n = t_n - \phi(x_n)^\top \mathbf{w} \sim \mathcal{N}(0, \sigma_\eta^2)$. La probabilidad de una muestra es:
$$p(t_n \mid x_n, \mathbf{w}, \sigma_\eta^2) = \frac{1}{(2\pi\sigma_\eta^2)^{1/2}} \exp\left( -\frac{(t_n - \phi(x_n)^\top \mathbf{w})^2}{2\sigma_\eta^2} \right)$$

Por independencia (i.i.d.), la verosimilitud conjunta del dataset $\mathbf{t} \in \mathbb{R}^{N \times 1}$ es el **producto** de las densidades:
$$L(\mathbf{w}) = p(\mathbf{t} \mid \mathbf{\Phi}, \mathbf{w}, \sigma_\eta^2) = \prod_{n=1}^N p(t_n \mid x_n, \mathbf{w}, \sigma_\eta^2)$$
$$L(\mathbf{w}) = (2\pi\sigma_\eta^2)^{-N/2} \exp\left( -\frac{1}{2\sigma_\eta^2} \sum_{n=1}^N (t_n - \phi(x_n)^\top \mathbf{w})^2 \right) = (2\pi\sigma_\eta^2)^{-N/2} \exp\left( -\frac{1}{2\sigma_\eta^2} \|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|_2^2 \right)$$

**Construcción del logaritmo ($\ln L$) y estimador $\mathbf{w}^*_{\text{MLE}}$:**
Aplicando logaritmo natural ($\ln$):
$$\ln L(\mathbf{w}) = -\frac{N}{2}\ln(2\pi\sigma_\eta^2) - \frac{1}{2\sigma_\eta^2} \|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|_2^2$$

Derivando respecto a $\mathbf{w} \in \mathbb{R}^{Q \times 1}$ e igualando a cero:
$$\nabla_{\mathbf{w}} \ln L(\mathbf{w}) = \frac{1}{\sigma_\eta^2} (\mathbf{\Phi}^\top \mathbf{t} - \mathbf{\Phi}^\top \mathbf{\Phi}\mathbf{w}) = \mathbf{0}$$
Multiplicando por $\sigma_\eta^2 > 0$:
$$\mathbf{\Phi}^\top \mathbf{\Phi}\mathbf{w} = \mathbf{\Phi}^\top \mathbf{t} \implies \mathbf{w}^*_{\text{MLE}} = (\mathbf{\Phi}^\top \mathbf{\Phi})^{-1}\mathbf{\Phi}^\top \mathbf{t}$$

**Relación explícita con mínimos cuadrados:**
* **Identidad:** $\mathbf{w}^*_{\text{MLE}} = \mathbf{w}^*_{\text{OLS}}$.
* **Fundamento:** Mínimos cuadrados es la consecuencia analítica exacta de maximizar la verosimilitud bajo ruido gaussiano i.i.d.
* **Invarianza:** La varianza del ruido $\sigma_\eta^2$ se cancela al derivar y no afecta el óptimo $\mathbf{w}^*$.

---

### 1.4 Máximo a Posteriori (MAP)

> **Enunciado Oficial:**  
> *Suponga además el prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \sigma_w^2 \mathbf{I}_Q)$. A partir de la regla de Bayes, deduzca el problema de optimización MAP y su solución; establezca la relación entre $\sigma_\eta^2$, $\sigma_w^2$ y la regularización cuadrática.*

**Distribución a priori (Prior):**  
El vector de parámetros sigue una distribución normal multivariada centrada en cero:
$$p(\mathbf{w}) = (2\pi \sigma_w^2)^{-Q/2} \exp\left( -\frac{1}{2\sigma_w^2} \|\mathbf{w}\|_2^2 \right)$$

**Regla de Bayes y deducción del problema MAP:**  
Aplicando el teorema de Bayes para la distribución a posteriori de los parámetros dado el vector de mediciones $\mathbf{t}$:
$$p(\mathbf{w} \mid \mathbf{t}) = \frac{p(\mathbf{t} \mid \mathbf{w}) \, p(\mathbf{w})}{p(\mathbf{t})}$$
donde la evidencia marginal $p(\mathbf{t}) = \int p(\mathbf{t} \mid \mathbf{w}) p(\mathbf{w}) \, d\mathbf{w}$ es un escalar positivo constante que no depende del vector de parámetros $\mathbf{w}$.

Por definición, el estimador MAP es el punto que maximiza la función a posteriori:
$$\mathbf{w}^*_{\text{MAP}} = \arg\max_{\mathbf{w}} p(\mathbf{w} \mid \mathbf{t}) = \arg\max_{\mathbf{w}} \left[ \frac{p(\mathbf{t} \mid \mathbf{w}) \, p(\mathbf{w})}{p(\mathbf{t})} \right]$$
Dado que dividir entre una constante positiva fija $p(\mathbf{t})$ no modifica la coordenada donde se produce el valor máximo:
$$\mathbf{w}^*_{\text{MAP}} = \arg\max_{\mathbf{w}} [ p(\mathbf{t} \mid \mathbf{w}) \, p(\mathbf{w}) ]$$

Tomando el logaritmo natural:
$$\ln p(\mathbf{w} \mid \mathbf{t}) = \ln p(\mathbf{t} \mid \mathbf{w}) + \ln p(\mathbf{w}) - \ln p(\mathbf{t})$$
Como $-\ln p(\mathbf{t})$ es constante respecto a $\mathbf{w}$, al derivar respecto a $\mathbf{w}$ se anula idénticamente ($\nabla_{\mathbf{w}} [-\ln p(\mathbf{t})] = \mathbf{0}$). Sustituyendo las expresiones gaussianas de verosimilitud y prior:
$$\ln p(\mathbf{w} \mid \mathbf{t}) = -\frac{N}{2}\ln(2\pi\sigma_\eta^2) - \frac{1}{2\sigma_\eta^2} \|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|_2^2 - \frac{Q}{2}\ln(2\pi\sigma_w^2) - \frac{1}{2\sigma_w^2} \|\mathbf{w}\|_2^2 - \ln p(\mathbf{t})$$

Agrupando todos los términos constantes independientes de $\mathbf{w}$ en una constante $C_0$:
$$\ln p(\mathbf{w} \mid \mathbf{t}) = -\frac{1}{2\sigma_\eta^2} \|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|_2^2 - \frac{1}{2\sigma_w^2} \|\mathbf{w}\|_2^2 + C_0$$

Descartando los términos constantes independientes de $\mathbf{w}$:
$$\arg\max_{\mathbf{w}} \ln p(\mathbf{w} \mid \mathbf{t}) = \arg\max_{\mathbf{w}} \left[ -\frac{1}{2\sigma_\eta^2} \|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|_2^2 - \frac{1}{2\sigma_w^2} \|\mathbf{w}\|_2^2 \right]$$

Multiplicando por $-\sigma_\eta^2$ (lo que invierte la maximización a minimización):
$$\mathbf{w}^*_{\text{MAP}} = \arg\min_{\mathbf{w}} \left[ \|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|_2^2 + \frac{\sigma_\eta^2}{\sigma_w^2} \|\mathbf{w}\|_2^2 \right]$$

**Deducción de la solución óptima $\mathbf{w}^*_{\text{MAP}}$:**  
A partir de la función de log-densidad a posteriori:
$$\ln p(\mathbf{w} \mid \mathbf{t}) = -\frac{1}{2\sigma_\eta^2} \|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|_2^2 - \frac{1}{2\sigma_w^2} \|\mathbf{w}\|_2^2 + C_0$$

Derivamos respecto a $\mathbf{w} \in \mathbb{R}^{Q \times 1}$:
$$\nabla_{\mathbf{w}} \ln p(\mathbf{w} \mid \mathbf{t}) = \frac{1}{\sigma_\eta^2} (\mathbf{\Phi}^\top \mathbf{t} - \mathbf{\Phi}^\top \mathbf{\Phi}\mathbf{w}) - \frac{1}{\sigma_w^2} \mathbf{w}$$

Imponemos la condición de primer orden igualando al vector cero ($\nabla_{\mathbf{w}} \ln p(\mathbf{w} \mid \mathbf{t}) = \mathbf{0}$):
$$\frac{1}{\sigma_\eta^2} \mathbf{\Phi}^\top \mathbf{t} - \frac{1}{\sigma_\eta^2} \mathbf{\Phi}^\top \mathbf{\Phi}\mathbf{w} - \frac{1}{\sigma_w^2} \mathbf{w} = \mathbf{0}$$

**¿De dónde sale $\lambda$? (Origen algebraico):**  
Para aislar $\mathbf{w}$ y normalizar el término de los datos (dejando a $\mathbf{\Phi}^\top \mathbf{\Phi}$ con coeficiente unitario, idéntico a las ecuaciones normales), multiplicamos toda la ecuación por la varianza del ruido $\sigma_\eta^2 > 0$:
$$\mathbf{\Phi}^\top \mathbf{t} - \mathbf{\Phi}^\top \mathbf{\Phi}\mathbf{w} - \left(\frac{\sigma_\eta^2}{\sigma_w^2}\right) \mathbf{w} = \mathbf{0}$$

Agrupando los términos que acompañan a $\mathbf{w}$:
$$\left( \mathbf{\Phi}^\top \mathbf{\Phi} + \left(\frac{\sigma_\eta^2}{\sigma_w^2}\right) \mathbf{I}_Q \right) \mathbf{w} = \mathbf{\Phi}^\top \mathbf{t}$$

Sustituyendo el cociente de varianzas por el parámetro de regularización cuadrática:
$$\lambda = \frac{\sigma_\eta^2}{\sigma_w^2}$$
la ecuación queda exactamente:
$$(\mathbf{\Phi}^\top \mathbf{\Phi} + \lambda \mathbf{I}_Q) \mathbf{w} = \mathbf{\Phi}^\top \mathbf{t} \implies \mathbf{w}^*_{\text{MAP}} = (\mathbf{\Phi}^\top \mathbf{\Phi} + \lambda \mathbf{I}_Q)^{-1} \mathbf{\Phi}^\top \mathbf{t}$$

**¿Por qué aparece este cociente? (Interpretación y significado físico):**
* **Balance de incertidumbres:** $\lambda$ es la razón entre dos fuentes de incertidumbre física:
  $$\lambda = \frac{\text{Varianza del ruido en las mediciones } (\sigma_\eta^2)}{\text{Varianza de nuestra creencia previa sobre los parámetros } (\sigma_w^2)}$$
* **Si las mediciones son muy ruidosas ($\sigma_\eta^2$ grande):** Confiamos poco en los datos. El cociente $\lambda$ crece, aumentando la penalización sobre $\mathbf{w}$ para evitar que el modelo sobreajuste al ruido (*shrinkage* agresivo hacia el cero del prior).
* **Si el prior es difuso o no informativo ($\sigma_w^2 \to \infty$):** No tenemos conocimiento previo sobre la magnitud de los parámetros. La dispersión del prior tiende a infinito, lo que anula el cociente:
  $$\lim_{\sigma_w^2 \to \infty} \lambda = \lim_{\sigma_w^2 \to \infty} \frac{\sigma_\eta^2}{\sigma_w^2} = 0 \implies \mathbf{w}^*_{\text{MAP}} = (\mathbf{\Phi}^\top \mathbf{\Phi})^{-1}\mathbf{\Phi}^\top \mathbf{t} = \mathbf{w}^*_{\text{OLS}} = \mathbf{w}^*_{\text{MLE}}$$
* **Si el sensor no tiene ruido ($\sigma_\eta^2 \to 0$):** Los datos observados son de precisión infinita. El cociente se anula ($\lambda \to 0$), por lo que confiamos plenamente en las observaciones y se recupera la solución ordinaria OLS.
* **Conclusión obligatoria del parcial:** La regularización de Ridge no es una técnica heurística arbitraria; es la consecuencia matemática exacta de aplicar inferencia Bayesiana con un prior gaussiano centrado en cero, donde la constante de penalización $\lambda$ surge naturalmente como el cociente $\sigma_\eta^2 / \sigma_w^2$.

---

### 1.5 Implementación y Análisis ($N=300, P=5, Q=20$)
*(En desarrollo: simulación numérica y comparativa de los 4 estimadores en Python)*

---

## 2. Espacios RKHS, Embeddings de Media y MMD

### 2.1 Definición Rigurosa del Embedding de Media ($\mu_k$)

> **Enunciado Oficial:**  
> *Defina rigurosamente el embedding de media asociado a una variable aleatoria $X$:*  
> $$\mu_k = \mathbb{E}[k(X, \cdot)]$$

Sea $X \sim P_X$ una variable aleatoria sobre $\mathcal{X}$ y $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ un kernel simétrico definido positivo con RKHS asociado $\mathcal{H}_k$.

1. **Definición Formal:**
   $$\mu_k = \mathbb{E}_{X \sim P_X}[k(X, \cdot)] = \int_{\mathcal{X}} k(x, \cdot) \, dP_X(x) \in \mathcal{H}_k$$
   * La integral se define en el sentido de **Bochner** en el espacio de Hilbert $\mathcal{H}_k$.
   * El punto $(\cdot)$ es el argumento libre de evaluación ($\phi(x) = k(x, \cdot)$ es una función completa en $\mathcal{H}_k$, no un escalar). Por tanto, $\mu_k$ es un elemento funcional de $\mathcal{H}_k$.

2. **Condición de Existencia (Integrabilidad de Bochner):**
   Para asegurar que la integral converja y $\mu_k \in \mathcal{H}_k$ tenga norma finita:
   $$\mathbb{E}_{X \sim P_X}\left[ \|k(X, \cdot)\|_{\mathcal{H}_k} \right] = \mathbb{E}_{X \sim P_X}\left[ \sqrt{k(X, X)} \right] < \infty$$
   *(Para kernels acotados como el Gaussiano/RBF donde $k(x, x) = 1$, se cumple trivialmente $\mathbb{E}[\sqrt{1}] = 1 < \infty$).*

3. **Evaluación Puntual y Propiedad Reproductora:**
   * Evaluación en $x' \in \mathcal{X}$: $\mu_k(x') = \mathbb{E}_{X \sim P_X}[k(X, x')] = \int_{\mathcal{X}} k(x, x') \, dP_X(x)$.
   * Para cualquier $f \in \mathcal{H}_k$: $\langle f, \mu_k \rangle_{\mathcal{H}_k} = \mathbb{E}_{X \sim P_X}[f(X)]$.

---

### 2.2 Pertenencia a $\mathcal{H}_k$ e Interpretación Geométrica

> **Enunciado Oficial:**  
> *Explique por qué $\mu_k \in \mathcal{H}_k$ e interprete geométricamente su significado como representante de la distribución de $X$ en el espacio RKHS.*

1. **¿Por qué $\mu_k \in \mathcal{H}_k$? (Demostración de pertenencia por norma finita):**
   Un elemento pertenece a $\mathcal{H}_k$ si y solo si $\|\mu_k\|_{\mathcal{H}_k} < \infty$. Introduciendo dos copias independientes $X, X' \sim P_X$:
   $$\|\mu_k\|_{\mathcal{H}_k}^2 = \langle \mu_k, \mu_k \rangle_{\mathcal{H}_k} = \left\langle \mathbb{E}_X[k(X, \cdot)], \, \mathbb{E}_{X'}[k(X', \cdot)] \right\rangle_{\mathcal{H}_k} = \mathbb{E}_{X, X'}\left[ \langle k(X, \cdot), k(X', \cdot) \rangle_{\mathcal{H}_k} \right] = \mathbb{E}_{X, X'}[k(X, X')]$$
   Por la desigualdad de Cauchy-Schwarz aplicada al kernel ($|k(X, X')| \le \sqrt{k(X, X)}\sqrt{k(X', X')}$):
   $$\|\mu_k\|_{\mathcal{H}_k}^2 \le \mathbb{E}_{X, X'}\left[ \sqrt{k(X, X)}\sqrt{k(X', X')} \right] = \left( \mathbb{E}_{X \sim P_X}\left[ \sqrt{k(X, X)} \right] \right)^2 < \infty \implies \mu_k \in \mathcal{H}_k \quad \blacksquare$$
   *(Alternativamente, por el Teorema de Riesz, el funcional $T(f) = \mathbb{E}[f(X)]$ es lineal y continuo, garantizando la existencia y unicidad de $\mu_k \in \mathcal{H}_k$).*

2. **Interpretación Geométrica en el RKHS:**
   * **Baricentro (Centro de masa funcional):** De forma análoga a la media clásica $\mathbb{E}[X]$ en $\mathbb{R}^P$, $\mu_k$ es el centro de masa de la distribución de probabilidad $P_X$ en el espacio de funciones $\mathcal{H}_k$.
   * **Representante Único e Inyectividad:** Para kernels característicos (como el Gaussiano), el mapeo $P \mapsto \mu_k$ es estrictamente inyectivo ($\mu_P = \mu_Q \iff P = Q$). Un único punto $\mu_k \in \mathcal{H}_k$ codifica todos los momentos estadísticos de orden infinito de la distribución sin pérdida de información.

---

*(Puntos 2.3 y 2.4 en desarrollo)*

---

## 3. Índice de Correlación de Pearson
*(Puntos 3.1 y 3.2 en desarrollo)*
