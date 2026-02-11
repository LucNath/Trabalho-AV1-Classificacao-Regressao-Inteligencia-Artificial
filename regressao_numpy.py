# regressao_numpy.py
# Trabalho AV1 - Parte Regressão
# Dataset: aerogerador.dat
# Modelos: Média, MQO (OLS), Ridge (λ = 0, 0.25, 0.5, 0.75, 1.0)
# Métrica: RSS | Validação Monte Carlo R = 500

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def add_intercept(X):
    return np.hstack([np.ones((X.shape[0],1)), X])

def ols_fit(X, y):
    Xb = add_intercept(X)
    beta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
    return beta

def ridge_fit(X, y, lam):
    Xb = add_intercept(X)
    p1 = Xb.shape[1]
    reg = lam * np.eye(p1)
    reg[0,0] = 0.0
    beta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    return beta

def predict(X, beta):
    return add_intercept(X) @ beta

# Carregar dataset
data = np.loadtxt("aerogerador.dat")
X_all = data[:,0].reshape(-1,1)
y_all = data[:,1].reshape(-1,1)
N = X_all.shape[0]

# Monte Carlo 1
# R = 500
# lambdas = [0,0.25,0.5,0.75,1.0]
# models = ["Mean","OLS"] + [f"Ridge_{lam}" for lam in lambdas]
# rss_results = {m: np.zeros(R) for m in models}

# indices = np.arange(N)
# for r in range(R):
#     np.random.shuffle(indices)
#     cut = int(0.8*N)
#     tr, te = indices[:cut], indices[cut:]
#     Xtr, ytr = X_all[tr], y_all[tr]
#     Xte, yte = X_all[te], y_all[te]

#     mu = np.mean(ytr)
#     rss_results["Mean"][r] = np.sum((yte - mu)**2)

#     beta = ols_fit(Xtr, ytr)
#     rss_results["OLS"][r] = np.sum((yte - predict(Xte, beta))**2)

#     for lam in lambdas:
#         beta_r = ridge_fit(Xtr, ytr, lam)
#         rss_results[f"Ridge_{lam}"][r] = np.sum((yte - predict(Xte, beta_r))**2)

#Monte Carlo 2

R = 500
lambdas = [0,0.25,0.5,0.75,1.0]
models = ["Mean","OLS"] + [f"Ridge_{lam}" for lam in lambdas]
rss_results = {m: np.zeros(R) for m in models}

for r in range(R):
    mask = np.random.rand(N) < 0.8   # sorteio aleatório
    Xtr, ytr = X_all[mask], y_all[mask]
    Xte, yte = X_all[~mask], y_all[~mask]

    # Modelo da média
    mu = np.mean(ytr)
    rss_results["Mean"][r] = np.sum((yte - mu)**2)

    # MQO (OLS)
    beta = ols_fit(Xtr, ytr)
    rss_results["OLS"][r] = np.sum((yte - predict(Xte, beta))**2)

    # Ridge
    for lam in lambdas:
        beta_r = ridge_fit(Xtr, ytr, lam)
        rss_results[f"Ridge_{lam}"][r] = np.sum((yte - predict(Xte, beta_r))**2)


# Estatísticas
print("Resultados - Regressão (RSS)")
for m in models:
    arr = rss_results[m]
    print(f"{m:12s} | Média={np.mean(arr):.2f} | Desvio={np.std(arr):.2f} | Max={np.max(arr):.2f} | Min={np.min(arr):.2f}")

# Boxplot
plt.boxplot([rss_results[m] for m in models], labels=models, showfliers=False)
plt.ylabel("RSS")
plt.title("Comparação de Modelos - Regressão")
plt.xticks(rotation=30)
plt.grid(True)
plt.show()
