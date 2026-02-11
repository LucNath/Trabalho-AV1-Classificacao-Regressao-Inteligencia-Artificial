# classificacao_numpy.py
# Trabalho AV1 - Parte Classificação
# Dataset: EMGsDataset.csv
# Modelos: MQO, Gauss tradicional, Gauss pooled, Naive Bayes, Gauss regularizado
# Métrica: Acurácia | Validação Monte Carlo R = 500

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def add_intercept(X):
    return np.hstack([np.ones((X.shape[0],1)), X])

def logpdf_gauss(X, mu, Sigma):
    p = X.shape[1]
    Sigma = Sigma + 1e-8*np.eye(p)
    sign, logdet = np.linalg.slogdet(Sigma)
    inv = np.linalg.inv(Sigma)
    dev = X - mu
    q = -0.5*np.sum((dev @ inv)*dev, axis=1)
    const = -0.5*p*np.log(2*np.pi) - 0.5*logdet
    return const + q

# Carregar dataset
raw = np.loadtxt("EMGsDataset.csv", delimiter=",")
sensor1, sensor2, labels = raw[0,:], raw[1,:], raw[2,:].astype(int)
X_all = np.column_stack([sensor1, sensor2])
N, p = X_all.shape
C = labels.max()

# Monte Carlo
R = 500
lambdas = [0,0.25,0.5,0.75,1.0]
models = ["MQO","Gauss_traditional","Gauss_pooled","NaiveBayes"] + [f"Gauss_reg_{lam}" for lam in lambdas]
acc_results = {m: np.zeros(R) for m in models}

for r in range(R):
    mask_tr = np.random.rand(N) < 0.8
    Xtr, Xte = X_all[mask_tr], X_all[~mask_tr]
    ytr, yte = labels[mask_tr], labels[~mask_tr]

    # MQO
    Ytr = np.zeros((Xtr.shape[0], C))
    Ytr[np.arange(Xtr.shape[0]), ytr-1] = 1
    B = np.linalg.solve(add_intercept(Xtr).T @ add_intercept(Xtr), add_intercept(Xtr).T @ Ytr)
    preds = np.argmax(add_intercept(Xte) @ B, axis=1) + 1
    acc_results["MQO"][r] = np.mean(preds == yte)

    pis = np.zeros(C); mus = np.zeros((C,p)); Sigmas = np.zeros((C,p,p))
    for k in range(1,C+1):
        Xk = Xtr[ytr==k]
        Nk = len(Xk)
        pis[k-1] = Nk / len(Xtr)
        if Nk > 0:
            mus[k-1] = np.mean(Xk, axis=0)
            dev = Xk - mus[k-1]
            Sigmas[k-1] = (dev.T @ dev)/Nk
        else:
            mus[k-1] = np.zeros(p)
            Sigmas[k-1] = np.eye(p)
    Sigma_pooled = ((Xtr - Xtr.mean(0)).T @ (Xtr - Xtr.mean(0)))/len(Xtr)

    for m in models:
        if m=="MQO": continue
        logp = np.zeros((len(Xte), C))
        for k in range(C):
            mu, prior = mus[k], pis[k]
            if m=="Gauss_traditional":
                Sigma = Sigmas[k]
            elif m=="Gauss_pooled":
                Sigma = Sigma_pooled
            elif m=="NaiveBayes":
                Sigma = np.diag(np.diag(Sigmas[k]))
            elif m.startswith("Gauss_reg"):
                lam = float(m.split("_")[2])
                Sigma = (1-lam)*Sigmas[k] + lam*Sigma_pooled
            logp[:,k] = logpdf_gauss(Xte, mu, Sigma) + np.log(prior+1e-12)
        preds = np.argmax(logp, axis=1)+1
        acc_results[m][r] = np.mean(preds == yte)

# Estatísticas
print("Resultados - Classificação (Acurácia)")
for m in models:
    arr = acc_results[m]
    print(f"{m:16s} | Média={np.mean(arr):.3f} | Desvio={np.std(arr):.3f} | Max={np.max(arr):.3f} | Min={np.min(arr):.3f}")

# Boxplot
plt.boxplot([acc_results[m] for m in models], labels=models, showfliers=False)
plt.ylabel("Acurácia")
plt.title("Comparação de Modelos - Classificação")
plt.xticks(rotation=30)
plt.grid(True)
plt.show()
