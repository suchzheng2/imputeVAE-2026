"""Depth-robust reference classifier for assigning training cell types to external nuclei (an alternative to the
latent kNN vote, which degrades on shallow / different-chemistry input). log1p(CPM) over the genes present in the
target dataset -> top-N variable genes (training) -> z-score with training statistics -> PCA -> multinomial
logistic regression, all fitted on the v3 training nuclei. Returns predictions, confidence, and (optionally) the
accuracy on the v3 validation nuclei as a sanity number."""
import time
import numpy as np


def cpm_log1p(X_log1p, present, scale=1e4):
    C = np.expm1(np.asarray(X_log1p, dtype=np.float32)[:, present])
    lib = C.sum(1, keepdims=True); lib[lib == 0] = 1.0
    return np.log1p(C / lib * scale)


def pca_lr_predict(Xn_tr, ct_tr, X_target, present, n_hvg=3000, n_pcs=50, X_valid=None, ct_valid=None, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    A = cpm_log1p(Xn_tr, present)
    var = A.var(0); hvg = np.argsort(-var)[:min(n_hvg, A.shape[1])]
    A = A[:, hvg]; mu, sd = A.mean(0), A.std(0) + 1e-3
    z = lambda M: np.clip((M - mu) / sd, -10, 10)
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=seed).fit(z(A))
    clf = LogisticRegression(max_iter=1000, C=0.5, random_state=seed).fit(pca.transform(z(A)), np.asarray(ct_tr).astype(str))
    def predict(M):
        P = clf.predict_proba(pca.transform(z(cpm_log1p(M, present)[:, hvg])))
        return clf.classes_[P.argmax(1)], P.max(1)
    pred, conf = predict(X_target)
    info = {"type_predictor": "pca_lr", "n_hvg": int(len(hvg)), "n_pcs": n_pcs, "uses_cell_type_label": False,
            "labels_used": "training nuclei only (log1p-CPM / HVG / PCA / logistic regression)",
            "confidence_mean": float(conf.mean()), "train_accuracy": float((predict(Xn_tr)[0] == np.asarray(ct_tr).astype(str)).mean())}
    if X_valid is not None:
        info["valid_accuracy"] = float((predict(X_valid)[0] == np.asarray(ct_valid).astype(str)).mean())
    info["fit_time_s"] = round(time.time() - t0, 1)
    return pred, info
