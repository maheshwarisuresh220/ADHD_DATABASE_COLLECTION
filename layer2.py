"""
=============================================================================
ADFI Framework — Layer 2: Hybrid Optimization & Ensemble Processing
=============================================================================
Maps to Chapter 3.4 of the ADFI thesis 
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

from sklearn.linear_model    import LogisticRegression
from sklearn.neural_network  import MLPClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import (LeaveOneOut, StratifiedKFold,
                                     cross_val_score, cross_val_predict)
from sklearn.metrics         import (log_loss, accuracy_score,
                                     classification_report, roc_auc_score, f1_score)


# =============================================================================
# Fitness system — LR surrogate + LOO-CV + Gentle sparsity penalty
# =============================================================================

class FitnessCache:
    def __init__(self):
        self._d: Dict[tuple, float] = {}
        self.hits = self.misses = 0

    def get(self, pos: np.ndarray):
        key = tuple((pos > 0.5).astype(int))
        if key in self._d:
            self.hits += 1
            return self._d[key]
        return None

    def put(self, pos: np.ndarray, val: float):
        key = tuple((pos > 0.5).astype(int))
        self._d[key] = val
        self.misses += 1

    @property
    def rate(self):
        t = self.hits + self.misses
        return self.hits / t if t else 0.0


def _fitness(X: np.ndarray, y: np.ndarray,
             position: np.ndarray,
             cache: FitnessCache,
             sparsity_lambda: float = 0.05) -> float: 
    selected = np.where(position > 0.5)[0]
    if len(selected) == 0:
        return 0.0

    cached = cache.get(position)
    if cached is not None:
        return cached

    n_total    = len(position)
    n_selected = len(selected)

    clf  = LogisticRegression(C=0.5, max_iter=300, solver="lbfgs", random_state=42)
    loo  = LeaveOneOut()
    
    scores = cross_val_score(clf, X[:, selected], y, cv=loo, scoring="accuracy", n_jobs=-1)
    loo_acc = float(scores.mean())

    sparsity_penalty = sparsity_lambda * (n_selected / n_total)
    result = loo_acc - sparsity_penalty

    cache.put(position, result)
    return result


def _train_production_clf(X_train: np.ndarray, y_train: np.ndarray, mask: np.ndarray):
    selected = np.where(mask > 0.5)[0]
    if len(selected) == 0:
        selected = np.arange(X_train.shape[1])

    clf = LogisticRegression(C=0.5, max_iter=500, solver="lbfgs", random_state=42)
    clf.fit(X_train[:, selected], y_train)
    return clf, selected


# =============================================================================
# Expert #1 — Firefly Algorithm  (§3.4.1)
# =============================================================================

@dataclass
class FireflyConfig:
    n_fireflies:  int   = 20
    max_iter:     int   = 50
    beta0:        float = 1.0
    gamma:        float = 0.5    
    alpha:        float = 0.20
    alpha_decay:  float = 0.97   
    patience:     int   = 12
    random_state: int   = 42

class FireflyOptimizer:
    def __init__(self, n_features: int, cfg: FireflyConfig = None):
        self.n_features      = n_features
        self.cfg             = cfg or FireflyConfig()
        self.best_mask_:     np.ndarray  = np.array([])
        self.best_fitness_:  float       = 0.0
        self.cache_          = FitnessCache()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FireflyOptimizer":
        rng = np.random.default_rng(self.cfg.random_state)
        n, cfg = self.n_features, self.cfg

        positions = rng.uniform(0.0, 1.0, (cfg.n_fireflies, n))
        fitness   = np.array([_fitness(X, y, p, self.cache_) for p in positions])

        best_ever  = float(fitness.max())
        no_improve = 0
        alpha      = cfg.alpha

        print(f"  [FA] Start | best = {best_ever:.4f} | n={n} | LOO-CV + sparsity penalty")

        for t in range(cfg.max_iter):
            best_idx  = int(np.argmax(fitness))
            best_pos  = positions[best_idx]
            best_fit  = float(fitness[best_idx])

            diff_to_best = best_pos - positions          
            r_to_best    = np.linalg.norm(diff_to_best, axis=1)  
            beta         = cfg.beta0 * np.exp(-cfg.gamma * r_to_best ** 2)  

            move_mask    = (fitness < best_fit).astype(float)[:, np.newaxis]
            attraction   = beta[:, np.newaxis] * diff_to_best * move_mask
            eps          = rng.standard_normal((cfg.n_fireflies, n))
            positions    = np.clip(positions + attraction + alpha * eps, 0.0, 1.0)

            fitness = np.array([_fitness(X, y, p, self.cache_) for p in positions])

            current_best = float(fitness.max())

            if current_best > best_ever + 1e-5:
                best_ever  = current_best; no_improve = 0
            else:
                no_improve += 1

            alpha = max(alpha * cfg.alpha_decay, 0.01)

            if (t + 1) % 10 == 0 or t == cfg.max_iter - 1:
                bi    = int(np.argmax(fitness))
                n_sel = int((positions[bi] > 0.5).sum())
                print(f"  [FA] Iter {t+1:3d}/{cfg.max_iter} | best = {current_best:.4f} | sel = {n_sel}/{n} | cache = {self.cache_.rate:.0%}")

            if no_improve >= cfg.patience:
                print(f"  [FA] Early stop iter {t+1}")
                break

        best_idx           = int(np.argmax(fitness))
        self.best_mask_    = (positions[best_idx] > 0.5).astype(int)
        self.best_fitness_ = float(fitness[best_idx])
        n_sel = int(self.best_mask_.sum())
        print(f"  [FA] Done | fitness = {self.best_fitness_:.4f} | selected = {n_sel}/{n}\n")
        return self

    @property
    def best_continuous_position_(self) -> np.ndarray:
        return self.best_mask_.astype(float)


# =============================================================================
# Expert #2 — Particle Swarm Optimization  (§3.4.2)
# =============================================================================

@dataclass
class PSOConfig:
    n_particles:  int   = 20
    max_iter:     int   = 50
    w:            float = 0.70
    c1:           float = 1.50
    c2:           float = 1.50
    patience:     int   = 12
    random_state: int   = 42

class PSOOptimizer:
    def __init__(self, n_features: int, cfg: PSOConfig = None,
                 gbest_seed: np.ndarray = None, shared_cache: FitnessCache = None):
        self.n_features  = n_features
        self.cfg         = cfg or PSOConfig()
        self.gbest_seed  = gbest_seed
        self.cache_      = shared_cache or FitnessCache()
        self.best_mask_: np.ndarray = np.array([])
        self.best_fitness_: float   = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PSOOptimizer":
        rng = np.random.default_rng(self.cfg.random_state)
        n, cfg = self.n_features, self.cfg

        positions  = rng.uniform(0.0, 1.0, (cfg.n_particles, n))
        velocities = rng.uniform(-0.5, 0.5, (cfg.n_particles, n))

        if self.gbest_seed is not None:
            positions[0] = np.clip(self.gbest_seed, 0.0, 1.0)

        fitness   = np.array([_fitness(X, y, p, self.cache_) for p in positions])
        pbest     = positions.copy()
        pbest_f   = fitness.copy()
        gbest_idx = int(np.argmax(pbest_f))
        gbest     = pbest[gbest_idx].copy()
        gbest_f   = float(pbest_f[gbest_idx])
        best_ever = gbest_f
        no_impr   = 0

        print(f"  [PSO] Start | best = {gbest_f:.4f} | n={n}")

        for t in range(cfg.max_iter):
            r1 = rng.uniform(0.0, 1.0, (cfg.n_particles, n))
            r2 = rng.uniform(0.0, 1.0, (cfg.n_particles, n))

            velocities = (cfg.w  * velocities + cfg.c1 * r1 * (pbest  - positions) + cfg.c2 * r2 * (gbest  - positions))
            positions  = np.clip(positions + velocities, 0.0, 1.0)

            fitness = np.array([_fitness(X, y, p, self.cache_) for p in positions])

            improved       = fitness > pbest_f
            pbest[improved] = positions[improved]
            pbest_f[improved] = fitness[improved]

            bi = int(np.argmax(pbest_f))
            if pbest_f[bi] > gbest_f:
                gbest, gbest_f = pbest[bi].copy(), float(pbest_f[bi])

            if gbest_f > best_ever + 1e-5:
                best_ever = gbest_f;  no_impr = 0
            else:
                no_impr += 1

            if (t + 1) % 10 == 0 or t == cfg.max_iter - 1:
                n_sel = int((gbest > 0.5).sum())
                print(f"  [PSO] Iter {t+1:3d}/{cfg.max_iter} | best = {gbest_f:.4f} | sel = {n_sel}/{n} | cache = {self.cache_.rate:.0%}")

            if no_impr >= cfg.patience:
                print(f"  [PSO] Early stop iter {t+1}")
                break

        self.best_mask_    = (gbest > 0.5).astype(int)
        self.best_fitness_ = gbest_f
        print(f"  [PSO] Done | fitness = {self.best_fitness_:.4f} | selected = {int(self.best_mask_.sum())}/{n}\n")
        return self


# =============================================================================
# Expert #3 — FA-PSO Hybrid  (§3.4.3)
# =============================================================================

class FAPSOHybridOptimizer:
    def __init__(self, n_features: int, fa_cfg:  FireflyConfig = None, pso_cfg: PSOConfig = None):
        self.n_features     = n_features
        self.fa_cfg         = fa_cfg  or FireflyConfig()
        self.pso_cfg        = pso_cfg or PSOConfig()
        self._cache         = FitnessCache()   
        self.best_mask_:    np.ndarray = np.array([])
        self.best_fitness_: float      = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FAPSOHybridOptimizer":
        print("  [Hybrid] Phase 1 — FA global exploration")
        fa          = FireflyOptimizer(self.n_features, self.fa_cfg)
        fa.cache_   = self._cache
        fa.fit(X, y)

        rng  = np.random.default_rng(0)
        seed = np.clip(fa.best_continuous_position_ + rng.uniform(-0.1, 0.1, self.n_features), 0.0, 1.0)

        print("  [Hybrid] Phase 2 — PSO local exploitation")
        pso = PSOOptimizer(self.n_features, self.pso_cfg, gbest_seed=seed, shared_cache=self._cache)
        pso.fit(X, y)

        self.best_mask_    = pso.best_mask_
        self.best_fitness_ = pso.best_fitness_
        print(f"  [Hybrid] Done | fitness = {self.best_fitness_:.4f}\n")
        return self


# =============================================================================
# Ensemble Voting Module  (§3.4.5) — NOW WITH THRESHOLD OPTIMIZATION
# =============================================================================

class SoftVotingEnsemble:
    def __init__(self):
        self.weights_:    np.ndarray  = np.array([])
        self.val_losses_: List[float] = []
        self.optimal_threshold_: float = 0.5

    def calibrate_weights(self, val_probas: List[np.ndarray], y_val: np.ndarray) -> "SoftVotingEnsemble":
        raw, labels = [], ["FA", "PSO", "Hybrid", "DNN"]
        for k, (p, lbl) in enumerate(zip(val_probas, labels), 1):
            loss = log_loss(y_val, p)
            w    = 1.0 / (loss + 1e-9)
            raw.append(w);  self.val_losses_.append(loss)
            print(f"  Expert {k} ({lbl:<6}) : out-of-fold log_loss = {loss:.4f}  →  raw_weight = {w:.3f}")
        
        total = sum(raw)
        self.weights_ = np.array([w / total for w in raw])
        
        print("\n  Normalised W_k (Eq. 3.6):")
        for lbl, w in zip(labels, self.weights_):
            print(f"    W_{lbl:<6} = {w:.4f}")
            
        # DYNAMIC THRESHOLD OPTIMIZATION
        oof_proba = np.zeros_like(y_val, dtype=float)
        for w, p in zip(self.weights_, val_probas):
            oof_proba += w * p[:, 1]
            
        best_thresh = 0.5
        best_f1 = 0.0
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (oof_proba >= t).astype(int)
            score = f1_score(y_val, preds)
            if score > best_f1:
                best_f1 = score
                best_thresh = t
                
        self.optimal_threshold_ = best_thresh
        print(f"\n  Optimized Decision Threshold = {self.optimal_threshold_:.2f}\n")
        return self

    def predict_proba(self, test_probas: List[np.ndarray]) -> np.ndarray:
        out = np.zeros_like(test_probas[0], dtype=float)
        for w, p in zip(self.weights_, test_probas):
            out += w * p
        return out

    def predict(self, test_probas: List[np.ndarray]) -> np.ndarray:
        # Predict using the mathematically optimal threshold instead of 0.5
        return (self.predict_proba(test_probas)[:, 1] >= self.optimal_threshold_).astype(int)


# =============================================================================
# Layer 2 entry point
# =============================================================================

def run_layer2_ensemble(
    X_train_pca:  np.ndarray,   
    y_train:      np.ndarray,
    X_test_pca:   np.ndarray,   
    y_test:       np.ndarray,
    X_train_full: np.ndarray,   
    X_test_full:  np.ndarray,   
) -> Tuple[np.ndarray, str]:

    n_pca   = X_train_pca.shape[1]
    n_full  = X_train_full.shape[1]

    print("=" * 65)
    print("  LAYER 2 — HYBRID OPTIMIZATION & ENSEMBLE PROCESSING")
    print(f"  Training samples (all, post-SMOTE) : {X_train_pca.shape[0]}")
    print(f"  PCA components  (Experts 1–3)      : {n_pca}")
    print(f"  Full features   (Expert 4 DNN)     : {n_full}")
    print(f"  Test samples                       : {X_test_pca.shape[0]}")
    print("=" * 65 + "\n")

    print("-" * 65)
    print("  Expert #1  —  Firefly Algorithm  (§3.4.1)")
    print("-" * 65)
    fa = FireflyOptimizer(n_pca)
    fa.fit(X_train_pca, y_train)       
    clf_fa, fa_sel = _train_production_clf(X_train_pca, y_train, fa.best_mask_)

    print("-" * 65)
    print("  Expert #2  —  Particle Swarm Optimization  (§3.4.2)")
    print("-" * 65)
    pso = PSOOptimizer(n_pca)
    pso.fit(X_train_pca, y_train)
    clf_pso, pso_sel = _train_production_clf(X_train_pca, y_train, pso.best_mask_)

    print("-" * 65)
    print("  Expert #3  —  FA-PSO Hybrid Integration  (§3.4.3)")
    print("-" * 65)
    hybrid = FAPSOHybridOptimizer(n_pca)
    hybrid.fit(X_train_pca, y_train)
    clf_hyb, hyb_sel = _train_production_clf(X_train_pca, y_train, hybrid.best_mask_)

    print("-" * 65)
    print("  Expert #4  —  Tree-Reduced Deep Neural Network (§3.4.4)")
    print("  (Random Forest Feature Selection -> MLP Classifier)")
    print("-" * 65)
    
    dnn_pipeline = Pipeline([
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42), 
            max_features=40  
        )),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(16, 8), 
            solver="lbfgs", 
            alpha=0.5, 
            random_state=42, 
            max_iter=500
        ))
    ])
    dnn_pipeline.fit(X_train_full, y_train)
    train_acc = accuracy_score(y_train, dnn_pipeline.predict(X_train_full))
    selected_features = dnn_pipeline.named_steps['feature_selection'].get_support().sum()
    print(f"  [DNN] Done | train acc = {train_acc:.4f} | Features reduced from 903 to {selected_features}\n")

    print("-" * 65)
    print("  Ensemble Calibration  —  5-Fold Out-Of-Fold (OOF) Scoring")
    print("-" * 65)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    proba_fa_oof  = cross_val_predict(clf_fa, X_train_pca[:, fa_sel], y_train, cv=cv, method='predict_proba', n_jobs=-1)
    proba_pso_oof = cross_val_predict(clf_pso, X_train_pca[:, pso_sel], y_train, cv=cv, method='predict_proba', n_jobs=-1)
    proba_hyb_oof = cross_val_predict(clf_hyb, X_train_pca[:, hyb_sel], y_train, cv=cv, method='predict_proba', n_jobs=-1)
    proba_dnn_oof = cross_val_predict(dnn_pipeline, X_train_full, y_train, cv=cv, method='predict_proba', n_jobs=-1)

    ensemble = SoftVotingEnsemble()
    ensemble.calibrate_weights([proba_fa_oof, proba_pso_oof, proba_hyb_oof, proba_dnn_oof], y_train)

    proba_fa_test  = clf_fa.predict_proba(X_test_pca[:, fa_sel])
    proba_pso_test = clf_pso.predict_proba(X_test_pca[:, pso_sel])
    proba_hyb_test = clf_hyb.predict_proba(X_test_pca[:, hyb_sel])
    proba_dnn_test = dnn_pipeline.predict_proba(X_test_full)

    ensemble_proba = ensemble.predict_proba([proba_fa_test, proba_pso_test, proba_hyb_test, proba_dnn_test])
    ensemble_preds = ensemble.predict([proba_fa_test, proba_pso_test, proba_hyb_test, proba_dnn_test])

    print("=" * 65)
    print("  LAYER 2 RESULTS  —  TEST SET")
    print("=" * 65)

    for lbl, preds, n_sel, sp in [
        ("Expert #1  FA",    clf_fa.predict(X_test_pca[:, fa_sel]),   len(fa_sel),  "PCA"),
        ("Expert #2  PSO",   clf_pso.predict(X_test_pca[:, pso_sel]), len(pso_sel), "PCA"),
        ("Expert #3  Hybrid",clf_hyb.predict(X_test_pca[:, hyb_sel]), len(hyb_sel), "PCA"),
        ("Expert #4  DNN",   dnn_pipeline.predict(X_test_full),       selected_features, "Trees"),
    ]:
        acc = accuracy_score(y_test, preds)
        try:    auc_s = f"AUC={roc_auc_score(y_test,preds):.4f}"
        except: auc_s = "AUC=N/A"
        print(f"  {lbl:<22}  acc={acc:.4f}  {auc_s}  ({sp}: {n_sel})")

    ens_acc = accuracy_score(y_test, ensemble_preds)
    try:    ens_auc = f"AUC={roc_auc_score(y_test, ensemble_proba[:,1]):.4f}"
    except: ens_auc = "AUC=N/A"
    print(f"  {'Ensemble':<22}  acc={ens_acc:.4f}  {ens_auc}  (Eq. 3.6 weighted, Thresh={ensemble.optimal_threshold_:.2f})")

    report = classification_report(y_test, ensemble_preds, target_names=["Control (0)", "ADHD (1)"])
    print("\n  Classification Report (Ensemble):\n")
    print(report)
    print("  Layer 2 complete — ready for Layer 3.")
    print("=" * 65)

    return ensemble_proba, report

if __name__ == "__main__":
    from datapreprocessing import run_adf_preprocessing

    print(">>> Layer 1 — Preprocessing …\n")
    (X_train, y_train, X_test, y_test,
     feat_names, X_train_pca, X_test_pca,
     scaler, pca) = run_adf_preprocessing()

    print("\n>>> Layer 2 — Hybrid Optimization …\n")
    ensemble_proba, report = run_layer2_ensemble(
        X_train_pca=X_train_pca,  y_train=y_train,
        X_test_pca=X_test_pca,    y_test=y_test,
        X_train_full=X_train,     X_test_full=X_test,
    )

    np.save("layer2_ensemble_proba.npy", ensemble_proba)
    print("\nSaved: layer2_ensemble_proba.npy  →  ready for Layer 3.")