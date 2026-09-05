"""
=============================================================================
ADFI Framework — Layer 2: Hybrid Optimization & Ensemble Processing
=============================================================================
Maps to Chapter 3.4 of the ADFI thesis (FYP_Chapter_03.docx)

Structure
---------
  Expert #1  — Firefly Algorithm (FA)          §3.4.1  Eqs. 3.1 – 3.2
  Expert #2  — Particle Swarm Optimization     §3.4.2  Eqs. 3.3 – 3.4
  Expert #3  — FA-PSO Hybrid Integration       §3.4.3
  Expert #4  — Deep Neural Network (DNN)       §3.4.4  Eq.  3.5
  Ensemble   — Soft Voting Module              §3.4.5  Eq.  3.6

Key design decision — PCA input
---------------------------------
Layer 1 produces two parallel feature sets:
  X_scaled  (903 features) — full normalised feature matrix
  X_pca     (64 features)  — 95 % variance PCA projection

Layer 2 runs on X_pca for the metaheuristic experts (FA, PSO, Hybrid)
because:
  1. FA and PSO use Euclidean distance in feature space.  Meaningless
     distances in 903-D cause every firefly / particle to look equally
     attractive, collapsing the search to random walk.
  2. 64 dimensions gives the population (n=20) enough density to explore
     the space meaningfully within the iteration budget.
  3. PCA removes multicollinear tsfresh features, so selected components
     carry orthogonal variance — each dimension adds real information.

The DNN expert (Expert #4) operates on the full X_scaled (903 features)
because its dense layers can exploit correlations that PCA discards, and
it does not rely on distance calculations.

Wrapper-based feature selection (FA / PSO / Hybrid) selects a binary
mask over the 64 PCA components.  The selected components are then used
to train a dedicated MLP classifier per expert.

Connection to Layer 1
----------------------
  from datapreprocessing import run_adf_preprocessing
  (X_train, y_train, X_test, y_test,
   feat_names, X_train_pca, X_test_pca, scaler, pca) = run_adf_preprocessing()

  ensemble_proba, report = run_layer2_ensemble(
      X_train_pca, y_train, X_test_pca, y_test,   # PCA data for experts 1-3
      X_train,     X_test                          # full data for DNN
  )

Output → Layer 3
-----------------
  ensemble_proba : np.ndarray (n_test, 2)  — ADHD probability per patient
  report         : str  sklearn classification_report
  Saved to layer2_ensemble_proba.npy for SHAP / LIME handoff
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, accuracy_score, classification_report, roc_auc_score


# =============================================================================
# Shared utilities
# =============================================================================

def _fitness(X: np.ndarray, y: np.ndarray,
             position: np.ndarray, cv_folds: int = 5) -> float:
    """
    Evaluate a real-valued position vector as a binary feature-selection
    mask over the PCA components (threshold > 0.5 → component selected).

    Returns mean stratified-CV accuracy — the 'brightness' I in FA and
    the objective value in PSO.  An all-zero mask returns 0.0 (penalty).

    Stratified 5-fold CV is used instead of 3-fold because the dataset
    is small (70 training samples) — 5-fold gives larger training folds
    and more stable fitness estimates.
    """
    selected = np.where(position > 0.5)[0]
    if len(selected) == 0:
        return 0.0

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X[:, selected], y,
                             cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean())


def _train_classifier_on_mask(X_train: np.ndarray, y_train: np.ndarray,
                               mask: np.ndarray,
                               random_state: int = 42):
    """
    Train a production MLP on the PCA components selected by a binary mask.
    Called once per expert after optimisation completes.

    Returns
    -------
    clf      : fitted MLPClassifier
    selected : np.ndarray of selected component indices
    """
    selected = np.where(mask > 0.5)[0]
    if len(selected) == 0:
        selected = np.arange(X_train.shape[1])   # safety: use all if mask empty

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        learning_rate_init=0.001,
        random_state=random_state
    )
    clf.fit(X_train[:, selected], y_train)
    return clf, selected


# =============================================================================
# Expert #1 — Firefly Algorithm  (§3.4.1)
# =============================================================================

@dataclass
class FireflyConfig:
    """
    Hyper-parameters for the Firefly Algorithm.

    n_fireflies : population size  n
    max_iter    : number of iterations  t
    beta0       : maximum attractiveness at r = 0  (β₀)
    gamma       : light absorption coefficient     (γ)
    alpha       : randomisation step size          (α)
    """
    n_fireflies:  int   = 20
    max_iter:     int   = 50
    beta0:        float = 1.0
    gamma:        float = 1.0
    alpha:        float = 0.20
    random_state: int   = 42


class FireflyOptimizer:
    """
    Wrapper-based global feature selector — Firefly Algorithm (Yang, 2010).

    Each firefly encodes a real-valued position x_i in [0, 1]^n_components.
    The position is decoded to a binary PCA-component mask at each fitness
    evaluation (component selected if position[j] > 0.5).

    Brighter fireflies (higher diagnostic accuracy) attract dimmer ones,
    driving the swarm towards high-fitness feature subsets.

    Eq. 3.1 — Attractiveness (decays with Cartesian distance r):
        β(r) = β₀ · exp(−γ · r²)

    Eq. 3.2 — Position update (firefly i attracted to brighter firefly j):
        x_i(t+1) = x_i(t)
                 + β₀ · exp(−γ · r_ij²) · (x_j(t) − x_i(t))
                 + α · ε_i(t)
        where ε_i(t) ~ N(0, 1) for global diversification

    Time complexity: O(n² · t)  (Khan et al., 2020)
    """

    def __init__(self, n_features: int, cfg: FireflyConfig = None):
        self.n_features      = n_features
        self.cfg             = cfg or FireflyConfig()
        self.best_mask_:     np.ndarray  = np.array([])
        self.best_fitness_:  float       = 0.0
        self.fitness_curve_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FireflyOptimizer":
        rng = np.random.default_rng(self.cfg.random_state)
        n, cfg = self.n_features, self.cfg

        # Initialise swarm uniformly in [0, 1]^n
        positions = rng.uniform(0.0, 1.0, (cfg.n_fireflies, n))
        fitness   = np.array([_fitness(X, y, p) for p in positions])

        print(f"  [FA] Start | best fitness = {fitness.max():.4f} | "
              f"components = {n}")

        for t in range(cfg.max_iter):
            for i in range(cfg.n_fireflies):
                for j in range(cfg.n_fireflies):
                    if fitness[j] > fitness[i]:
                        # Eq. 3.1 — attractiveness
                        r_ij = np.linalg.norm(positions[i] - positions[j])
                        beta = cfg.beta0 * np.exp(-cfg.gamma * r_ij ** 2)
                        # Eq. 3.2 — position update
                        eps          = rng.standard_normal(n)
                        positions[i] = np.clip(
                            positions[i]
                            + beta * (positions[j] - positions[i])
                            + cfg.alpha * eps,
                            0.0, 1.0
                        )
                        fitness[i] = _fitness(X, y, positions[i])

            best_t = float(fitness.max())
            self.fitness_curve_.append(best_t)

            if (t + 1) % 10 == 0 or t == cfg.max_iter - 1:
                n_sel = int((positions[int(np.argmax(fitness))] > 0.5).sum())
                print(f"  [FA] Iter {t+1:3d}/{cfg.max_iter} | "
                      f"best fitness = {best_t:.4f} | selected = {n_sel}/{n}")

        best_idx           = int(np.argmax(fitness))
        self.best_mask_    = (positions[best_idx] > 0.5).astype(int)
        self.best_fitness_ = float(fitness[best_idx])
        print(f"  [FA] Done  | fitness = {self.best_fitness_:.4f} | "
              f"components selected = {int(self.best_mask_.sum())}/{n}\n")
        return self

    @property
    def best_continuous_position_(self) -> np.ndarray:
        """Continuous position of best firefly — used as PSO warm-start seed."""
        return self.best_mask_.astype(float)


# =============================================================================
# Expert #2 — Particle Swarm Optimization  (§3.4.2)
# =============================================================================

@dataclass
class PSOConfig:
    """
    Hyper-parameters for Particle Swarm Optimization.

    n_particles : swarm size
    max_iter    : iterations
    w           : inertia weight  — controls momentum of previous velocity
    c1          : cognitive rate  — pull towards personal best
    c2          : social rate     — pull towards global best
    """
    n_particles:  int   = 20
    max_iter:     int   = 50
    w:            float = 0.70
    c1:           float = 1.50
    c2:           float = 1.50
    random_state: int   = 42


class PSOOptimizer:
    """
    Fast local feature selector — Particle Swarm Optimization
    (Kennedy & Eberhart, 1995).

    Each particle holds a position x_i and velocity v_i in [0, 1]^n.
    Movement is guided by the particle's personal best (pbest_i) and
    the swarm's global best (gbest).

    Eq. 3.3 — Velocity update:
        v_i(t+1) = w · v_i(t)
                 + c₁ · r₁ · (pbest_i − x_i(t))
                 + c₂ · r₂ · (gbest   − x_i(t))
        where r₁, r₂ ~ Uniform[0, 1]

    Eq. 3.4 — Position update:
        x_i(t+1) = x_i(t) + v_i(t+1)

    Parameters
    ----------
    gbest_seed : optional warm-start from FA.  When provided (Hybrid mode),
                 particle 0 is initialised at FA's best position so PSO
                 begins its local search from the most promising point found.
    """

    def __init__(self, n_features: int, cfg: PSOConfig = None,
                 gbest_seed: np.ndarray = None):
        self.n_features      = n_features
        self.cfg             = cfg or PSOConfig()
        self.gbest_seed      = gbest_seed
        self.best_mask_:     np.ndarray  = np.array([])
        self.best_fitness_:  float       = 0.0
        self.fitness_curve_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PSOOptimizer":
        rng = np.random.default_rng(self.cfg.random_state)
        n, cfg = self.n_features, self.cfg

        positions  = rng.uniform(0.0, 1.0, (cfg.n_particles, n))
        velocities = rng.uniform(-0.5, 0.5, (cfg.n_particles, n))

        # Inject FA warm-start into particle 0 (hybrid mode)
        if self.gbest_seed is not None:
            positions[0] = np.clip(self.gbest_seed, 0.0, 1.0)

        fitness  = np.array([_fitness(X, y, p) for p in positions])
        pbest    = positions.copy()
        pbest_f  = fitness.copy()
        gbest    = pbest[int(np.argmax(pbest_f))].copy()
        gbest_f  = float(pbest_f.max())

        print(f"  [PSO] Start | best fitness = {gbest_f:.4f} | "
              f"components = {n}")

        for t in range(cfg.max_iter):
            r1 = rng.uniform(0.0, 1.0, (cfg.n_particles, n))
            r2 = rng.uniform(0.0, 1.0, (cfg.n_particles, n))

            # Eq. 3.3 — velocity update
            velocities = (cfg.w  * velocities
                          + cfg.c1 * r1 * (pbest  - positions)
                          + cfg.c2 * r2 * (gbest  - positions))

            # Eq. 3.4 — position update
            positions = np.clip(positions + velocities, 0.0, 1.0)

            for i in range(cfg.n_particles):
                f = _fitness(X, y, positions[i])
                if f > pbest_f[i]:
                    pbest[i], pbest_f[i] = positions[i].copy(), f
                    if f > gbest_f:
                        gbest, gbest_f = positions[i].copy(), f

            self.fitness_curve_.append(gbest_f)

            if (t + 1) % 10 == 0 or t == cfg.max_iter - 1:
                n_sel = int((gbest > 0.5).sum())
                print(f"  [PSO] Iter {t+1:3d}/{cfg.max_iter} | "
                      f"best fitness = {gbest_f:.4f} | selected = {n_sel}/{n}")

        self.best_mask_    = (gbest > 0.5).astype(int)
        self.best_fitness_ = gbest_f
        print(f"  [PSO] Done  | fitness = {self.best_fitness_:.4f} | "
              f"components selected = {int(self.best_mask_.sum())}/{n}\n")
        return self


# =============================================================================
# Expert #3 — FA-PSO Hybrid Integration  (§3.4.3)
# =============================================================================

class FAPSOHybridOptimizer:
    """
    Two-phase sequential hybrid optimizer (Aydilek, 2018).

    Phase 1 — FA (global exploration):
        Scans the full PCA-component hyperspace to identify the
        high-potential region and produce a promising gbest seed.

    Phase 2 — PSO (local exploitation):
        Receives FA's best continuous position as a warm-start seed and
        fine-tunes the solution to its mathematical optimum.

    The FA seed is perturbed by ±0.1 uniform noise before injection into
    PSO particle 0, preserving swarm diversity while still giving PSO a
    head-start near FA's best region.

    Overall time complexity: O(MaxFES · n² · t)  worst case (Khan, 2020)
    """

    def __init__(self, n_features: int,
                 fa_cfg:  FireflyConfig = None,
                 pso_cfg: PSOConfig     = None):
        self.n_features = n_features
        self.fa_cfg     = fa_cfg  or FireflyConfig()
        self.pso_cfg    = pso_cfg or PSOConfig()
        self.fa_:  FireflyOptimizer = None
        self.pso_: PSOOptimizer     = None
        self.best_mask_:    np.ndarray = np.array([])
        self.best_fitness_: float      = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FAPSOHybridOptimizer":

        print("  [Hybrid] Phase 1 — FA global exploration")
        self.fa_ = FireflyOptimizer(self.n_features, self.fa_cfg)
        self.fa_.fit(X, y)

        # Perturb FA seed to maintain PSO swarm diversity
        rng  = np.random.default_rng(0)
        seed = np.clip(
            self.fa_.best_continuous_position_
            + rng.uniform(-0.1, 0.1, self.n_features),
            0.0, 1.0
        )

        print("  [Hybrid] Phase 2 — PSO local exploitation (warm-started from FA)")
        self.pso_ = PSOOptimizer(self.n_features, self.pso_cfg,
                                 gbest_seed=seed)
        self.pso_.fit(X, y)

        self.best_mask_    = self.pso_.best_mask_
        self.best_fitness_ = self.pso_.best_fitness_
        print(f"  [Hybrid] Done | final fitness = {self.best_fitness_:.4f}\n")
        return self


# =============================================================================
# Expert #4 — Deep Neural Network  (§3.4.4)
# =============================================================================

class DNNExpert:
    """
    End-to-end MLP operating on the FULL 903-feature scaled matrix —
    NOT the PCA projection.  This is the chapter's 'end-to-end check':
    the DNN can recognise complex non-linear patterns across all features
    that distance-based metaheuristics searching PCA space may miss.

    Architecture: Input(903) → Dense(256) → Dense(128) → Dense(64) → Softmax(2)

    Leaky ReLU activation (Eq. 3.5):
        f(x) = max(0.01x, x)
    Prevents vanishing gradients on sparse behavioral features so the
    network detects subtle signals like micro-fidgeting or latency drift.

    sklearn MLPClassifier used for portability.  Production Keras
    implementation with true Leaky ReLU is in the docstring below.

    -- Keras / TensorFlow production implementation -------------------------
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization

    def build_dnn(n_features: int) -> Sequential:
        model = Sequential([
            Dense(256, input_dim=n_features),
            LeakyReLU(alpha=0.01),
            Dropout(0.30),
            BatchNormalization(),
            Dense(128),
            LeakyReLU(alpha=0.01),
            Dropout(0.30),
            Dense(64),
            LeakyReLU(alpha=0.01),
            Dense(2, activation='softmax')    # Softmax -> probability dist.
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    ------------------------------------------------------------------------
    """

    def __init__(self, random_state: int = 42):
        self.clf = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",          # Leaky ReLU via Keras in production
            solver="adam",
            learning_rate_init=0.001,
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=random_state
        )
        self.train_accuracy_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DNNExpert":
        """X must be the full scaled matrix (903 features), NOT PCA."""
        self.clf.fit(X, y)
        self.train_accuracy_ = accuracy_score(y, self.clf.predict(X))
        print(f"  [DNN] Done  | train accuracy = {self.train_accuracy_:.4f} | "
              f"iterations = {self.clf.n_iter_} | "
              f"input features = {X.shape[1]}\n")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_samples, 2) probability array."""
        return self.clf.predict_proba(X)


# =============================================================================
# Ensemble Voting Module  (§3.4.5)
# =============================================================================

class SoftVotingEnsemble:
    """
    Confidence-weighted soft voting fusion of all four expert outputs.

    Eq. 3.6 — Weighted ensemble probability:
        ŷ(x) = Σ_{k=1}^{4}  W_k · P_k(x)

    Confidence weights W_k
    ----------------------
    W_k is inversely proportional to each expert's cross-entropy
    validation loss — experts with lower loss (higher reliability) receive
    a proportionally larger vote:

        raw_k = 1 / (val_cross_entropy_k + ε)
        W_k   = raw_k / Σ raw_k

    Weights are calibrated on a held-out 20% validation slice of the
    training data.  The test set is NEVER used for weight fitting.
    This design guarantees diagnostic stability and minimises the risk
    of variance and overfitting (Ziarul Islam et al., 2025).
    """

    def __init__(self):
        self.weights_:    np.ndarray  = np.array([])
        self.val_losses_: List[float] = []

    def calibrate_weights(self,
                          val_probas: List[np.ndarray],
                          y_val:      np.ndarray) -> "SoftVotingEnsemble":
        """
        Compute normalised confidence weights from validation probabilities.

        Parameters
        ----------
        val_probas : list of 4 arrays, each (n_val, 2)
        y_val      : ground-truth validation labels
        """
        raw    = []
        labels = ["FA", "PSO", "Hybrid", "DNN"]

        for k, (proba, label) in enumerate(zip(val_probas, labels), start=1):
            loss = log_loss(y_val, proba)
            w    = 1.0 / (loss + 1e-9)
            raw.append(w)
            self.val_losses_.append(loss)
            print(f"  Expert {k} ({label:<6}) : "
                  f"val_loss = {loss:.4f}  →  raw_weight = {w:.3f}")

        total         = sum(raw)
        self.weights_ = np.array([w / total for w in raw])

        print("\n  Normalised confidence weights W_k (Eq. 3.6):")
        for label, w in zip(labels, self.weights_):
            print(f"    W_{label:<6} = {w:.4f}")
        print()
        return self

    def predict_proba(self, test_probas: List[np.ndarray]) -> np.ndarray:
        """
        Apply Eq. 3.6:  ŷ(x) = Σ W_k · P_k(x)
        Returns np.ndarray of shape (n_test, 2).
        """
        result = np.zeros_like(test_probas[0], dtype=float)
        for w, proba in zip(self.weights_, test_probas):
            result += w * proba
        return result

    def predict(self, test_probas: List[np.ndarray],
                threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(test_probas)[:, 1] >= threshold).astype(int)


# =============================================================================
# Layer 2 entry point
# =============================================================================

def run_layer2_ensemble(
    X_train_pca:  np.ndarray,       # PCA-reduced training set  (n_train, 64)
    y_train:      np.ndarray,       # training labels
    X_test_pca:   np.ndarray,       # PCA-reduced test set      (n_test,  64)
    y_test:       np.ndarray,       # test labels
    X_train_full: np.ndarray,       # full scaled training set  (n_train, 903)
    X_test_full:  np.ndarray,       # full scaled test set      (n_test,  903)
    fa_cfg:       FireflyConfig = None,
    pso_cfg:      PSOConfig     = None,
) -> Tuple[np.ndarray, str]:
    """
    Execute the complete Layer 2 pipeline.

    Parameters
    ----------
    X_train_pca  : PCA-reduced SMOTE-balanced training data (Experts 1–3)
    y_train      : training labels (balanced by SMOTE)
    X_test_pca   : PCA-reduced held-out test data (Experts 1–3)
    y_test       : held-out test labels
    X_train_full : full scaled training data (Expert 4 DNN only)
    X_test_full  : full scaled test data (Expert 4 DNN only)
    fa_cfg       : optional FireflyConfig override
    pso_cfg      : optional PSOConfig override

    Returns
    -------
    ensemble_proba : np.ndarray (n_test, 2) — passed to Layer 3 SHAP/LIME
    report         : str classification_report
    """
    n_pca  = X_train_pca.shape[1]
    n_full = X_train_full.shape[1]
    fa_cfg  = fa_cfg  or FireflyConfig()
    pso_cfg = pso_cfg or PSOConfig()

    print("=" * 65)
    print("  LAYER 2 — HYBRID OPTIMIZATION & ENSEMBLE PROCESSING")
    print(f"  Training samples (post-SMOTE)  : {X_train_pca.shape[0]}")
    print(f"  PCA components (Experts 1–3)   : {n_pca}")
    print(f"  Full features  (Expert 4 DNN)  : {n_full}")
    print(f"  Test samples                   : {X_test_pca.shape[0]}")
    print("=" * 65 + "\n")

    # ── Calibration split (20% of training) ──────────────────────────────────
    # Held out for ensemble weight calibration ONLY.
    # The test set is never seen during weight fitting.
    (X_tr_pca, X_val_pca,
     X_tr_full, X_val_full,
     y_tr, y_val) = train_test_split(
        X_train_pca, X_train_full, y_train,
        test_size=0.20, stratify=y_train, random_state=0
    )

    print(f"  Calibration split — Train: {len(y_tr)} | Val: {len(y_val)}\n")

    # =========================================================================
    # Expert #1 — Firefly Algorithm
    # =========================================================================
    print("-" * 65)
    print("  Expert #1  —  Firefly Algorithm  (§3.4.1 / Eqs. 3.1–3.2)")
    print("-" * 65)
    fa = FireflyOptimizer(n_pca, fa_cfg)
    fa.fit(X_tr_pca, y_tr)
    clf_fa, fa_sel   = _train_classifier_on_mask(X_tr_pca, y_tr, fa.best_mask_)
    proba_fa_val     = clf_fa.predict_proba(X_val_pca[:, fa_sel])
    proba_fa_test    = clf_fa.predict_proba(X_test_pca[:, fa_sel])

    # =========================================================================
    # Expert #2 — Particle Swarm Optimization
    # =========================================================================
    print("-" * 65)
    print("  Expert #2  —  Particle Swarm Optimization  (§3.4.2 / Eqs. 3.3–3.4)")
    print("-" * 65)
    pso = PSOOptimizer(n_pca, pso_cfg)
    pso.fit(X_tr_pca, y_tr)
    clf_pso, pso_sel = _train_classifier_on_mask(X_tr_pca, y_tr, pso.best_mask_)
    proba_pso_val    = clf_pso.predict_proba(X_val_pca[:, pso_sel])
    proba_pso_test   = clf_pso.predict_proba(X_test_pca[:, pso_sel])

    # =========================================================================
    # Expert #3 — FA-PSO Hybrid
    # =========================================================================
    print("-" * 65)
    print("  Expert #3  —  FA-PSO Hybrid Integration  (§3.4.3)")
    print("-" * 65)
    hybrid = FAPSOHybridOptimizer(n_pca, fa_cfg, pso_cfg)
    hybrid.fit(X_tr_pca, y_tr)
    clf_hyb, hyb_sel = _train_classifier_on_mask(X_tr_pca, y_tr, hybrid.best_mask_)
    proba_hyb_val    = clf_hyb.predict_proba(X_val_pca[:, hyb_sel])
    proba_hyb_test   = clf_hyb.predict_proba(X_test_pca[:, hyb_sel])

    # =========================================================================
    # Expert #4 — Deep Neural Network (full feature matrix)
    # =========================================================================
    print("-" * 65)
    print("  Expert #4  —  Deep Neural Network  (§3.4.4 / Eq. 3.5)")
    print("  Input: full scaled matrix (903 features) — not PCA")
    print("-" * 65)
    dnn = DNNExpert(random_state=42)
    dnn.fit(X_tr_full, y_tr)
    proba_dnn_val    = dnn.predict_proba(X_val_full)
    proba_dnn_test   = dnn.predict_proba(X_test_full)

    # =========================================================================
    # Soft Voting Ensemble (§3.4.5 / Eq. 3.6)
    # =========================================================================
    print("-" * 65)
    print("  Ensemble  —  Soft Voting Module  (§3.4.5 / Eq. 3.6)")
    print("-" * 65)
    ensemble = SoftVotingEnsemble()
    ensemble.calibrate_weights(
        val_probas=[proba_fa_val, proba_pso_val, proba_hyb_val, proba_dnn_val],
        y_val=y_val
    )
    ensemble_proba = ensemble.predict_proba(
        [proba_fa_test, proba_pso_test, proba_hyb_test, proba_dnn_test]
    )
    ensemble_preds = ensemble.predict(
        [proba_fa_test, proba_pso_test, proba_hyb_test, proba_dnn_test]
    )

    # =========================================================================
    # Results
    # =========================================================================
    print("=" * 65)
    print("  LAYER 2 RESULTS  —  TEST SET")
    print("=" * 65)

    experts_summary = [
        ("Expert #1  FA",
         clf_fa.predict(X_test_pca[:, fa_sel]),   len(fa_sel),   "PCA"),
        ("Expert #2  PSO",
         clf_pso.predict(X_test_pca[:, pso_sel]), len(pso_sel),  "PCA"),
        ("Expert #3  Hybrid",
         clf_hyb.predict(X_test_pca[:, hyb_sel]), len(hyb_sel),  "PCA"),
        ("Expert #4  DNN",
         dnn.clf.predict(X_test_full),             n_full,        "Full"),
    ]

    for label, preds, n_sel, space in experts_summary:
        acc = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, preds)
            auc_str = f"AUC = {auc:.4f}"
        except Exception:
            auc_str = "AUC = N/A"
        print(f"  {label:<22}  acc = {acc:.4f}  {auc_str}  "
              f"({space} features: {n_sel})")

    ens_acc = accuracy_score(y_test, ensemble_preds)
    try:
        ens_auc = roc_auc_score(y_test, ensemble_proba[:, 1])
        ens_auc_str = f"AUC = {ens_auc:.4f}"
    except Exception:
        ens_auc_str = "AUC = N/A"

    print(f"  {'Ensemble':<22}  acc = {ens_acc:.4f}  {ens_auc_str}  "
          f"(Eq. 3.6 weighted)")

    report = classification_report(
        y_test, ensemble_preds,
        target_names=["Control (0)", "ADHD (1)"]
    )
    print("\n  Classification Report (Ensemble):\n")
    print(report)
    print("  Layer 2 complete — ready for Layer 3 (SHAP / LIME / LLM Agents).")
    print("=" * 65)

    return ensemble_proba, report


# =============================================================================
# Direct execution — connects Layer 1 → Layer 2
# =============================================================================

if __name__ == "__main__":
    from datapreprocessing import run_adf_preprocessing

    print(">>> Running Layer 1 ...\n")
    (X_train, y_train, X_test, y_test,
     feat_names, X_train_pca, X_test_pca,
     scaler, pca) = run_adf_preprocessing()

    print("\n>>> Running Layer 2 ...\n")
    ensemble_proba, report = run_layer2_ensemble(
        X_train_pca=X_train_pca,   # 64 PCA components → Experts 1, 2, 3
        y_train=y_train,
        X_test_pca=X_test_pca,
        y_test=y_test,
        X_train_full=X_train,      # 903 full features → Expert 4 DNN
        X_test_full=X_test,
    )

    # Save ensemble probabilities for Layer 3 (SHAP / LIME / LLM handoff)
    np.save("layer2_ensemble_proba.npy", ensemble_proba)
    print("\nSaved: layer2_ensemble_proba.npy  ->  ready for Layer 3.")