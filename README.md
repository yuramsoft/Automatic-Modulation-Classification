# Automatic-Modulation-Classification with Classical Machine Learnin
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Models](https://img.shields.io/badge/models-linear%20regression%20%7C%20random%20forest-green)
Automatic Modulation Classification (AMC) is a classic wireless-communications problem solved as a multi-class ML task.

The project bridges **wireless communications** and **machine learning**: it generates a **synthetic wireless dataset** from a digital communication
transmitter/receiver chain, extracts statistical signal features, and **benchmarks classical ML classifiers** (Logistic Regression, Random Forest,
Gradient Boosting, SVM, KNN and a small MLP) on the task of **Automatic Modulation Classification (AMC)**.

---

## 1. The Problem

A receiver in a cognitive-radio spectrum-monitoring system observes a passband/baseband signal and must answer:

> *"Which digital modulation is being used — BPSK, QPSK, 16-QAM, ...?"*

Knowing the modulation is a prerequisite for demodulation, link adaptation and interference management. This project treats AMC as a **supervised
multi-class classification** problem over hand-crafted signal features.

### Signal model

The received baseband block is modeled as

```
y[n] = (x * h)[n] + w[n]
```
* `x[n]` — transmitted symbols, pulse-shaped with a **root-raised-cosine (RRC)** filter (roll-off 0.35, 4 samples/symbol),
* `h[n]` — channel (flat channel `h = 1` in this project),
* `w[n]` — complex **additive white Gaussian noise (AWGN)**, scaled so the block is observed at a target **SNR** (swept from −10 dB to +20 dB).

Because cumulants are **blind to channel phase and robust to Gaussian noise** (white Gaussian noise contributes to the second-order cumulant only), they are
classical AMC features — see Swami & Sadler, *"Hierarchical digital modulation classification using cumulants,"* IEEE Trans. Communications, 2000.

### Features extracted per block

| Feature | Definition (E[·] = sample average, y = received block) |
|---|---|
| `c20_abs` | \|C₂₀\|/C₂₁ with C₂₀ = E[y²], C₂₁ = E[\|y\|²] |
| `c40` | (E[y⁴] − 3E[y²]²) / C₂₁² |
| `c41` | \|E[y³ȳ] − 3E[y²]E[\|y\|²]\| / C₂₁² |
| `c42` | (E[\|y\|⁴] − \|E[y²]\|² − 2E[\|y\|²]²) / C₂₁² |
| `energy` | average power E[\|y\|²] of the block |
| `spectral_entropy` | Shannon entropy of the normalized PSD |
| `snr` | true SNR in dB (known/estimated in practice) |

Each modulation has a distinct cumulant "fingerprint" (e.g. BPSK has a large real C₄₀, QPSK has C₄₀ ≈ 1, 16-QAM C₄₀ ≈ −0.68, all normalized), which makes
them linearly *almost* separable — an ideal playground for comparing a linear model against non-linear ones.

---

## 2. Fundamental Theory

### 2.1 The classifiers

**Logistic Regression** models
`P(y = k | x) = softmax(Wx + b)` and maximizes the log-likelihood. Despite the name it *is* a linear classifier — here it is the baseline that shows how far linearly-separable cumulant features get you.

**Random Forest** (Breiman, 2001) builds `B` decision trees on bootstrapped samples of the data, splitting each node on the feature/threshold that maximizes information gain (Gini impurity). Prediction is the majority vote. Bagging + random feature subsets decorrelate the trees and reduce variance.

**Gradient Boosting** (Friedman, 2001; XGBoost-style) builds trees *sequentially*: each new tree fits the **negative gradient of the loss** (residuals) of the ensemble so far,

```
F_m(x) = F_{m−1}(x) + ν · h_m(x),   h_m ≈ argmin Σ L(y_i, F_{m−1}(x_i) + h(x_i))
```

with learning rate `ν`. Boosting reduces *bias* and typically edges out
random forests on tabular data.

**Support Vector Machine (RBF kernel)** finds the maximum-margin hyperplane in a feature space implicitly defined by the kernel
`K(x, x') = exp(−γ‖x − x'‖²)`. The soft-margin formulation with slack variables `ξ_i` minimizes `½‖w‖² + C Σ ξ_i` subject to
`y_i(wᵀx_i + b) ≥ 1 − ξ_i`. Multi-class probabilities are obtained via internal one-vs-rest Platt scaling (`probability=True`).

**KNN** predicts by majority vote of the `k` nearest training points, a non-parametric sanity check. **MLP** is a small 2-hidden-layer neural
network included as a modern reference point.

### 2.2 Evaluation metrics

With TP/FP/FN counted per class and then **macro-averaged** (treating all modulations equally):

| Metric | Formula |
|---|---|
| Accuracy | (TP + TN) / N |
| Precision | TP / (TP + FP) |
| Recall (sensitivity) | TP / (TP + FN) |
| F1-score | 2 · P · R / (P + R) |
| ROC-AUC | Area under the curve of TPR vs FPR as the decision threshold varies; computed **one-vs-rest** with macro averaging from `predict_proba` |

AUC measures **ranking quality** independent of threshold and class priors, which is why it is reported alongside the threshold-fixed metrics.
The code also produces **per-SNR accuracy curves**, the standard diagnostic in wireless ML (deep-learning AMC papers, e.g. O'Shea & Hoydis, 2017, use the
same convention).

---

## 3. Repository Structure

```
wireless-ml-classification/
├── README.md                  ← this file
├── requirements.txt
├── .gitignore
├── src/
│   ├── generate_dataset.py    # synthetic TX/RX chain + feature extraction
│   └── train_evaluate.py      # training, metrics, figures, tables
├── data/
│   └── dataset.csv            # generated dataset (gitignored, regenerate)
└── results/
    ├── figures/               # ROC curves, confusion matrices, SNR curve
    ├── tables/                # metrics_summary.csv + per-model reports
    └── run_summary.json
```

---

## 4. Implementation Steps

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate the wireless dataset (3 modulations × SNR sweep)
python src/generate_dataset.py \
    --out data/dataset.csv \
    --snr-min -10 --snr-max 20 --snr-step 2 \
    --samples 200          # per (class, SNR) → 19,200 rows total

# 3. Train and evaluate all models
python src/train_evaluate.py --data data/dataset.csv --out results
```

Everything downstream of step 3 is automatic: the script trains each model inside a `StandardScaler` + classifier `Pipeline`, evaluates it on a held-out
stratified test set, and writes metrics tables and four diagnostic figures.

**Pipeline details**

1. Stratified train/test split (75/25, seed 42) — class ratios and SNR distribution are preserved in both splits.
2. Standardization inside each pipeline (important for LR/SVM/KNN/MLP).
3. Model zoo: `LogisticRegression`, `RandomForest(300 trees)`, `GradientBoosting(200, lr=0.1)`, `SVC(rbf, C=10, probability=True)`, `KNN(7)`, `MLP(64-32)`.
4. Metrics: accuracy, macro precision/recall/F1, macro one-vs-rest ROC-AUC per-model `classification_report`.
5. Figures: grouped metric bars, confusion matrices, per-class ROC curves, and the signature **accuracy-vs-SNR** curve.

---

## 5. Example Results

Run with `--samples 100` per (class, SNR) over SNR −10…20 dB (4,800 blocks, 7 features):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| LogisticRegression | **0.9125** | 0.9127 | 0.9125 | 0.9126 | 0.9847 |
| GradientBoosting | 0.9100 | 0.9100 | 0.9100 | 0.9100 | 0.9845 |
| SVM (RBF) | 0.9067 | 0.9068 | 0.9067 | 0.9067 | 0.9849 |
| MLP (64-32) | 0.9042 | 0.9062 | 0.9042 | 0.9040 | **0.9870** |
| RandomForest | 0.9000 | 0.9002 | 0.9000 | 0.9001 | 0.9847 |
| KNN | 0.8958 | 0.8960 | 0.8958 | 0.8959 | 0.9820 |

Observations consistent with wireless-ML literature:

* Cumulant features are *nearly* linearly separable at moderate/high SNR, so the **linear baseline is highly competitive**; tree ensembles and SVM add a few points mostly at low SNR.
* All models degrade gracefully as SNR decreases (see `results/figures/accuracy_vs_snr.png`) — errors concentrate on the
  **QPSK ↔ 16-QAM** boundary visible in the confusion matrices.
* ROC-AUC > 0.98 for every model: even when the argmax is wrong, the true class is usually the second-most probable — a sign that the feature space carries nearly all the available information.

---

## 6. Ideas to Extend the Project

* **More modulations**: add 8-PSK, 64-QAM, CPFSK; the cumulant fingerprints remain distinct (higher-order cumulants separate even same-order constellations).
* **Realistic channels**: multipath (FIR Rayleigh channel), carrier-frequency and phase offsets, sample-rate offset, IQ imbalance.
* **Sequence models**: feed raw I/Q windows (or their STFT) to a 1-D CNN or LSTM and compare against the hand-crafted-feature baseline.
* **Cross-validation & hyperparameter search**: wrap the pipeline in `GridSearchCV` / `Optuna`.
* **Calibration**: replace `predict` with threshold tuning per SNR using the ROC curves; or use cost-sensitive learning when misclassifying BPSK as
  16-QAM is more expensive than the reverse.

## 7. References
1. A. Swami, B. M. Sadler, "Hierarchical digital modulation classification
   using cumulants," *IEEE Trans. Communications*, 2000.
2. L. Breiman, "Random Forests," *Machine Learning*, 2001.
3. J. H. Friedman, "Greedy function approximation: a gradient boosting
   machine," *Annals of Statistics*, 2001.
4. T. O'Shea, J. Hoydis, "An Introduction to Deep Learning for the Physical
   Layer," *IEEE Trans. Cognitive Communications and Networking*, 2017.

## Author

**Ibrahim Mustapha, PhD**

Department of Electrical & Electronic Engineering
University of Maiduguri
Communications Engineering / Wireless Communications
Python & Machine Learning

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This project is primarily an educational and research-oriented implementation. Predictions produced by the initial model should not be treated as a substitute for detailed radio-frequency planning, field measurements, or validated commercial propagation models.
6. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python,"
   *JMLR*, 2011.
