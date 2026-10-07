"""
Train and evaluate classical ML classifiers for Automatic Modulation
Classification (AMC).

Models compared
---------------
  * Logistic Regression   (linear baseline)
  * Random Forest         (bagged decision trees)
  * Gradient Boosting     (AdaBoost-style sequential error correction)
  * Support Vector Machine (RBF kernel, one-vs-rest probabilities)
  * k-Nearest Neighbors   (non-parametric baseline)
  * MLPClassifier         (small neural-network reference)

Metrics
-------
accuracy, macro precision / recall / F1, ROC-AUC (one-vs-rest, macro).
Also produces per-SNR accuracy curves and confusion matrices.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc,
                             classification_report)

RANDOM_STATE = 42


def get_models() -> dict:
    """Model zoo. All models expose predict_proba for ROC-AUC."""
    return {
        #"LogisticRegression": LogisticRegression(max_iter=2000,
        #                                         multi_class="multinomial"),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=300,
                                               max_depth=None,
                                               random_state=RANDOM_STATE,
                                               n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200,
                                                       learning_rate=0.1,
                                                       max_depth=3,
                                                       random_state=RANDOM_STATE),
        "SVM_RBF": SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                             random_state=RANDOM_STATE),
    }


def evaluate_model(name, model, X_test, y_test, classes) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro",
                                           zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro",
                                     zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc_ovr_macro": roc_auc_score(y_test, y_prob, multi_class="ovr",
                                           average="macro"),
    }


def plot_confusion_matrices(results: dict, y_test, classes, outdir):
    n = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (name, model) in zip(axes.ravel(), results.items()):
        y_pred = model.predict(X_test_global)
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(name)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Confusion matrices (test set)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{outdir}/confusion_matrices.png", dpi=150)
    plt.close(fig)


X_test_global = None  # used by plot_confusion_matrices


def plot_roc_curves(results: dict, y_test, classes, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (name, model) in zip(axes.ravel(), results.items()):
        y_prob = model.predict_proba(X_test_global)
        for i, cls in enumerate(classes):
            binary = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(binary, y_prob[:, i])
            ax.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(name)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("One-vs-Rest ROC curves (test set)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{outdir}/roc_curves.png", dpi=150)
    plt.close(fig)


def plot_snr_accuracy(results: dict, X_test, y_test, snr_test, outdir):
    plt.figure(figsize=(8, 5))
    snr_vals = np.sort(np.unique(snr_test))
    for name, model in results.items():
        accs = []
        y_pred = model.predict(X_test)
        for s in snr_vals:
            mask = snr_test == s
            accs.append(accuracy_score(y_test[mask], y_pred[mask]))
        plt.plot(snr_vals, accs, "o-", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Test accuracy")
    plt.title("Classification accuracy vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/accuracy_vs_snr.png", dpi=150)
    plt.close()


def plot_metric_bars(metrics_df, outdir):
    metrics = ["accuracy", "precision_macro", "recall_macro",
               "f1_macro", "roc_auc_ovr_macro"]
    df_plot = metrics_df.set_index("model")[metrics]
    ax = df_plot.plot(kind="bar", figsize=(11, 5), rot=20, colormap="viridis")
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (macro-averaged test metrics)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/metrics_comparison.png", dpi=150)
    plt.close()


def main():
    global X_test_global
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(f"{args.out}/figures", exist_ok=True)
    os.makedirs(f"{args.out}/tables", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(args.data)
    feature_cols = [c for c in df.columns if c not in ("modulation",)]
    X = df[feature_cols].values
    y = df["modulation"].values
    snr = df["snr"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"classes={list(classes)}")

    # ------------------------------------------------------------------
    # 2. Split (stratified). SNR stays as a feature so stratification by
    #    label is enough to keep SNRs balanced between splits.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
        X, y_enc, snr, test_size=0.25, stratify=y_enc, random_state=RANDOM_STATE)
    X_test_global = X_test

    # ------------------------------------------------------------------
    # 3. Train / evaluate each model inside a scaling pipeline
    # ------------------------------------------------------------------
    metrics_rows, fitted = [], {}
    for name, clf in get_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        row = evaluate_model(name, pipe, X_test, y_test, classes)
        metrics_rows.append(row)
        print(f"{name:20s} acc={row['accuracy']:.4f} "
              f"F1={row['f1_macro']:.4f} AUC={row['roc_auc_ovr_macro']:.4f}")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        "accuracy", ascending=False)
    metrics_df.to_csv(f"{args.out}/tables/metrics_summary.csv", index=False)
    print("\n", metrics_df.round(4).to_string(index=False))

    # Per-model detailed classification report
    for name, pipe in fitted.items():
        rep = classification_report(y_test, pipe.predict(X_test),
                                    target_names=classes, digits=4)
        with open(f"{args.out}/tables/report_{name}.txt", "w") as f:
            f.write(rep)

    # ------------------------------------------------------------------
    # 4. Figures
    # ------------------------------------------------------------------
    fig_dir = f"{args.out}/figures"
    plot_metric_bars(metrics_df, fig_dir)
    plot_confusion_matrices(fitted, y_test, classes, fig_dir)
    plot_roc_curves(fitted, y_test, classes, fig_dir)
    plot_snr_accuracy(fitted, X_test, y_test, snr_test, fig_dir)

    # ------------------------------------------------------------------
    # 5. Persist a run summary for the README
    # ------------------------------------------------------------------
    summary = {"dataset": args.data,
               "n_samples": int(X.shape[0]),
               "features": feature_cols,
               "classes": list(classes),
               "metrics": metrics_df.round(4).to_dict(orient="records")}
    with open(f"{args.out}/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved under {args.out}/")


if __name__ == "__main__":
    main()
