"""Evaluate model predictions with AUROC and AUPR, and optionally plot curves."""

from sklearn import metrics
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

def modeleval(y_true, y_pred, plot=True):
    """
    Evaluates model predictions with AUROC and AUPR, and optionally plots curves.

    Filters out invalid predictions labeled as "Invalid SMILES", then computes
    AUROC and Precision-Recall AUC. Optionally plots the PR and ROC curves.

    Parameters:
    y_true (list): List of ground-truth binary labels (0/1).
    y_pred (list): List of predicted scores (floats) or "Invalid SMILES".
    plot (bool, optional): Whether to display PR and ROC plots. Default is True.

    Returns:
    tuple: A tuple containing:
        - float: AUROC score.
        - float: Area under the Precision-Recall curve (AUPR).
        - list: Filtered ground-truth labels after removing invalid predictions.
        - list: Filtered predicted scores after removing invalid predictions.
    """
    new_ytrue = []
    new_ypred = []
    for x, y in zip(y_true, y_pred):
        if y != "Invalid SMILES":
            new_ytrue.append(x)
            new_ypred.append(y)
    y_true = new_ytrue
    y_pred = new_ypred

    auroc = float(roc_auc_score(y_true, y_pred))
    print("auroc: " + str(auroc))

    # Compute Precision-Recall and plot curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr = float(auc(recall, precision))
    print("precision recall: " + str(pr))

    if plot:
        fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
        plt.clf()
        plt.plot(recall, precision, label="Precision-recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.tight_layout()
        plt.show()

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
        plt.clf()
        plt.plot(fpr, tpr, label="ROC curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.tight_layout()
        plt.show()

    return (auroc, pr, y_true, y_pred)


def test_vs_true(data_path, data_file, model_path, model_results):
    """
    Compares model predictions to ground-truth labels and reports evaluation metrics.

    Reads true hit labels and predicted scores from CSV files, evaluates predictions
    using AUROC and AUPR, and computes top-100 recall and precision statistics.

    Parameters:
    data_path (str): Path to directory containing the ground-truth data file.
    data_file (str): Filename of the CSV file with true hit labels (column: 'hit').
    model_path (str): Path to directory containing the model results file.
    model_results (str): Filename of the CSV file with model predictions (column: 'hit').

    Returns:
    None: Prints evaluation metrics to stdout and generates plots via modeleval.
    """
    true = pd.read_csv(data_path + data_file)
    true = list(true["hit"])

    test = pd.read_csv(model_path + model_results)
    test = [float(x) if x != "Invalid SMILES" else x for x in list(test["hit"])]

    # all stats
    print("all stats")
    roc, pr, y_true, y_pred = modeleval(true, test)

    # top-100 recall
    print("top 100 predicted stats")
    top100pred = pd.DataFrame()
    top100pred["true"] = y_true
    top100pred["test"] = y_pred
    top100pred = top100pred.sort_values("test", ascending=False)
    top100pred = top100pred.iloc[0:100, :]
    top100pred_bin = [x > 0.5 for x in list(top100pred["test"])]
    print("recall: ")
    print(metrics.recall_score(top100pred["true"], top100pred_bin))
    print("precision: ")
    print(metrics.precision_score(top100pred["true"], top100pred_bin))
    _, _, _, _ = modeleval(top100pred["true"], top100pred["test"], plot=False)