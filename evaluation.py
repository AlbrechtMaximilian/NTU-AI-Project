import torch
import json
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as sk_auc
import numpy as np
import matplotlib.pyplot as plt


def plotTrainingAccuracy():
    with open("training_metrics.json") as f:
        metrics = json.load(f)

    plt.plot(metrics["train_accuracy"], label="Train Acc")
    plt.plot(metrics["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.grid()
    plt.show()

def plotConfusionMatrix(cm):
    class_names = ['AS', 'MR', 'MS', 'MVP', 'N']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_names)
    
    # 6. Plot it
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)   # you can pass fmt='d' for integer formatting
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
def compute_binary_metrics(cm, class_names=['AS', 'MR', 'MS', 'MVP', 'N']):
    n = len(class_names)
    metrics = {}
    total = cm.sum()
    for i, label in enumerate(class_names):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - (TP + FP + FN)

        TPR = TP / (TP + FN) if (TP + FN) else 0.0            # sensitivity / recall
        FPR = FP / (FP + TN) if (FP + TN) else 0.0            # fall-out
        specificity = TN / (TN + FP) if (TN + FP) else 0.0
        precision = TP / (TP + FP) if (TP + FP) else 0.0      # positive predictive value
        NPV = TN / (TN + FN) if (TN + FN) else 0.0            # negative predictive value
        F1 = 2 * (precision * TPR) / (precision + TPR) if (precision + TPR) else 0.0

        metrics[label] = {
            "TP":       TP,
            "FP":       FP,
            "FN":       FN,
            "TN":       TN,
            "Sensitivity (TPR)":    TPR,
            "Specificity":           specificity,
            "False Positive Rate":   FPR,
            "Precision (PPV)":       precision,
            "Negative PV":           NPV,
            "F1-score":              F1
        }
    return metrics

def printBinaryMetrics(cm):
    class_names = ['AS','MR','MS','MVP','N']
    metrics = compute_binary_metrics(cm, class_names)
    for cls, m in metrics.items():
        print(f"\n— {cls} —")
        for name, val in m.items():
            print(f"{name:20s}: {val:.3f}")

def plotROC(targets, probs, class_names):
    y_true_bin = label_binarize(targets, classes=list(range(len(class_names))))
    
    n_classes = y_true_bin.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc[i] = sk_auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), 
        probs.ravel()
    )
    roc_auc["micro"] = sk_auc(fpr["micro"], tpr["micro"])
    
    # collect all FPRs
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # interpolate TPRs at those points & average
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = sk_auc(fpr["macro"], tpr["macro"])
    
    plt.figure(figsize=(8,6))
    # micro & macro
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'micro-average (AUC = {roc_auc["micro"]:.2f})',
            linestyle=':', linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"],
            label=f'macro-average (AUC = {roc_auc["macro"]:.2f})',
            linestyle=':', linewidth=2)

    # each class
    for i, name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], 
                label=f'{name} (AUC = {roc_auc[i]:.2f})')

    # chance line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc='lower right')
    plt.show()
    
    
if __name__ == "__main__":
    # Load saved data
    preds = torch.load("val_preds.pt").numpy()
    targets = torch.load("val_targets.pt").numpy()
    probs = torch.load("val_probs.pt").numpy()
    class_names=['AS', 'MR', 'MS', 'MVP', 'N']

    confusionMatrix = confusion_matrix(targets, preds)

    # Compute metrics
    print("Accuracy:", accuracy_score(targets, preds))
    print("Confusion Matrix:\n", confusionMatrix)

    # AUC (macro)
    auc = roc_auc_score(targets, probs, multi_class='ovr')
    print("Macro AUC:", auc)

    # plotTrainingAccuracy()
    # plotConfusionMatrix(confusion_matrix(targets, preds))
    # printBinaryMetrics(confusionMatrix)
    plotROC(targets, probs, class_names)