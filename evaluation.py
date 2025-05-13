import torch
import json
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# Load saved data
preds = torch.load("val_preds.pt").numpy()
targets = torch.load("val_targets.pt").numpy()
probs = torch.load("val_probs.pt").numpy()

# Compute metrics
print("Accuracy:", accuracy_score(targets, preds))
print("Confusion Matrix:\n", confusion_matrix(targets, preds))

# AUC (macro)
auc = roc_auc_score(targets, probs, multi_class='ovr')
print("Macro AUC:", auc)

# Example

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
