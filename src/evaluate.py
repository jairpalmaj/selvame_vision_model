from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np


def evaluate_model(model, loaders, root_folder):
    output_dir = root_folder / "data"
    y_true = []
    y_pred = []
    test_dataset = loaders[2]
    for x_batch, y_batch in test_dataset:
        preds = model.predict(x_batch)
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1))

    class_names = loaders[3]
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_title("Matriz de confusion")
    axis.set_xlabel("Prediccion")
    axis.set_ylabel("Etiqueta real")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)

    print("Accuracy:", acc)
    print("F1 macro:", f1_macro)
    print("F1 weighted:", f1_weighted)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }