from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def evaluate_model(model, loaders):
    y_true = []
    y_pred = []
    test_dataset = loaders[2]
    for x_batch, y_batch in test_dataset:
        preds = model.predict(x_batch)
        y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)

    print("Accuracy:", acc)
    print("F1 macro:", f1_macro)
    print("F1 weighted:", f1_weighted)