from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def compute_classification_metrics(y_true, y_prob, y_pred=None, average="binary"):
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        m["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        m["auc"] = None
    cm = confusion_matrix(y_true, y_pred).tolist()
    return m, cm
