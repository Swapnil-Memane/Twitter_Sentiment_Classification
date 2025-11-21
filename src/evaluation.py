from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate(model, vectorizer, X_test, y_test):
    X_vec = vectorizer.transform(X_test)
    preds = model.predict(X_vec)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "report": classification_report(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds)
    }
