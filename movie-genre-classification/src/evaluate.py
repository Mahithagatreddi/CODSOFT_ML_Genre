from sklearn.metrics import classification_report, f1_score, accuracy_score
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and prints metrics.
    """
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    print("-" * 30)
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    
    return y_pred, report
