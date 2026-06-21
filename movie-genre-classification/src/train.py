from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import joblib
import os

def train_model(X_train, y_train, model_type='logistic_regression'):
    """
    Trains a classification model on the extracted features.
    """
    print(f"Training {model_type} model...")
    
    if model_type == 'logistic_regression':
        # class_weight='balanced' helps with the imbalanced nature of movie genres
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
    else:
        raise ValueError("Unsupported model_type. Use 'logistic_regression' or 'naive_bayes'.")

    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def train_model_gridsearch(X_train, y_train, model_type='logistic_regression'):
    """
    Trains a model using GridSearchCV to find the best hyperparameters.
    """
    print(f"Starting Hyperparameter Tuning for {model_type}...")
    
    if model_type == 'logistic_regression':
        base_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        param_grid = {
            'C': [0.1, 1.0, 10.0] # Regularization parameter
        }
    else:
        raise ValueError("GridSearchCV currently only configured for 'logistic_regression'.")

    # Use 3-fold cross-validation to save time (default is 5)
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation Macro F1-score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def save_model(model, vectorizer, model_path="models/model.joblib", vectorizer_path="models/vectorizer.joblib"):
    """
    Saves the trained model and vectorizer to disk.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
