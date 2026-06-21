import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.data_loader import load_data
from src.preprocess import preprocess_data, get_tfidf_features
from src.train import train_model, train_model_gridsearch, save_model
from src.evaluate import evaluate_model

# Make deep learning imports optional so standard ML models still work without TensorFlow
try:
    from src.train_dl import train_dl_model, predict_dl_model
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description="Movie Genre Classification Pipeline")
    parser.add_argument('--model', type=str, default='logistic_regression', 
                        choices=['logistic_regression', 'tuned_logistic_regression', 'naive_bayes', 'deep_learning'],
                        help="Model type to train")
    args = parser.parse_args()

    data_path = 'data/train_data.txt'
    print(f"=== Movie Genre Classification Pipeline ({args.model}) ===")
    
    # 1. Load Data
    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Preprocess Data
    df = preprocess_data(df)
    
    print("Splitting data into training and validation sets...")
    X = df['clean_description'].tolist()
    y = df['genre'].tolist()
    
    # Label encoding for Deep Learning
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    # We will split twice: once with string labels (for traditional) and once with encoded labels (for DL)
    X_train_text, X_val_text, y_train_str, y_val_str = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    _, _, y_train_enc, y_val_enc = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training samples: {len(X_train_text)}")
    print(f"Validation samples: {len(X_val_text)}")
    print(f"Number of distinct genres: {num_classes}")
    
    if args.model in ['logistic_regression', 'tuned_logistic_regression', 'naive_bayes']:
        # 3. TF-IDF Feature Extraction
        df_train = pd.DataFrame({'clean_description': X_train_text})
        df_val = pd.DataFrame({'clean_description': X_val_text})
        vectorizer, X_train_feat, X_val_feat = get_tfidf_features(df_train, df_val, max_features=10000)
        
        # 4. Train Traditional Model (Using string labels)
        if args.model == 'tuned_logistic_regression':
            model = train_model_gridsearch(X_train_feat, y_train_str, model_type='logistic_regression')
        else:
            model = train_model(X_train_feat, y_train_str, model_type=args.model)
            
        # 5. Evaluate Traditional Model
        evaluate_model(model, X_val_feat, y_val_str)
        
        # 6. Save Traditional Model
        save_model(model, vectorizer, model_path=f"models/{args.model}.joblib")
        
    elif args.model == 'deep_learning':
        if not DL_AVAILABLE:
            print("Error: TensorFlow is not installed. Please install it using 'pip install tensorflow' to use the deep learning model.")
            return
            
        # 3 & 4. Train Deep Learning Model
        model = train_dl_model(X_train_text, y_train_enc, X_val_text, y_val_enc, num_classes=num_classes)
        
        # 5. Evaluate DL Model
        print("Evaluating Deep Learning Model...")
        from sklearn.metrics import classification_report, accuracy_score, f1_score
        y_pred = predict_dl_model(model, X_val_text)
        
        accuracy = accuracy_score(y_val_enc, y_pred)
        f1 = f1_score(y_val_enc, y_pred, average='macro')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val_enc, y_pred, target_names=label_encoder.classes_, zero_division=0))
        
        # 6. Save DL Model
        # model.save('models/dl_model.keras')
        print("DL Model trained successfully. (Saving DL models omitted for simplicity, but can be added).")

    print("=== Pipeline Execution Finished ===")

if __name__ == "__main__":
    main()
