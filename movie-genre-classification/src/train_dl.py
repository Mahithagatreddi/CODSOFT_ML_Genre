import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, TextVectorization
import numpy as np

def train_dl_model(X_train_text, y_train, X_val_text=None, y_val=None, num_classes=1, max_tokens=10000, output_sequence_length=100):
    """
    Trains an LSTM deep learning model for text classification.
    Expects raw text strings, not TF-IDF vectors.
    """
    print("Building Deep Learning Model (LSTM)...")
    
    # 1. Text Vectorization Layer (Tokenization)
    vectorize_layer = TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=output_sequence_length
    )
    
    print("Adapting vocabulary on training data...")
    vectorize_layer.adapt(X_train_text)
    
    # 2. Build the model architecture
    model = Sequential([
        vectorize_layer,
        Embedding(input_dim=max_tokens, output_dim=64, mask_zero=True),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        # Assuming we encode labels to integers (0 to num_classes-1)
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    print(model.summary())
    
    # 3. Train the model
    print("Training Deep Learning model...")
    validation_data = (np.array(X_val_text), np.array(y_val)) if X_val_text is not None and y_val is not None else None
    
    # Convert input to numpy arrays
    X_train_np = np.array(X_train_text)
    y_train_np = np.array(y_train)
    
    model.fit(
        X_train_np, y_train_np,
        validation_data=validation_data,
        epochs=3, # Keep low for demonstration/speed
        batch_size=32
    )
    
    print("Deep Learning model training complete.")
    return model

def predict_dl_model(model, X_test_text):
    """
    Generates predictions from the DL model.
    """
    X_test_np = np.array(X_test_text)
    predictions = model.predict(X_test_np)
    
    if predictions.shape[1] > 1:
        # Multi-class
        predicted_classes = np.argmax(predictions, axis=-1)
    else:
        # Binary
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
    return predicted_classes
