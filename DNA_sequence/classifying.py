import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('dna_sequences.csv')

# Preprocess the data
X = data['sequence'].values
y = data['label'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert DNA sequences to numerical format
def encode_sequences(sequences):
    """Encodes DNA sequences into numerical format for model input."""
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    return [[mapping.get(base, 0) for base in seq] for seq in sequences]

X_train_encoded = encode_sequences(X_train)
X_test_encoded = encode_sequences(X_test)

# Pad sequences to ensure uniform input size
max_length = max(len(seq) for seq in X_train_encoded)
X_train_padded = pad_sequences(X_train_encoded, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_encoded, maxlen=max_length, padding='post', truncating='post')

# Build the LSTM model with improvements
model = Sequential()
model.add(Embedding(input_dim=5, output_dim=128, input_length=max_length))
model.add(LSTM(128, return_sequences=True))
model.add(BatchNormalization())  # Batch normalization for stable training
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the model with improvements
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks with improved patience and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(X_train_padded, y_train, 
                    epochs=50,   # Increased epochs for more training opportunity
                    batch_size=32, 
                    validation_data=(X_test_padded, y_test),
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Predictions and evaluation
y_pred = (model.predict(X_test_padded) > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot training history with enhanced visuals
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
