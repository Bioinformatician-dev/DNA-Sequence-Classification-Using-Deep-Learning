import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
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
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    return [[mapping[base] for base in seq] for seq in sequences]

X_train_encoded = encode_sequences(X_train)
X_test_encoded = encode_sequences(X_test)

# Pad sequences to ensure uniform input size
max_length = max(len(seq) for seq in X_train_encoded)
X_train_padded = pad_sequences(X_train_encoded, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_encoded, maxlen=max_length, padding='post')

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=5, output_dim=64, input_length=max_length))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))  # Dropout layer for regularization
model.add(LSTM(32))
model.add(Dropout(0.2))  # Another dropout layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train the model
history = model.fit(X_train_padded, y_train, 
                    epochs=10, 
                    batch_size=32, 
                    validation_data=(X_test_padded, y_test),
                    callbacks=[early_stopping, model_checkpoint])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Predictions and evaluation
y_pred = (model.predict(X_test_padded) > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
