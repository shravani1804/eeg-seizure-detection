import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import json

# -------------------------------
# Load and prepare data
# -------------------------------
df = pd.read_csv('eeg_epilepsy_dataset_400.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,   # 🔥 IMPORTANT
    random_state=42
)

# -------------------------------
# Build model
# -------------------------------
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# -------------------------------
# Train model
# -------------------------------
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=32,
    verbose=1,
    validation_split=0.2,
    callbacks=[es]
)

# -------------------------------
# Save model and preprocessing tools
# -------------------------------
model.save('eeg_lstm.h5')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

# -------------------------------
# Create output directory
# -------------------------------
os.makedirs('static/plots', exist_ok=True)

# -------------------------------
# Plot Training & Validation Accuracy
# -------------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('static/plots/accuracy.png', bbox_inches='tight')
plt.close()


# -------------------------------
# Plot Training & Validation Loss
# -------------------------------
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('static/plots/loss.png', bbox_inches='tight')  
plt.close()

# -------------------------------
# Predictions & Confusion Matrix
# -------------------------------
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(8,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='plasma', xticks_rotation=45, values_format='d', ax=plt.gca())
plt.title('Confusion Matrix')
plt.savefig('static/plots/confusion_matrix.png', bbox_inches='tight')
plt.close()

# -------------------------------
# Classification Metrics
# -------------------------------
precision = precision_score(y_test, y_pred_labels, average='weighted')
recall = recall_score(y_test, y_pred_labels, average='weighted')
f1 = f1_score(y_test, y_pred_labels, average='weighted')

# Classification report (per-class)
report = classification_report(
    y_test, y_pred_labels,
    target_names=le.classes_,
    output_dict=True
)
report_df = pd.DataFrame(report).transpose()

# Save classification report as heatmap
plt.figure(figsize=(10,6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f")
plt.title("Classification Report (Precision, Recall, F1-score)")
plt.savefig("static/plots/classification_report.png", bbox_inches='tight')
plt.close()

# Save overall weighted metrics as bar chart
plt.figure(figsize=(6,4))
metrics = {"Precision": precision, "Recall": recall, "F1-score": f1}
plt.bar(metrics.keys(), metrics.values(), color=['skyblue','orange','green'])
plt.ylim(0,1)
for i, (k,v) in enumerate(metrics.items()):
    plt.text(i, v+0.02, f"{v:.2f}", ha='center', fontsize=10)
plt.title("Overall Weighted Metrics")
plt.ylabel("Score")
plt.savefig("static/plots/overall_metrics.png", bbox_inches='tight')
plt.close()

print("✅ Training complete. Plots and metrics saved in static/plots/")
