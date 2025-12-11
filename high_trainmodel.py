import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib
import os

# ---------------------------
# Step 1: Define Path
# ---------------------------
base_path = r"E:\dlproject"

# Load datasets
df1 = pd.read_csv(os.path.join(base_path, "songs_dataset1.csv"))
df2 = pd.read_csv(os.path.join(base_path, "songs_dataset2.csv"))
df = pd.concat([df1, df2], ignore_index=True)

# ---------------------------
# Step 2: Select Features & Target
# ---------------------------
features = [
    'danceability', 'energy', 'acousticness', 'valence', 'speechiness',
    'instrumentalness', 'liveness', 'tempo', 'loudness', 'duration_ms'
]

X = df[features]
y = df["track_popularity"].apply(lambda x: 1 if x >= 50 else 0)

# ---------------------------
# Step 3: Preprocessing
# ---------------------------
X = X.fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Step 4: Build MLP Model
# ---------------------------
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------
# Step 5: Train Model
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=1
)

# ---------------------------
# Step 6: Evaluate Accuracy on Test Set
# ---------------------------
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# ---------------------------
# Step 7: Save Model & Scaler
# ---------------------------
model.save(os.path.join(base_path, "song_popularity_mlp.h5"))
joblib.dump(scaler, os.path.join(base_path, "scaler.pkl"))
print("✅ Model and Scaler saved successfully at E:\\dlproject")

# ---------------------------
# Step 8: Example Prediction on New Data
# ---------------------------
new_song = pd.DataFrame([{
    "danceability": 0.7,
    "energy": 0.8,
    "acousticness": 0.2,
    "valence": 0.6,
    "speechiness": 0.05,
    "instrumentalness": 0.0,
    "liveness": 0.1,
    "tempo": 120.0,
    "loudness": -5.0,
    "duration_ms": 210000
}])

# Load saved scaler & model
loaded_scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
loaded_model = tf.keras.models.load_model(os.path.join(base_path, "song_popularity_mlp.h5"))

# Scale new song features
new_song_scaled = loaded_scaler.transform(new_song[features])

# Predict
prediction = (loaded_model.predict(new_song_scaled) > 0.5).astype(int)
print("Prediction (1=Hit, 0=Non-Hit):", prediction[0][0])
