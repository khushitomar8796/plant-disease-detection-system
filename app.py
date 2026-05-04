print("APP STARTING...")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hides tensorflow noise

import cv2
import numpy as np
import json
from datetime import datetime
from flask import Flask, render_template, request

print("IMPORTS DONE")

from tensorflow.keras.models import load_model

print("LOADING MODEL...")

model = load_model("plant_model.keras", compile=False)

print("MODEL LOADED SUCCESSFULLY")

app = Flask(__name__)

# Load trained model
model = load_model("plant_model.keras")

# Class labels
class_names = [
'Pepper__bell___Bacterial_spot',
'Pepper__bell___healthy',
'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',
'Tomato_Bacterial_spot',
'Tomato_Early_blight',
'Tomato_Late_blight',
'Tomato_Leaf_Mold',
'Tomato_Septoria_leaf_spot',
'Tomato_Spider_mites_Two_spotted_spider_mite',
'Tomato__Target_Spot',
'Tomato__Tomato_YellowLeaf__Curl_Virus',
'Tomato__Tomato_mosaic_virus',
'Tomato_healthy'
]

# General solutions
solutions = {
    "Pepper__bell___Bacterial_spot": "Avoid overhead watering and remove infected leaves.",
    "Pepper__bell___healthy": "Your plant is healthy. Maintain proper care.",
    "Potato___Early_blight": "Remove infected leaves and maintain soil health.",
    "Potato___Late_blight": "Avoid excess moisture and improve air circulation.",
    "Potato___healthy": "Your plant is healthy.",
    "Tomato_Bacterial_spot": "Avoid overhead irrigation and use clean tools.",
    "Tomato_Early_blight": "Remove affected leaves and rotate crops.",
    "Tomato_Late_blight": "Avoid wet conditions and monitor plant closely.",
    "Tomato_Leaf_Mold": "Reduce humidity and improve ventilation.",
    "Tomato_Septoria_leaf_spot": "Prune infected leaves and avoid splashing water.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Wash leaves and maintain humidity.",
    "Tomato__Target_Spot": "Remove infected parts and maintain hygiene.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control insect vectors like whiteflies.",
    "Tomato__Tomato_mosaic_virus": "Remove infected plants immediately.",
    "Tomato_healthy": "Your plant is healthy."
}

# 🌿 NEW: Pesticide Recommendations
pesticides = {
    "Pepper__bell___Bacterial_spot": "Copper-based fungicide (Spray every 7 days)",
    "Pepper__bell___healthy": "No pesticide required",

    "Potato___Early_blight": "Mancozeb fungicide (Spray every 7–10 days)",
    "Potato___Late_blight": "Metalaxyl fungicide (Apply early stage)",
    "Potato___healthy": "No pesticide required",

    "Tomato_Bacterial_spot": "Copper fungicide spray",
    "Tomato_Early_blight": "Mancozeb or Chlorothalonil",
    "Tomato_Late_blight": "Chlorothalonil or Metalaxyl",
    "Tomato_Leaf_Mold": "Chlorothalonil spray",
    "Tomato_Septoria_leaf_spot": "Mancozeb fungicide",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Neem oil spray",
    "Tomato__Target_Spot": "Azoxystrobin fungicide",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Imidacloprid (for whiteflies)",
    "Tomato__Tomato_mosaic_virus": "No chemical cure, remove plant",
    "Tomato_healthy": "No pesticide required"
}

# Ensure history file exists
if not os.path.exists("history.json"):
    with open("history.json", "w") as f:
        json.dump([], f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Image preprocessing
    img = cv2.imread(filepath)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))

    # Prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    predicted_label = class_names[class_index]
    confidence = float(round(np.max(prediction) * 100, 2))

    solution = solutions.get(predicted_label, "No recommendation available.")
    pesticide = pesticides.get(predicted_label, "No pesticide info available.")

    # Load history
    with open("history.json", "r") as f:
        history = json.load(f)

    # Add new entry
    history.append({
        "label": predicted_label,
        "confidence": confidence,
        "image": filepath,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    # Save history
    with open("history.json", "w") as f:
        json.dump(history, f)

    return render_template(
        'index.html',
        prediction=predicted_label,
        confidence=confidence,
        solution=solution,
        pesticide=pesticide,
        img_path=filepath
    )


@app.route('/dashboard')
def dashboard():
    try:
        with open("history.json", "r") as f:
            history = json.load(f)
    except:
        history = []

    total = len(history)

    healthy = sum(1 for h in history if "healthy" in str(h.get("label", "")).lower())
    diseased = total - healthy

    freq = {}
    for h in history:
        label = str(h.get("label", "Unknown"))
        freq[label] = freq.get(label, 0) + 1

    most_common = max(freq, key=freq.get) if freq else "N/A"

    # CLEAN DATA (IMPORTANT FIX)
    labels = [str(k) for k in freq.keys()]
    values = [int(v) for v in freq.values()]

    return render_template(
        "dashboard.html",
        total=int(total),
        healthy=int(healthy),
        diseased=int(diseased),
        most_common=str(most_common),
        history=history[::-1],
        chart_labels=labels,
        chart_values=values
    )
if __name__ == "__main__":
    app.run(debug=True, port=5001)