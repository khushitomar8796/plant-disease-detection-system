print("APP STARTING...")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import json
from datetime import datetime
from flask import Flask, render_template, request
import gdown

print("IMPORTS DONE")

# ✅ Download model from Google Drive (FIXED LINK)
if not os.path.exists("plant_model.keras"):
    print("DOWNLOADING MODEL...")
    url = "https://drive.google.com/uc?id=1_2zo9RgbkuyZu0XH4zBULUBBGh_BBWWs"
    gdown.download(url, "plant_model.keras", quiet=False)

from tensorflow.keras.models import load_model

print("LOADING MODEL...")
model = load_model("plant_model.keras", compile=False)
print("MODEL LOADED SUCCESSFULLY")

app = Flask(__name__)

# -------------------------------
# CLASS LABELS
# -------------------------------
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

# -------------------------------
# SOLUTIONS + PESTICIDES
# -------------------------------
solutions = {
    "Pepper__bell___Bacterial_spot": "Avoid overhead watering and remove infected leaves.",
    "Pepper__bell___healthy": "Your plant is healthy.",
    "Potato___Early_blight": "Remove infected leaves.",
    "Potato___Late_blight": "Avoid excess moisture.",
    "Potato___healthy": "Healthy plant.",
    "Tomato_Bacterial_spot": "Use clean tools.",
    "Tomato_Early_blight": "Rotate crops.",
    "Tomato_Late_blight": "Avoid wet conditions.",
    "Tomato_Leaf_Mold": "Improve ventilation.",
    "Tomato_Septoria_leaf_spot": "Prune leaves.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Wash leaves.",
    "Tomato__Target_Spot": "Maintain hygiene.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies.",
    "Tomato__Tomato_mosaic_virus": "Remove plant.",
    "Tomato_healthy": "Healthy plant."
}

pesticides = {
    "Pepper__bell___Bacterial_spot": "Copper fungicide",
    "Pepper__bell___healthy": "None",
    "Potato___Early_blight": "Mancozeb",
    "Potato___Late_blight": "Metalaxyl",
    "Potato___healthy": "None",
    "Tomato_Bacterial_spot": "Copper spray",
    "Tomato_Early_blight": "Mancozeb",
    "Tomato_Late_blight": "Chlorothalonil",
    "Tomato_Leaf_Mold": "Chlorothalonil",
    "Tomato_Septoria_leaf_spot": "Mancozeb",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Neem oil",
    "Tomato__Target_Spot": "Azoxystrobin",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Imidacloprid",
    "Tomato__Tomato_mosaic_virus": "Remove plant",
    "Tomato_healthy": "None"
}

# -------------------------------
# HISTORY FILE
# -------------------------------
if not os.path.exists("history.json"):
    with open("history.json", "w") as f:
        json.dump([], f)

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    predicted_label = class_names[class_index]
    confidence = float(round(np.max(prediction) * 100, 2))

    solution = solutions.get(predicted_label)
    pesticide = pesticides.get(predicted_label)

    with open("history.json", "r") as f:
        history = json.load(f)

    history.append({
        "label": predicted_label,
        "confidence": confidence,
        "image": filepath,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

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
    healthy = sum(1 for h in history if "healthy" in h["label"].lower())
    diseased = total - healthy

    return render_template(
        "dashboard.html",
        total=total,
        healthy=healthy,
        diseased=diseased,
        history=history[::-1]
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)