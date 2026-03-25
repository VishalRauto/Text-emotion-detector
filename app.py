from flask import Flask, request, jsonify, render_template
from sklearn.calibration import CalibratedClassifierCV
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
vectorizer = bundle["vectorizer"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "provide 'text' field"}), 400
    text = data["text"]
    vec = vectorizer.transform([text])
    emotion = model.predict(vec)[0]

    # confidence scores via decision function
    scores = model.decision_function(vec)[0]
    scores = scores - scores.min()
    total = scores.sum()
    probs = (scores / total).tolist() if total > 0 else [1/len(scores)] * len(scores)
    classes = model.classes_.tolist()
    confidence = {c: round(p * 100, 1) for c, p in zip(classes, probs)}

    return jsonify({"text": text, "emotion": emotion, "confidence": confidence})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
