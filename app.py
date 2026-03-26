from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

model      = bundle["model"]
vectorizer = bundle["vectorizer"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "provide 'text' field"}), 400
    text = data["text"]
    vec     = vectorizer.transform([text])
    emotion = model.predict(vec)[0]
    # softmax over decision scores for confidence
    scores = model.decision_function(vec)[0]
    e = np.exp(scores - scores.max())
    probs = e / e.sum()
    confidence = {c: round(float(p)*100, 1) for c, p in zip(model.classes_, probs)}
    return jsonify({"text": text, "emotion": emotion, "confidence": confidence})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
