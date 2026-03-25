from flask import Flask, request, jsonify
import pickle

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
    return jsonify({"text": text, "emotion": emotion})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
