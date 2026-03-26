from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import pickle, json, os, numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "emotion-secret-2025")

# ── Model ─────────────────────────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)
model      = bundle["model"]
vectorizer = bundle["vectorizer"]

# ── Simple file-based user store ──────────────────────────────────────────────
USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE) as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route("/")
def root():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        users = load_users()
        user = users.get(data.get("username"))
        if user and check_password_hash(user["password"], data.get("password", "")):
            session["user"] = data["username"]
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": "Invalid username or password"}), 401
    return render_template("login.html")

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if not username or not password:
        return jsonify({"ok": False, "error": "Username and password required"}), 400
    users = load_users()
    if username in users:
        return jsonify({"ok": False, "error": "Username already taken"}), 400
    users[username] = {"password": generate_password_hash(password), "history": []}
    save_users(users)
    session["user"] = username
    return jsonify({"ok": True})

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["user"])

# ── Predict ───────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "provide 'text' field"}), 400
    text = data["text"]
    vec     = vectorizer.transform([text])
    emotion = model.predict(vec)[0]
    scores  = model.decision_function(vec)[0]
    e       = np.exp(scores - scores.max())
    probs   = e / e.sum()
    confidence = {c: round(float(p)*100, 1) for c, p in zip(model.classes_, probs)}

    # Save to user history
    if "user" in session:
        users = load_users()
        entry = {"text": text, "emotion": emotion, "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "confidence": confidence}
        users[session["user"]].setdefault("history", []).insert(0, entry)
        users[session["user"]]["history"] = users[session["user"]]["history"][:50]
        save_users(users)

    return jsonify({"text": text, "emotion": emotion, "confidence": confidence})

@app.route("/history")
def history():
    if "user" not in session:
        return jsonify([])
    users = load_users()
    return jsonify(users.get(session["user"], {}).get("history", []))

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
