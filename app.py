from flask import Flask, request, jsonify, render_template, redirect, url_for, session, Response
from werkzeug.security import generate_password_hash, check_password_hash
import pickle, json, os, numpy as np, csv, io, secrets, re
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "emotion-secret-2025")

# ── Model ─────────────────────────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)
model      = bundle["model"]
vectorizer = bundle["vectorizer"]

USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE): return {}
    with open(USERS_FILE) as f: return json.load(f)

def save_users(u):
    with open(USERS_FILE, "w") as f: json.dump(u, f)

def predict_text(text):
    vec     = vectorizer.transform([text])
    scores  = model.decision_function(vec)[0]
    e       = np.exp(scores - scores.max())
    probs   = e / e.sum()
    confidence = {c: round(float(p)*100, 1) for c, p in zip(model.classes_, probs)}

    # ── Mixed emotion override ────────────────────────────────────────────────
    # When text has strong positive words alongside negative ones,
    # boost the positive signal so it isn't drowned out by single negative words.
    POSITIVE_WORDS = {"happy","happiness","joy","joyful","excited","love","great",
                      "amazing","wonderful","fantastic","glad","pleased","delighted",
                      "awesome","good","excellent","thrilled","blessed","grateful",
                      "cheerful","elated","ecstatic","proud","enjoy","enjoyed"}
    NEGATIVE_WORDS = {"sad","sadness","unhappy","depressed","miserable","terrible",
                      "awful","horrible","hate","angry","fear","scared","shame",
                      "disgusted","worried","anxious","upset","hurt","pain","cry"}

    words = set(re.findall(r"[a-z']+", text.lower()))
    pos_hits = len(words & POSITIVE_WORDS)
    neg_hits = len(words & NEGATIVE_WORDS)

    # If positive words dominate or tie, lean toward joy
    if pos_hits > 0 and pos_hits >= neg_hits:
        emotion = "joy"
    elif neg_hits > pos_hits and neg_hits >= 1:
        emotion = model.predict(vec)[0]
    else:
        emotion = model.predict(vec)[0]

    return emotion, confidence

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/")
def root():
    return redirect(url_for("dashboard") if "user" in session else url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        data  = request.get_json()
        users = load_users()
        user  = users.get(data.get("username"))
        if user and check_password_hash(user["password"], data.get("password","")):
            session["user"] = data["username"]
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": "Invalid username or password"}), 401
    return render_template("login.html")

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username","").strip()
    password = data.get("password","").strip()
    if not username or not password:
        return jsonify({"ok": False, "error": "Username and password required"}), 400
    users = load_users()
    if username in users:
        return jsonify({"ok": False, "error": "Username already taken"}), 400
    users[username] = {"password": generate_password_hash(password), "history": [], "journal": [], "api_key": secrets.token_hex(16)}
    save_users(users)
    session["user"] = username
    return jsonify({"ok": True})

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session: return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["user"])

# ── Core predict ──────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    # support API key auth
    api_key = request.headers.get("X-API-Key")
    user = None
    if api_key:
        users = load_users()
        for u, d in users.items():
            if d.get("api_key") == api_key:
                user = u; break
        if not user:
            return jsonify({"error": "Invalid API key"}), 401
    elif "user" in session:
        user = session["user"]

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "provide 'text' field"}), 400
    text = data["text"]
    emotion, confidence = predict_text(text)

    if user:
        users = load_users()
        entry = {"text": text, "emotion": emotion, "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "confidence": confidence}
        users[user].setdefault("history", []).insert(0, entry)
        users[user]["history"] = users[user]["history"][:100]
        save_users(users)

    return jsonify({"text": text, "emotion": emotion, "confidence": confidence})

# ── Bulk CSV analyze ──────────────────────────────────────────────────────────
@app.route("/bulk", methods=["POST"])
def bulk():
    if "user" not in session: return jsonify({"error": "login required"}), 401
    file = request.files.get("file")
    if not file: return jsonify({"error": "no file"}), 400
    content = file.read().decode("utf-8")
    reader  = csv.reader(io.StringIO(content))
    results = []
    for i, row in enumerate(reader):
        if i == 0 and row and row[0].lower() in ("text","sentence","content"): continue
        if not row: continue
        text = row[0].strip()
        if not text: continue
        emotion, confidence = predict_text(text)
        results.append({"text": text, "emotion": emotion, "top_confidence": max(confidence.values())})
    return jsonify(results)

# ── Rewrite suggestion ────────────────────────────────────────────────────────
REWRITES = {
    "anger":   ["Consider softening the tone: try 'I feel concerned about...' instead of accusatory language.",
                "Replace strong negative words with neutral ones to sound more professional."],
    "sadness": ["Add a hopeful closing line to balance the tone.",
                "Try starting with a positive acknowledgment before the concern."],
    "fear":    ["Use confident language — replace 'I'm worried' with 'I'd like to address'.",
                "State the issue clearly and propose a solution to sound more in control."],
    "disgust": ["Remove evaluative language and stick to facts.",
                "Try: 'I noticed X, which I'd like to discuss' instead of expressing strong reactions."],
    "shame":   ["Own the situation with confidence: 'I made a mistake and here's how I'll fix it'.",
                "Avoid over-apologizing — one clear apology is more effective."],
    "surprise":["Clarify your reaction and follow with a clear next step.",
                "Channel the surprise into curiosity: 'I'd love to understand more about...'"],
    "joy":     ["Great tone! Consider adding a specific call to action to make it actionable.",
                "Your positive energy comes through — make sure the key message is clear."],
    "neutral": ["Add a warm opening line to make it more engaging.",
                "Consider ending with a question to invite dialogue."],
}

@app.route("/rewrite", methods=["POST"])
def rewrite():
    if "user" not in session: return jsonify({"error": "login required"}), 401
    data = request.get_json()
    text = data.get("text","")
    emotion, confidence = predict_text(text)
    suggestions = REWRITES.get(emotion, [])
    return jsonify({"emotion": emotion, "confidence": confidence, "suggestions": suggestions})

# ── Compare two texts ─────────────────────────────────────────────────────────
@app.route("/compare", methods=["POST"])
def compare():
    if "user" not in session: return jsonify({"error": "login required"}), 401
    data = request.get_json()
    t1, t2 = data.get("text1",""), data.get("text2","")
    e1, c1 = predict_text(t1)
    e2, c2 = predict_text(t2)
    return jsonify({"text1": {"text": t1, "emotion": e1, "confidence": c1},
                    "text2": {"text": t2, "emotion": e2, "confidence": c2}})

# ── Journal ───────────────────────────────────────────────────────────────────
@app.route("/journal", methods=["GET","POST"])
def journal():
    if "user" not in session: return jsonify({"error": "login required"}), 401
    users = load_users()
    if request.method == "POST":
        data  = request.get_json()
        entry = data.get("entry","").strip()
        if not entry: return jsonify({"error": "empty entry"}), 400
        emotion, confidence = predict_text(entry)
        record = {"entry": entry, "emotion": emotion, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
        users[session["user"]].setdefault("journal", []).insert(0, record)
        users[session["user"]]["journal"] = users[session["user"]]["journal"][:60]
        save_users(users)
        return jsonify({"ok": True, "emotion": emotion, "confidence": confidence})
    return jsonify(users[session["user"]].get("journal", []))

# ── Weekly report (CSV export) ────────────────────────────────────────────────
@app.route("/export")
def export():
    if "user" not in session: return redirect(url_for("login"))
    users = load_users()
    history = users[session["user"]].get("history", [])
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Time", "Emotion", "Text"])
    for h in history:
        writer.writerow([h["time"], h["emotion"], h["text"]])
    output.seek(0)
    return Response(output, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment;filename=emotion_report_{session['user']}.csv"})

# ── API key ───────────────────────────────────────────────────────────────────
@app.route("/apikey")
def apikey():
    if "user" not in session: return jsonify({"error": "login required"}), 401
    users = load_users()
    key = users[session["user"]].get("api_key")
    if not key:
        key = secrets.token_hex(16)
        users[session["user"]]["api_key"] = key
        save_users(users)
    return jsonify({"api_key": key})

@app.route("/apikey/regenerate", methods=["POST"])
def regen_apikey():
    if "user" not in session: return jsonify({"error": "login required"}), 401
    users = load_users()
    key = secrets.token_hex(16)
    users[session["user"]]["api_key"] = key
    save_users(users)
    return jsonify({"api_key": key})

# ── History & leaderboard ─────────────────────────────────────────────────────
@app.route("/history")
def history():
    if "user" not in session: return jsonify([])
    users = load_users()
    return jsonify(users.get(session["user"], {}).get("history", []))

@app.route("/leaderboard")
def leaderboard():
    users = load_users()
    today = datetime.now().strftime("%Y-%m-%d")
    board = []
    for uname, udata in users.items():
        joy_today = sum(1 for h in udata.get("history",[]) if h["emotion"]=="joy" and h["time"].startswith(today))
        if joy_today > 0:
            board.append({"user": uname, "joy_count": joy_today})
    board.sort(key=lambda x: x["joy_count"], reverse=True)
    return jsonify(board[:10])

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
