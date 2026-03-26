# 🧠 EmotionAI — Text Emotion Detector

A full-stack AI web app that detects emotions from text with 92% accuracy, built with Flask and scikit-learn.

🌐 **Live Demo:** [web-production-e813b.up.railway.app](https://web-production-e813b.up.railway.app)

---

## Features

- **Emotion Detection** — detects 8 emotions: joy, sadness, anger, fear, surprise, disgust, shame, neutral
- **Live Detection** — emotion updates as you type (debounced)
- **Voice Input** — speak instead of type (Chrome)
- **Sentence Timeline** — analyze a paragraph sentence by sentence
- **Confidence Scores** — see probability for each emotion
- **Writing Assistant** — tone checker with suggestions for emails & messages
- **Compare Texts** — side-by-side emotion comparison of two texts
- **Bulk CSV Analyzer** — upload a CSV, get predictions for every row
- **Mood Journal** — private daily entries with emotion tracking
- **Dashboard** — stats, emotion distribution chart, 28-day heatmap, trend line
- **History** — full analysis history stored per user
- **Wellness Check-in** — detects negative emotion streaks, suggests coping tips
- **Leaderboard** — top joyful users today
- **API Access** — personal API key for external integrations
- **Export CSV** — download your full history
- **Dark / Light mode**
- **User Auth** — register, login, sessions

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Backend | Python, Flask |
| ML Model | scikit-learn (LinearSVC + TF-IDF) |
| Frontend | Vanilla JS, Chart.js |
| Auth | Werkzeug password hashing |
| Deploy | Railway (auto-deploy from GitHub) |

---

## Model

- Dataset: 34,792 labeled text samples across 8 emotions
- Vectorizer: TF-IDF (word n-grams 1–3, 50k features)
- Classifier: LinearSVC with oversampling for class balance
- Accuracy: **92%**

---

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Start the server
python app.py
```

Visit `http://localhost:5000`

---

## API Usage

```bash
POST /predict
Headers: X-API-Key: your_key_here
Content-Type: application/json
Body: {"text": "I am so happy today!"}

Response:
{
  "emotion": "joy",
  "confidence": {"joy": 82.3, "sadness": 4.1, ...}
}
```

Get your API key from the dashboard → API Access tab.

---

## Project Structure

```
├── app.py                  # Flask app & routes
├── train.py                # Model training
├── predict.py              # CLI prediction
├── requirements.txt
├── Procfile                # Railway/Heroku deploy
├── model.pkl               # Trained model
├── emotion_dataset_raw.csv # Training data
└── templates/
    ├── login.html          # Login & register page
    └── dashboard.html      # Main dashboard
```

---

## Deploy

Deployed on Railway with auto-deploy from GitHub `main` branch.

```bash
# Or deploy manually
railway login
railway link
railway up
```

---

Made by [VishalRauto](https://github.com/VishalRauto)
