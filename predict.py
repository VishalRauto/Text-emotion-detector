import pickle
import sys

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
vectorizer = bundle["vectorizer"]

def predict(text: str) -> str:
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter text: ")
    emotion = predict(text)
    print(f"Emotion: {emotion}")
