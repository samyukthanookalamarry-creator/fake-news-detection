# ================================
# Fake News Detection - Prediction
# ================================

import pickle

# Load saved model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_news(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    
    return "Fake News" if prediction == 0 else "Real News"

# Example test
if __name__ == "__main__":
    sample = "Breaking news: Government announces new policy"
    print("Prediction:", predict_news(sample))
