from flask import Flask, render_template, request
import random

app = Flask(__name__)

def predict_news(text):
    # Demo prediction logic
    fake_keywords = ["aliens", "conspiracy", "fake"]

    if any(word in text.lower() for word in fake_keywords):
        return "Fake News"
    else:
        return "Real News"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        news_text = request.form["news"]

        if news_text.strip() == "":
            prediction = "Please enter news text"
        else:
            prediction = predict_news(news_text)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)