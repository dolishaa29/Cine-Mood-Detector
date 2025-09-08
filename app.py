from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorize = joblib.load("vectorize.pkl")

@app.route("/")
def index():
    return render_template("index.html") 

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    X = vectorize.transform([text])
    pred = model.predict(X)[0]

    return jsonify({"Sentiment": pred})

if __name__ == "__main__":
    app.run(debug=True)
