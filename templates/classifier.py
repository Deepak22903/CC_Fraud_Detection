from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("fraud_detection_model.sav")

@app.route("/")
def home():
    return render_template("classifier.html")

@app.route("/", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    if prediction == 1:
        return "The transaction is fraudulent."
    else:
        return "The transaction is not fraudulent."

if __name__ == "__main__":
    app.run(debug=True)
