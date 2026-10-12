from flask import Flask, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("rf_model.pkl")

@app.route("/webhook", methods=["POST"])
def webhook():

    data = request.json

    # extract answers
    features = [...]

    prediction = model.predict(
        np.array(features).reshape(1, -1)
    )[0]

    # send email here

    return "OK", 200