from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__, template_folder="templates")

# load trained model
model = joblib.load("fraud_model.pkl")

# ---------------- HOME PAGE ----------------
@app.route('/')
def home():
    print("Home page opened")
    return render_template("home.html")

# ---------------- INDEX PAGE ----------------
@app.route('/index')
def index():
    return render_template("index.html")

# ---------------- PREDICTION ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(request.form)

        step = float(request.form['step'])
        type_val = float(request.form['type'])
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
        isFlaggedFraud = float(request.form['isFlaggedFraud'])

        data = np.array([[step, type_val, amount, oldbalanceOrg,
                          newbalanceOrig, oldbalanceDest,
                          newbalanceDest, isFlaggedFraud]])

        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "Fraud Transaction ❌"
        else:
            result = "Legitimate Transaction ✅"

        return render_template("submit.html", prediction=result)

    except Exception as e:
        print("ERROR:", e)
        return render_template("submit.html", prediction="Error in input")
    
if __name__ == "__main__":
    app.run(debug=True)

