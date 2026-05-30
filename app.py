import numpy as np
import joblib
import warnings
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore")

app = Flask(__name__)

data = joblib.load("loan_scoring_model.pkl")
model = data["model"]
scaler = data["scaler"]
le_employment = data["label_encoder_employment"]
le_decision = data["label_encoder_decision"]
EMPLOYMENT_TYPES = list(le_employment.classes_)

DECISION_LABELS = {
    "approved": {"mn": "Зээл олгогдлоо", "color": "green"},
    "manual_review": {"mn": "Гараар шалгах шаардлагатай", "color": "orange"},
    "rejected": {"mn": "Зээл татгалзагдлаа", "color": "red"},
}


def build_features(monthly_income, employment_years, requested_amount, employment_type):
    emp_encoded = int(le_employment.transform([employment_type])[0])
    ratio = requested_amount / monthly_income
    annual_dti = (requested_amount / (monthly_income * 12)) * 100
    log_income = np.log1p(monthly_income)
    log_amount = np.log1p(requested_amount)
    return np.array([[monthly_income, employment_years, requested_amount,
                      ratio, annual_dti, log_income, log_amount, emp_encoded]])


@app.route("/")
def index():
    return render_template("index.html", employment_types=EMPLOYMENT_TYPES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        monthly_income = float(request.form["monthly_income"])
        employment_years = float(request.form["employment_years"])
        requested_amount = float(request.form["requested_amount"])
        employment_type = request.form["employment_type"]

        if employment_type not in EMPLOYMENT_TYPES:
            return jsonify({"error": "Буруу ажил мэргэжил"}), 400
        if monthly_income <= 0 or requested_amount <= 0:
            return jsonify({"error": "Орлого болон зээлийн дүн 0-ээс их байх ёстой"}), 400

        X = build_features(monthly_income, employment_years, requested_amount, employment_type)
        X_scaled = scaler.transform(X)
        pred_encoded = model.predict(X_scaled)[0]
        probas = model.predict_proba(X_scaled)[0]
        decision = le_decision.inverse_transform([pred_encoded])[0]

        proba_map = {
            le_decision.classes_[i]: round(float(p) * 100, 1)
            for i, p in enumerate(probas)
        }

        return jsonify({
            "decision": decision,
            "label": DECISION_LABELS[decision]["mn"],
            "color": DECISION_LABELS[decision]["color"],
            "probabilities": proba_map,
        })

    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Оруулсан өгөгдөл буруу байна: {e}"}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
