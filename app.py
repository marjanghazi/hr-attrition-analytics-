from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model, encoders, and columns
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
model_columns = joblib.load("model_columns.pkl")

# ---------- NORMALIZATION MAPS ----------
NORMALIZE = {
    "BusinessTravel": {
        "notravel": "Non-Travel",
        "no travel": "Non-Travel",
        "nontravel": "Non-Travel",
        "frequently": "Travel_Frequently",
        "travel frequently": "Travel_Frequently",
        "rarely": "Travel_Rarely",
        "travel rarely": "Travel_Rarely"
    },
    "Department": {
        "human_resources": "Human Resources",
        "human resources": "Human Resources",
        "hr": "Human Resources",
        "research and development": "Research & Development",
        "research_and_development": "Research & Development",
        "rnd": "Research & Development",
        "sales": "Sales"
    }
}

def normalize_value(col, value):
    key = value.lower().replace("_", " ").strip()
    return NORMALIZE.get(col, {}).get(key, value)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        row = {}
        for col in model_columns:
            value = request.form.get(col)
            if value is None:
                return f"Missing field: {col}"

            # Handle categorical columns
            if col in encoders:
                value = normalize_value(col, value)
                encoder = encoders[col]

                if value not in encoder.classes_:
                    return (
                        f"Invalid value '{value}' for {col}. "
                        f"Allowed: {list(encoder.classes_)}"
                    )
                value = encoder.transform([value])[0]
            else:
                value = float(value)

            row[col] = value

        df = pd.DataFrame([row])
        prediction = model.predict(df)[0]

        result = "Yes, Employee Will Leave" if prediction == 1 else "No, Employee Will Stay"
        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
