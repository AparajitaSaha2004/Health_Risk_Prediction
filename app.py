from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("diabetes_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["Age"])

    gender = 1 if request.form["Gender"] == "Male" else 0
    polyuria = 1 if request.form["Polyuria"] == "Yes" else 0
    polydipsia = 1 if request.form["Polydipsia"] == "Yes" else 0
    sudden_weight_loss = 1 if request.form["sudden weight loss"] == "Yes" else 0
    weakness = 1 if request.form["weakness"] == "Yes" else 0
    polyphagia = 1 if request.form["Polyphagia"] == "Yes" else 0
    genital_thrush = 1 if request.form["Genital thrush"] == "Yes" else 0
    visual_blurring = 1 if request.form["visual blurring"] == "Yes" else 0
    itching = 1 if request.form["Itching"] == "Yes" else 0
    irritability = 1 if request.form["Irritability"] == "Yes" else 0
    delayed_healing = 1 if request.form["delayed healing"] == "Yes" else 0
    partial_paresis = 1 if request.form["partial paresis"] == "Yes" else 0
    muscle_stiffness = 1 if request.form["muscle stiffness"] == "Yes" else 0
    alopecia = 1 if request.form["Alopecia"] == "Yes" else 0
    obesity = 1 if request.form["Obesity"] == "Yes" else 0

    input_data = pd.DataFrame([[
        age,
        gender,
        polyuria,
        polydipsia,
        sudden_weight_loss,
        weakness,
        polyphagia,
        genital_thrush,
        visual_blurring,
        itching,
        irritability,
        delayed_healing,
        partial_paresis,
        muscle_stiffness,
        alopecia,
        obesity
    ]], columns=[
        "Age",
        "Gender",
        "Polyuria",
        "Polydipsia",
        "sudden weight loss",
        "weakness",
        "Polyphagia",
        "Genital thrush",
        "visual blurring",
        "Itching",
        "Irritability",
        "delayed healing",
        "partial paresis",
        "muscle stiffness",
        "Alopecia",
        "Obesity"
    ])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Prediction probability
    probability = model.predict_proba(input_data)[0][1] * 100
    probability = round(probability, 2)

    if prediction == 1:
        result = "Positive"
        message = "The model predicts a higher likelihood of diabetes based on the information provided."
    else:
        result = "Negative"
        message = "The model predicts a lower likelihood of diabetes based on the information provided."

    return render_template(
        "index.html",
        prediction=result,
        probability=probability,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)