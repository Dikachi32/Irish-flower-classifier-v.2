from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models (Ensure these files are in your project folder)
try:
    dt_model = joblib.load("iris_decision_tree_model.pkl")
    rf_model = joblib.load("iris_random_forest_model.pkl")
    gb_model = joblib.load("iris_gradient_boosting_model.pkl")
    ensemble_model = joblib.load("iris_ensemble_model.pkl")
except Exception as e:
    print(f"Error loading models: {e}")

# ✅ FIXED: image filenames now match actual files in static/species_images/
#    Iris-setosa.jpg | Iris-versicolor.jpg | Iris-virginica.jpg
species_dict = {
    0: {
        "name": "Iris Setosa",
        "image": "Iris-setosa.jpg",
        "description": "A small and delicate iris species with narrow petals.",
        "origin": "North America"
    },
    1: {
        "name": "Iris Versicolor",
        "image": "Iris-versicolor.jpg",
        "description": "A medium-sized iris with beautiful purple-blue tones.",
        "origin": "Eastern United States"
    },
    2: {
        "name": "Iris Virginica",
        "image": "Iris-virginica.jpg",
        "description": "The largest iris species with vibrant violet petals.",
        "origin": "North America"
    }
}

@app.route('/')
def home():
    return render_template("decision_tree.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        sepal_length = float(request.form['sepal_length'])
        sepal_width  = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width  = float(request.form['petal_width'])
        model_choice = request.form['model']

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Select model based on dropdown choice
        if model_choice == "dt":
            model      = dt_model
            model_name = "Decision Tree"
        elif model_choice == "rf":
            model      = rf_model
            model_name = "Random Forest"
        elif model_choice == "gb":
            model      = gb_model
            model_name = "Gradient Boosting"
        else:
            model      = ensemble_model
            model_name = "Ensemble Model"

        # Predict
        prediction = model.predict(features)[0]
        prediction = int(round(prediction))

        species = species_dict[prediction]

        return render_template('result.html',
            species=species,
            selected_model=model_name,
            sepal_length=request.form['sepal_length'],
            sepal_width=request.form['sepal_width'],
            petal_length=request.form['petal_length'],
            petal_width=request.form['petal_width']
        )

    except Exception as e:
        return render_template("decision_tree.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)