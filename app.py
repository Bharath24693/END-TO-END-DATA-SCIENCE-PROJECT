from flask import Flask, request, render_template
import pickle
import numpy as np

# Load model
model = pickle.load(open('iris_model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    species = ['Setosa', 'Versicolor', 'Virginica']
    output = species[prediction[0]]

    return render_template('index.html', prediction_text=f'Iris Species is: {output}')

if __name__ == "__main__":
    app.run(debug=True)
