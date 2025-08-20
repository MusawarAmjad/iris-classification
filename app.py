from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('model/iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/iris_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Scale the features
    scaled_features = scaler.transform(final_features)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    
    # Get probability
    probability = model.predict_proba(scaled_features)
    max_probability = round(np.max(probability) * 100, 2)
    
    species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    result = species_map[np.argmax(probability)]
    
    return render_template('index.html', 
                           prediction_text=f'Predicted Species: {result}',
                           probability_text=f'Confidence: {max_probability}%',
                           feature_names=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                           feature_values=features)

if __name__ == '__main__':
    app.run(debug=True)