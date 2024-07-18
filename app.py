from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('/home/ahsanejaz1227/lab_4/fish_species_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form
    weight = float(request.form['weight'])
    Length1 = float(request.form['Length1'])
    Length2 = float(request.form['Length2'])
    Length3 = float(request.form['Length3'])
    Height = float(request.form['Height'])
    Width = float(request.form['Width'])
    # Add other features as needed

    # Create a feature array
    features = np.array([[weight, Length1,Length2,Length3,Height,Width]])  # Add other features as needed
    
    # Scale the features
    features = scaler.transform(features)
    
    # Predict the species
    prediction = model.predict(features)
    
    # Return the prediction result
    return f'The predicted species is: {prediction[0]}'

if __name__ == '__main__':
    app.run(debug=True)
