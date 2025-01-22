import os
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))
car = pd.read_csv('Cleaned_Car.csv')

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    return render_template('index.html', companies=companies, car_models=car_models, year=year, fuel_type=fuel_type)

@app.route('/get_models', methods=['GET'])
def get_models():
    company = request.args.get('company')
    if company:
        models = car[car['company'] == company]['name'].unique()
        return jsonify(models)
    return jsonify([])

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    fuel_type = request.form.get('fuel_type')
    year = int(request.form.get('year'))
    kms_driven = int(request.form.get('kilo_driven'))
    
    print(f"Company: {company}, Car Model: {car_model}, Year: {year}, Fuel Type: {fuel_type}, KMS Driven: {kms_driven}")
    
    prediction_input = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    
    prediction = model.predict(prediction_input)
    
    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    # Get the port number from the environment variable, default to 5000 if not set
    port = int(os.environ.get("PORT", 5000))
    # Listen on all interfaces (0.0.0.0) and the specified port
    app.run(host='0.0.0.0', port=port, debug=True)
