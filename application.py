from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)
model = pickle.load(open("LinearRegressionModel.pkl",'rb'))
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
        # Filter the car models based on the selected company
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
    
    # Add your prediction logic here
    print(company, car_model, year, fuel_type, kms_driven)
    
    prediction  = model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    print(prediction[0])
    # Return a response after processing the form data
    return str(np.round(prediction[0],2))

if __name__ == '__main__':
    app.run(debug=True)
