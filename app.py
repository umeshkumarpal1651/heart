# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        anaemia = float(request.form['anaemia'])
        creatinine_phosphokinase = float(request.form['creatinine_phosphokinase'])
        diabetes = float(request.form['diabetes'])
        ejection_fraction = float(request.form['ejection_fraction'])
        high_blood_pressure = float(request.form['high_blood_pressure'])
        platelets = float(request.form['platelets'])
        serum_creatinine = float(request.form['serum_creatinine'])
        
        serum_sodium = float(request.form['serum_sodium'])
        sex = float(request.form['sex'])
        smoking = float(request.form['smoking'])
        time = float(request.form['time'])
        
        
        
        
        
        
        data = np.array([[age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction, high_blood_pressure,platelets,
                          serum_creatinine,serum_sodium,sex,smoking,time]])
        my_prediction = classifier.predict(data)
        
        return render_template('results.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)