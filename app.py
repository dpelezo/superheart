import pandas as pd
#import Numpy
import numpy as np
#import Flask
from flask import Flask, render_template, request
#import pickle
import pickle


#create an instance of Flask
app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        form_value = request.form.to_dict()
        features = list(form_value.values())
        data = list(map(int, features))
        df = np.array(data).reshape(1, 13)
        prediction = model.predict(df)
        output = prediction[0]
        if int(prediction) == 0:
            info = "Absence of HD"
            details ="Keep up the good work, Please stay Hydrated and practice exercise"
        else:
            info = "Presence of HD"
            details = "Please consult your doctor for further information and analyses. Note: This diagnostic is 70% accurate and trained with less data so don't take it to serious"

    return render_template('index.html', prediction_text=output, prediction_result=info, prediction_info=details)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
""""""
if __name__ == '__main__':

    app.run(debug=True)