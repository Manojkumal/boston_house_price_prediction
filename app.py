import pickle
from flask import Flask,render_template,request,jsonify,url_for,app
import pandas as pd
import numpy as np

app = Flask(__name__)

# load the model
model = pickle.load(open("regression.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict_api",methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug=True)


