from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle


app = Flask(__name__)
df = pd.read_csv("20160930_203718.csv")

X = df[['Time (s)']].values
y = df[['R1 (MOhm)']].values

model = LinearRegression()
model.fit(X, y)

# model = joblib.load('regression_model.pkl')

with open('regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    time = sorted(df['Time (s)'])
    return render_template('index.html', time=time)

@app.route('/predict', methods=['POST'])
def predict():
    input_time = float(request.form['text'])
    prediction = model.predict(np.array([[input_time]]))
    prediction = prediction.tolist()
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
