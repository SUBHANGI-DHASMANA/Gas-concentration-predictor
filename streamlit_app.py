import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle


df = pd.read_csv("20160930_203718.csv")

X = df[['Time (s)']].values
y = df[['R1 (MOhm)']].values

model = LinearRegression()
model.fit(X, y)

# Load the trained model
with open('regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

def predict(input_time):
    prediction = model.predict(np.array([[input_time]]))
    prediction = prediction.tolist()
    return prediction

def main():
    st.title('Gas Concentration Predictor')
    st.text('A chemical detection platform composed of 14 temperature-modulated metal oxide (MOX) gas sensors was exposed during 3 weeks to mixtures of carbon monoxide and humid synthetic air in a gas chamber.')

    time = sorted(df['Time (s)'])

    input_time = st.text_input('Enter time:')
    if st.button('Predict'):
        if input_time:
            prediction = predict(float(input_time))
            st.write('Prediction:', prediction)
        else:
            st.write('Please enter a valid time.')

if __name__ == "__main__":
    main()
