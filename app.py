import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title("🌸 Iris Flower Prediction App")

# Description
st.write("Enter the flower measurements below and predict the Iris species!")

# Taking input from user
sepal_length = st.text_input('Sepal Length (cm)')
sepal_width = st.text_input('Sepal Width (cm)')
petal_length = st.text_input('Petal Length (cm)')
petal_width = st.text_input('Petal Width (cm)')

# Predict button
if st.button('Predict'):
    try:
        # Convert inputs to float
        features = np.array([[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Decode prediction
        species = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        
        st.success(f'Prediction: {species[prediction[0]]}')
    
    except ValueError:
        st.error('⚠️ Please enter valid numerical values!')