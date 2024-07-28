import streamlit as st
import pickle

# Load the model, vectorizer, and label encoder
with open('model.pkl', 'rb') as model_file:
    model, vectorizer, le = pickle.load(model_file)

st.title('Disease Prediction based on Symptoms')

# Input symptoms from user
symptoms = st.text_input('Enter symptoms (comma-separated):')

if st.button('Predict'):
    # Process the input
    input_symptoms = [' '.join(symptoms.split(','))]
    input_vector = vectorizer.transform(input_symptoms)
    
    # Make prediction
    prediction = model.predict(input_vector)
    disease = le.inverse_transform(prediction)[0]
    
    st.write(f'Predicted Disease: {disease}')
