import streamlit as st
import torch
from predictor import predict_diabetes

# Streamlit user interface
st.title('Diabetes Prediction ')

# Input fields
pregnancies = st.number_input('Number of Pregnancies', min_value=0, value=0)
glucose = st.number_input('Glucose Level', min_value=0, value=120)
blood_pressure = st.number_input('Blood Pressure', min_value=0, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, value=20)
insulin = st.number_input('Insulin', min_value=0, value=30)
bmi = st.number_input('BMI', min_value=0.0, value=20.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.1)
age = st.number_input('Age', min_value=18, value=30)

# Prediction button
if st.button('Predict Diabetes'):
    input_data = torch.tensor([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], dtype=torch.float32)
    prediction = predict_diabetes(input_data)
    st.write(f'Chances of diabetes: {prediction.item()*100 } %')
