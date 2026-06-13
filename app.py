
import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('iris_model.pkl', 'rb'))

st.title("Iris Flower Species Prediction")

st.write("Enter flower measurements:")

sepal_length = st.number_input("Sepal Length (cm)")
sepal_width = st.number_input("Sepal Width (cm)")
petal_length = st.number_input("Petal Length (cm)")
petal_width = st.number_input("Petal Width (cm)")

if st.button("Predict"):

    data = np.array([[sepal_length,
                      sepal_width,
                      petal_length,
                      petal_width]])

    prediction = model.predict(data)

    species = ["Setosa", "Versicolor", "Virginica"]

    st.success(
        f"Predicted Species: {species[prediction[0]]}"
    )
