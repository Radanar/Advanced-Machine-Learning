import streamlit as st
import pickle 
import numpy as np

# Streamlit UI
st.title("🌸 Iris Flower Classification")
st.write("Welcome! Enter the flower's measurements to predict its species.")
st.write("Thynn Yadanar Su")

#Input fields
petal_length = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, step=0.1)

with open('knn_iris_model.pkl','rb') as f:
    loaded_model=pickle.load(f)
    
with open('knn_scaler.pkl','rb') as f:
    loaded_scaler=pickle.load(f)

# Prediction button
if st.button("Predict"):
    try:
        input_features=np.array([[petal_length, petal_width, sepal_length, sepal_width]])
        input_features = loaded_scaler.transform(input_features) 
        
        value=loaded_model.predict(input_features)[0]

        # Map predicted value to class names
        species_dict = {0: "Setosa 🌱", 1: "Versicolor 🌿", 2: "Virginica 🌺"}
        st.success(f"🌸 Predicted Species: {species_dict[value]}")

    except Exception as e:
        st.error(f"Error: {e}")
