import streamlit as st
import pickle
import numpy as np

# loading the saved model
loaded_model = pickle.load(open('models/kmeans_model.pkl', 'rb'))

# title
st.title('Customer Segmentation using K-Means Clustering')
st.write('Predict customer category based on annual income and spending behavior.')

# taking user input
annual_income = st.number_input(
    'Annual Income',
    min_value=1.0,
    max_value=150.0
)

spending_score = st.number_input(
    'Spending Score',
    min_value=1.0,
    max_value=100.0
)

# prediction
if st.button('Predict Customer Segment'):

    input_data = np.array([[annual_income, spending_score]])

    prediction = loaded_model.predict(input_data)

    if prediction[0] == 0:
        result = 'Medium Income Medium Spending Customer'

    elif prediction[0] == 1:
        result = 'High Income High Spending Customer'

    elif prediction[0] == 2:
        result = 'High Income Low Spending Customer'

    elif prediction[0] == 3:
        result = 'Low Income Low Spending Customer'

    else:
        result = 'Low Income High Spending Customer'

    st.success(f'Predicted Customer Type: {result}')