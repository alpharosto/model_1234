import streamlit as st
import pandas as pd
import pickle

# Load the model and the mean/std values
model_filename = 'decision_tree_model(2).pkl'
mean_std_filename = 'mean_std_values.pkl.url'

# Add error handling for file loading
try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
except (FileNotFoundError, pickle.UnpicklingError) as e:
    st.error("Error loading model. Please check the file and try again.")
    model = None

try:
    with open(mean_std_filename, 'rb') as file:
        mean_std_values = pickle.load(file)
except (FileNotFoundError, pickle.UnpicklingError) as e:
    st.error("Error loading mean/std values. Please check the file and try again.")
    mean_std_values = None

# Define the main function for the Streamlit app
def main():
    st.title('Heart Disease Prediction')

    # Gather user inputs
    age = st.slider('Age', 18, 100, 50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    sex_num = 1 if sex == 'Male' else 0

    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    cp = st.selectbox('Chest Pain Type', cp_options)
    cp_num = cp_options.index(cp)

    trestbps = st.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.slider('Cholesterol', 100, 600, 250)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])
    fbs_num = 1 if fbs == 'True' else 0

    restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
    restecg = st.selectbox('Resting Electrocardiographic Results', restecg_options)
    restecg_num = restecg_options.index(restecg)

    thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, 150)
    exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
    exang_num = 1 if exang == 'Yes' else 0

    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    slope_num = slope.index(slope)

    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)
    thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    thal_num = thal.index(thal)

    if st.button('Predict'):
        if model and mean_std_values:
            # Prepare the user input
            user_input = pd.DataFrame({
                'age': [age],
                'sex': [sex_num],
                'cp': [cp_num],
                'trestbps': [trestbps],
                'chol': [chol],
                'fbs': [fbs_num],
                'restecg': [restecg_num],
                'thalach': [thalach],
                'exang': [exang_num],
                'oldpeak': [oldpeak],
                'slope': [slope_num],
                'ca': [ca],
                'thal': [thal_num]
            })

            # Apply normalization using mean/std
            user_input_normalized = (user_input - mean_std_values['mean']) / mean_std_values['std']

            # Predict and get probability
            prediction = model.predict(user_input_normalized)
            prediction_proba = model.predict_proba(user_input_normalized)

            # Display prediction result and confidence
            if prediction[0] == 1:
                bg_color = 'red'
                prediction_result = 'Positive'
            else:
                bg_color = 'green'
                prediction_result = 'Negative'

            st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction: {prediction_result}</p>", unsafe_allow_html=True)

        else:
            st.error("Model or mean/std values are not properly loaded.")

# Run the main function
if __name__ == '__main__':
    main()
