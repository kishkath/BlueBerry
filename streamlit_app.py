import streamlit as st
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import encoding


from prediction import get_predictions

model = joblib.load(r"model/rf.joblib")
st.set_page_config(
    page_title="Blueberry Pollination Prediction",
    layout='wide')

st.markdown("<h1 style='text-align: center;'>Bluberry Pollination Simulation</h1>",unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):
        st.subheader("Enter input for the required info: ")
        clonesize = st.slider("Choose the clone-size: ",10,50,format="%d")
        bumbles = st.slider("Choose Bumbles density:  ",0,1,format="%f")
        andrena = st.slider("Choose Andrena density: ",0,1,format="%f")
        osmia = st.slider("Choose Osmia density:  ",0,1,format="%f")
        RainingDays = st.slider("Select number of raining days: ",5,15,format="%d")
        Average_rainingDays = st.selectbox("Average number of raining days: ",0,1,format="%f")

        submit = st.form_submit_button("Predict")
        if submit:
            data = np.array([clonesize,honey,bumbles,andrena,osmia,RainingDays,Average_rainingDays]).reshape(1,-1)
            pred = get_predictions(data=data,model=model)
            accidentMapping = {2: 'Slight Injury', 1: 'Serious Injury', 0: 'Fatal Injury'}
            st.write(f"Prediction Severity is: {accidentMapping[pred[0]]}")

if __name__ == "__main__":
    main()