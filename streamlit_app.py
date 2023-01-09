import streamlit as st
import numpy as np
import joblib
import random
from sklearn.ensemble import RandomForestClassifier



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
        bumbles = st.slider("Choose Bumbles density:  ",0.0,1.0,random.uniform(0.01,0.9),format="%f")
        andrena = st.slider("Choose Andrena density: ",0.0,1.0,random.uniform(0.01,0.9),format="%f")
        osmia = st.slider("Choose Osmia density:  ",0.0,1.0,random.uniform(0.01,0.9),format="%f")
        RainingDays = st.slider("Select number of raining days: ",5,15,format="%d")
        Average_rainingDays = st.slider("Average number of raining days: ",0.0,1.0,random.uniform(0.1,0.9),format="%f")
        fruitset = st.slider("Select the set of fruits: ",0.0,1.0,format="%d")
        fruitmass = st.slider("Select mass of fruit: ",0.0,1.0,format="%f")
        seeds = st.slider("Select number of seeds: ",20,50,format="%f")

        submit = st.form_submit_button("Predict")
        if submit:
            data = np.array([clonesize,bumbles,andrena,osmia,RainingDays,Average_rainingDays]).reshape(1,-1)
            pred = get_predictions(data=data,model=model)
            simulation = pred
            st.write(f"Pollination is: {simulation}")

if __name__ == "__main__":
    main()
