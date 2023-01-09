import streamlit as st
import numpy as np
import joblib
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
        clonesize = st.slider("Choose the clone-size: ",10,50,random.uniform(10,50),format="%d")
        bumbles = st.slider("Choose Bumbles density:  ",0,1,random.uniform(0.01,0.9),format="%f")
        andrena = st.slider("Choose Andrena density: ",0,1,random.uniform(0.01,0.9),format="%f")
        osmia = st.slider("Choose Osmia density:  ",0,1,random.uniform(0.01,0.9),format="%f")
        RainingDays = st.slider("Select number of raining days: ",5,15,random.uniform(5,15),format="%d")
        Average_rainingDays = st.slider("Average number of raining days: ",0,1,random.uniform(0.1,0.9),format="%f")

        submit = st.form_submit_button("Predict")
        if submit:
            data = np.array([clonesize,honey,bumbles,andrena,osmia,RainingDays,Average_rainingDays]).reshape(1,-1)
            pred = get_predictions(data=data,model=model)
            simulation = pred
            st.write(f"Pollination is: {simulation}")

if __name__ == "__main__":
    main()
