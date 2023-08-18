import streamlit as st
import joblib

st.title('Will you admitted or not to your desired university?')
Titanic_LR_Model = joblib.load("AdmissionDataset_Model.pkl")

image = 'Harvard_University.jpg'  
st.image(image, caption='Image from https://online-learning.harvard.edu/catalog/free')
name = st.text_input('Enter your name', '')

if name:
    st.success(f" **Hi {name}! This prediction is just for fun. For prediction, Please provide your desired university's ranking, GRE Scores, TOFEL Scores, CGPA if applicable.**")
    
    Uni_ranking = st.radio("Select your Desired University's Ranking: ", ("1", "2", "3", "4", "5"))
    if Uni_ranking == "1":
        ranking = 1
    elif Uni_ranking == "2":
        ranking = 2
    elif Uni_ranking == "3":
        ranking = 3
    elif Uni_ranking == "4":
        ranking = 4
    else:
        ranking = 5

    GRE_Scores = st.number_input('Enter your Your GRE Scores (290~340): ', min_value=290, max_value=340, value=290, step=1)
    TOFEL_Scores = st.number_input('Enter your Your TOFEL Scores (92~120): ', min_value=92, max_value=120, value=92, step=1)
    CGPA = st.number_input('Enter your Your High School or University GPA (6.8~10.0): ', min_value=6.8, max_value=10., value=6.8, step=1., format="%.2f")
    
    data = [ranking, GRE_Scores, TOFEL_Scores, CGPA]
    result = Titanic_LR_Model.predict([data])
    predict_btn = st.button("Predict")

    if predict_btn:
        if result == 0:
            st.success(f"**Based on the available data, {name}! You will not be admitted to your desired University**")
        else:
            st.success(f"**Based on the available data, {name}! You will be admitted to your desired University**")
