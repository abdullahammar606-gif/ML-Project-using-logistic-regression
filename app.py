import streamlit as st
import pickle
import numpy as np

# Load trained Logistic Regression model
model = pickle.load(open("football_logistic_model.pkl", "rb"))

st.title("⚽ Fantasy Football Clean Sheet Predictor")

st.write("Predict whether a clean sheet will occur using Logistic Regression")

# User Inputs
minutes = st.number_input("Minutes Played", 0, 120, 90)
goals_conceded = st.number_input("Goals Conceded", 0, 10, 0)
saves = st.number_input("Saves", 0, 20, 3)
was_home = st.selectbox("Home Match?", [0, 1])
influence = st.number_input("Influence", 0.0)
creativity = st.number_input("Creativity", 0.0)
threat = st.number_input("Threat", 0.0)
ict_index = st.number_input("ICT Index", 0.0)
opponent_team = st.number_input("Opponent Team ID", 1)

if st.button("Predict"):
    X = np.array([[minutes, goals_conceded, saves, was_home,
                   influence, creativity, threat, ict_index, opponent_team]])

    prediction = model.predict(X)

    if prediction[0] == 1:
        st.success("✅ Clean Sheet Expected")
    else:
        st.error("❌ No Clean Sheet Expected")
