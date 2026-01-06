import streamlit as st
import pickle
import numpy as np

# Load model + scaler
with open("decision_tree_app.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]

st.title("⚽ Fantasy Football Clean Sheet Predictor")

st.write("Decision Tree Model")

# User Inputs
minutes = st.number_input("Minutes Played", 0, 120)
goals_conceded = st.number_input("Goals Conceded", 0, 10)
saves = st.number_input("Saves", 0, 20)
was_home = st.selectbox("Home Match?", [0, 1])
influence = st.number_input("Influence", 0.0)
creativity = st.number_input("Creativity", 0.0)
threat = st.number_input("Threat", 0.0)
ict_index = st.number_input("ICT Index", 0.0)
opponent_team = st.number_input("Opponent Team ID", 1)

if st.button("Predict Clean Sheet"):
    X = np.array([[minutes, goals_conceded, saves, was_home,
                   influence, creativity, threat, ict_index, opponent_team]])
    
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)

    if prediction[0] == 1:
        st.success("✅ Clean Sheet Expected")
    else:
        st.error("❌ No Clean Sheet")
