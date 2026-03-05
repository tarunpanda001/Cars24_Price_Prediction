import pandas as pd
import streamlit as st
import pickle
import numpy as np

st.title("Cars24 Price Prediction")
st.subheader("This is a simple web app to predict the price of a used car based on its features.")

# Load the dataset
cars_df = pd.read_csv("cars24-car-price-cleaned.csv")
# --- Prepare mappings from make/model strings to numeric values ---
make_mean_map = cars_df.groupby('make')['selling_price'].mean().to_dict()
model_mean_map = cars_df.groupby('model')['selling_price'].mean().to_dict()

# Display the Data
# st.dataframe(cars_df.head())

with open("car_price_model.pkl","rb") as f:
    Model = pickle.load(f)

# User Input
st.subheader("Enter the details of the car to predict its price:")

make = st.selectbox("Company of the Car", cars_df['make'].unique())
car_model = st.selectbox("Car Model", cars_df[cars_df['make']==make]['model'].unique())

# Convert user selections to numeric values
make_val = float(make_mean_map[make])
model_val = float(model_mean_map[car_model])


col1,col2 = st.columns(2)
with col1:
    year =st.number_input("Manfacturing Year",2000,2025)
    km_driven = st.number_input("Kilometers Driven",0,1000000)
    mileage = st.number_input("Mileage (km/L)",0.0,50.0)
with col2:
    engine = st.number_input("Engine (CC)",500,5000)
    max_power = st.number_input("Max Power (bhp)",10,500)
    age = st.number_input("Age of the Car (Years)",0,25)

fuel = st.selectbox("Fuel Type",["Petrol","Diesel","Electric","LPG"]) 
petrol = 1 if fuel == "Petrol" else 0
diesel = 1 if fuel == "Diesel" else 0
electric = 1 if fuel=="Electric" else 0
lpg = 1 if fuel=="LPG" else 0

seats = st.selectbox("Seats", ["5", ">5"])
seat_5 = 1 if seats=="5" else 0
seat_gt5 = 1 if seats==">5" else 0

transmission = st.selectbox("Transmission Type",["Manual","Automatic"])
manual = 1 if transmission == "Manual" else 0
automatic = 1 if transmission == "Automatic" else 0

seller = st.selectbox("Seller Type", ["Individual","Trustmark Dealer"])
individual = 1 if seller=="Individual" else 0
trustmark = 1 if seller=="Trustmark Dealer" else 0

# Prepare the input data for prediction
if st.button("Predict Price"):
    features = np.array([[year, km_driven, mileage, engine, max_power,
                      age, make_val, model_val, individual, trustmark,
                      diesel, electric, lpg, petrol, manual, seat_5, seat_gt5]])
    
    predicted_price = Model.predict(features)
    st.subheader(f"The predicted Price of the car is : ₹ {predicted_price[0]:.2f} Lakhs")

st.markdown("---")
st.markdown(
"<center>Developed by <b>Tarun Panda</b> | Cars24 Price Prediction</center>",
unsafe_allow_html=True
)