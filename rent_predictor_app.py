
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and prepare the data
df = pd.read_csv('TrueFinalData.csv')
df = df.drop(columns=['Address', '0', 'Latitude', 'Longitude', 'Census Tract', 'Traffic', 'SoundScore'], errors='ignore')

# Store unique options for dropdowns
city_options = df['City'].dropna().unique().tolist()
home_type_options = df['Home Type'].dropna().unique().tolist()

# One-hot encode categorical features
df = pd.get_dummies(df, columns=["City", "Home Type"], prefix=["City", "Home Type"], drop_first=True)
X = df.drop(columns=["Minimum Price"])
y = df["Minimum Price"]

# Train linear regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)

# Create sidebar inputs
st.title("Bay Area Rent Price Estimator")

selected_city = st.selectbox("Select City", city_options)
selected_home_type = st.selectbox("Select Home Type", home_type_options)
beds = st.number_input("Minimum Beds", min_value=0, max_value=10, value=2)
baths = st.number_input("Minimum Baths", min_value=0, max_value=10, value=1)
sqft = st.number_input("Square Footage", min_value=100, max_value=10000, value=800)
units = st.number_input("Number of Units in Building", min_value=1, max_value=1000, value=1)
noise = st.slider("Noise Pollution Level (1 = Low, 10 = High)", min_value=1, max_value=10, value=5)
pm25 = st.slider("Air Pollution Level (PM2.5, 1 = Low, 10 = High)", min_value=1, max_value=10, value=5)
poverty = st.slider("Poverty Rate (1 = Low, 10 = High)", min_value=1, max_value=10, value=5)
distance_school = st.number_input("Distance to School (mi)", min_value=0.0, max_value=10.0, value=1.0)
distance_hospital = st.number_input("Distance to Hospital (mi)", min_value=0.0, max_value=10.0, value=1.0)
distance_grocery = st.number_input("Distance to Grocery Store (mi)", min_value=0.0, max_value=10.0, value=1.0)

# Prepare user input as model feature vector
input_dict = {
    'Minimum Beds': beds,
    'Minimum Baths': baths,
    'Sqft': sqft,
    'Units': units,
    'Noise Pollution': noise,
    'PM2.5': pm25,
    'Poverty': poverty,
    'Distance to School': distance_school,
    'Distance to Hospital': distance_hospital,
    'Distance to Grocery Store': distance_grocery
}

# Add one-hot encoded features
for col in X.columns:
    if col.startswith("City_"):
        input_dict[col] = 1 if col == f"City_{selected_city}" else 0
    elif col.startswith("Home Type_"):
        input_dict[col] = 1 if col == f"Home Type_{selected_home_type}" else 0
    elif col not in input_dict:
        input_dict[col] = 0  # Default for missing columns

# Convert input to DataFrame
input_df = pd.DataFrame([input_dict])

# Make prediction
if st.button("Predict Rent"):
    prediction = model.predict(input_df)[0]
    st.subheader(f"Estimated Rent Price: ${int(prediction):,}")
