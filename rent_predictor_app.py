import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and clean dataset
df = pd.read_csv('TrueFinalData.csv')
df = df.drop(columns=['Address', '0', 'Latitude', 'Longitude', 'Census Tract', 'Traffic', 'SoundScore'], errors='ignore')

# Combine MULTIUNIT and MULTI_FAMILY into Multi-Family
df['Home Type'] = df['Home Type'].replace({'MULTIUNIT': 'MULTI_FAMILY'})

# Keep dropdown options before encoding
city_options = sorted(df['City'].dropna().unique())
home_type_options = {
    'APARTMENT': 'Apartment',
    'SINGLE_FAMILY': 'Single-Family',
    'TOWNHOUSE': 'Townhouse',
    'MULTI_FAMILY': 'Multi-Family',
    'CONDO': 'Condo'
}

# Save scale ranges
scale_ranges = {
    "Noise Pollution": (df["Noise Pollution"].min(), df["Noise Pollution"].max()),
    "PM2.5": (df["PM2.5"].min(), df["PM2.5"].max()),
    "Poverty": (df["Poverty"].min(), df["Poverty"].max()),
}

# One-hot encode
df = pd.get_dummies(df, columns=["City", "Home Type"], prefix=["City", "Home Type"], drop_first=True)
X = df.drop(columns=["Minimum Price"])
y = df["Minimum Price"]

# Remove outliers from training set
model_temp = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model_temp.fit(X_train, y_train)
residuals = y_train - model_temp.predict(X_train)
z_scores = np.abs((residuals - residuals.mean()) / residuals.std())
X_train_no_outliers = X_train[z_scores < 3]
y_train_no_outliers = y_train[z_scores < 3]

# Final model without outliers
model = LinearRegression()
model.fit(X_train_no_outliers, y_train_no_outliers)

# Streamlit UI
st.title("Bay Area Rent Price Estimator")

selected_city = st.selectbox("Select City", city_options)
selected_home_type_label = st.selectbox("Select Home Type", list(home_type_options.values()))
selected_home_type = [k for k, v in home_type_options.items() if v == selected_home_type_label][0]

bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1)
sqft = st.number_input("Square Footage", min_value=100, max_value=10000, value=800)
units = st.number_input("Number of Units in Building", min_value=1, max_value=1000, value=1)

noise_scaled = st.slider("Noise Pollution (1 = Low, 10 = High)", 1, 10, 5)
pm25_scaled = st.slider("Air Pollution (PM2.5) (1 = Low, 10 = High)", 1, 10, 5)
poverty_scaled = st.slider("Poverty Rate (1 = Low, 10 = High)", 1, 10, 5)

distance_school = st.number_input("Distance to School (mi)", min_value=0.0, max_value=10.0, value=1.0)
distance_hospital = st.number_input("Distance to Hospital (mi)", min_value=0.0, max_value=10.0, value=1.0)
distance_grocery = st.number_input("Distance to Grocery Store (mi)", min_value=0.0, max_value=10.0, value=1.0)

# Reverse-scale back to original ranges
def scale_to_raw(val, col):
    min_val, max_val = scale_ranges[col]
    return min_val + (val - 1) / 9 * (max_val - min_val)

input_data = {
    "Minimum Beds": bedrooms,
    "Minimum Baths": bathrooms,
    "Sqft": sqft,
    "Units": units,
    "Noise Pollution": scale_to_raw(noise_scaled, "Noise Pollution"),
    "PM2.5": scale_to_raw(pm25_scaled, "PM2.5"),
    "Poverty": scale_to_raw(poverty_scaled, "Poverty"),
    "Distance to School": distance_school,
    "Distance to Hospital": distance_hospital,
    "Distance to Grocery Store": distance_grocery,
}

# Add one-hot encoded city and home type
for col in X.columns:
    if col.startswith("City_"):
        input_data[col] = 1 if col == f"City_{selected_city}" else 0
    elif col.startswith("Home Type_"):
        input_data[col] = 1 if col == f"Home Type_{selected_home_type}" else 0
    elif col not in input_data:
        input_data[col] = 0  # for any missing column

input_df = pd.DataFrame([input_data])[X.columns]

# Predict rent
if st.button("Predict Rent"):
    prediction = model.predict(input_df)[0]
    st.subheader(f"Estimated Rent Price: ${int(prediction):,}")
