import streamlit as st  # type: ignore # streamlit run diamond-price-prediction1.py
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("diamonds.csv")
    if "Unnamed: 0" in data.columns:
        data.drop("Unnamed: 0", axis=1, inplace=True)
    data["size"] = data["x"] * data["y"] * data["z"]
    data["cut"] = data["cut"].map({"Ideal": 1, "Premium": 2, "Good": 3, "Very Good": 4, "Fair": 5})
    data["color"] = data["color"].map({"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7})
    data["clarity"] = data["clarity"].map({"SI1": 1, "SI2": 2, "VS1": 3, "VS2": 4,
                                           "VVS1": 5, "VVS2": 6, "I1": 7, "IF": 8})
    return data.dropna()

# Load and prepare data
data = load_data()
features = ["carat", "cut", "color", "clarity", "size"]
target = "price"
X = data[features]
y = data[target]

# Imputation and Train-Test Split
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Title
st.title("💎 Diamond Price Predictor")

# Sidebar Input
st.sidebar.header("Enter Diamond Features")
carat = st.sidebar.slider("Carat", 0.2, 5.0, 1.0)
cut = st.sidebar.selectbox("Cut", options=["Ideal", "Premium", "Good", "Very Good", "Fair"])
color = st.sidebar.selectbox("Color", options=["D", "E", "F", "G", "H", "I", "J"])
clarity = st.sidebar.selectbox("Clarity", options=["SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2", "I1", "IF"])
x = st.sidebar.number_input("x", min_value=1.0)
y = st.sidebar.number_input("y", min_value=1.0)
z = st.sidebar.number_input("z", min_value=1.0)

# Map inputs
cut_map = {"Ideal": 1, "Premium": 2, "Good": 3, "Very Good": 4, "Fair": 5}
color_map = {"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7}
clarity_map = {"SI1": 1, "SI2": 2, "VS1": 3, "VS2": 4, "VVS1": 5, "VVS2": 6, "I1": 7, "IF": 8}

size = x * y * z
input_features = np.array([[carat, cut_map[cut], color_map[color], clarity_map[clarity], size]])

# Prediction
predicted_price = model.predict(input_features)[0]
st.subheader("💰 Predicted Diamond Price")
st.write(f"**${predicted_price:,.2f}**")

# Optional: Add a graph
fig = px.scatter(data_frame=data, x="carat", y="price", color="cut",
                 title="Carat vs Price by Cut", trendline="ols", height=500)
st.plotly_chart(fig)

