
import os
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_superkart_sales_model_v1.joblib"
)

model = joblib.load(model_path)

# Application title
st.title("SuperKart Sales Prediction")

st.write("""
This application predicts product store sales based on
product and store operational parameters.

Enter the details below to get a sales prediction.
""")

# Product details
product_weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    max_value=25.0,
    value=12.0,
    step=0.1
)

product_sugar_content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

product_mrp = st.number_input(
    "Product MRP",
    min_value=0.0,
    max_value=300.0,
    value=50.0,
    step=1.0
)

product_allocated_area = st.number_input(
    "Product Allocated Area",
    min_value=0.004,
    max_value=0.298,
    value=0.050,
    step=0.001,
    format="%.3f"
)

type_of_food = st.selectbox(
    "Type of Food",
    ["Perishable", "Non Perishable"]
)

# Store details
store_id = st.selectbox(
    "Store ID",
    ["OUT001", "OUT002", "OUT003", "OUT004"]
)

store_size = st.selectbox(
    "Store Size",
    ["Small", "Medium", "High"]
)

store_location_city_type = st.selectbox(
    "Store Location City Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

store_type = st.selectbox(
    "Store Type",
    [
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2"
    ]
)

store_age = st.number_input(
    "Store Age (years)",
    min_value=0,
    max_value=100,
    value=20
)

# Predict sales
if st.button("Predict Sales"):

    # Prepare input data
    input_data = pd.DataFrame([{
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "type_of_food": type_of_food,
        "Product_MRP": product_mrp,
        "Store_Id": store_id,
        "Store_Age": store_age,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type
    }])

    try:
        prediction = model.predict(input_data)

        st.success("Prediction completed successfully!")

        st.subheader("Predicted Product Store Sales")

        st.write(
            f"**Predicted Sales: {prediction[0]:.2f}**"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
