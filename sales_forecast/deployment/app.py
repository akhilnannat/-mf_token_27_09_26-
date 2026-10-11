import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_sales_revenue_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("SuperKart Total Sales Prediction")
st.write("""
This application predicts the expected **Total Sales** of a SuperKart store
based on its characteristics such as product_type,Store Year,Store Type,etc..
Please enter the app details below to get a Sales prediction.
""")

# User input
Product_Sugar_Content= st.selectbox("Product_Sugar_Content",["Low Sugar","Regular","No Sugar","reg"])
Product_Type =st.selectbox("Product_Type",['Frozen Foods','Dairy','Canned','Baking Goods','Health and Hygiene','Snack Foods','Meat','Fruits and Vegetables','Breads','Starchy Foods'])
Store_Size = st.selectbox("Store_Size",['High','Medium','Small'])
Store_Location_City_Type = st.selectbox("Store_Location_City_Type",['Tier 1','Tier 2','Tier 3'])    
Store_Type= st.selectbox("Store_Type",['Supermarket Type1','Supermarket Type2','Departmental Store','Food Mart'])                     
Store_Id = st.selectbox("Store_Id",['OUT001','OUT002','OUT003','OUT004'])
                          
Product_Weight= st.number_input("Product_Weight",min_value=1.0, max_value=1000.0, value=50.0, step=0.1)
Product_Allocated_Area=st.number_input("Product_Allocated_Area(Ratio)",min_value=1.0, max_value=1000.0, value=50.0, step=0.1)
Product_MRP=st.number_input("Product_MRP",min_value=1.0, max_value=100000.0, value=50.0, step=0.1)
Store_Establishment_Year= st.number_input("Store_Establishment_Year",min_value=1.0, max_value=10000.0, value=50.0, step=1)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Type': Product_Type,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type,
    'Store_Id': Store_Id,
    'Product_Weight': Product_Weight,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_MRP': Product_MRP,
    'Store_Establishment_Year': Store_Establishment_Year
}])

# Predict button
if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Estimated Sales: **${prediction:,.2f} USD**")
