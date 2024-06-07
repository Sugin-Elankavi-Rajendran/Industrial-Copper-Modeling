import pandas as pd
import numpy as np
import re
import streamlit as st
import pickle
st.set_page_config(layout="wide")

# Title of the application
st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Industrial Copper Modeling Application</h1>
</div>
""", unsafe_allow_html=True)

# Define the tabs for different functionalities
tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])

with tab1:
    # Define the options for dropdown menus
    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    products = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

    with st.form("predict_price_form"):
        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            st.write(' ')
            status = st.selectbox("Status", status_options, key=1)
            item_type = st.selectbox("Item Type", item_type_options, key=2)
            country = st.selectbox("Country", sorted(country_options), key=3)
            application = st.selectbox("Application", sorted(application_options), key=4)
            product_ref = st.selectbox("Product Reference", products, key=5)
        with col3:
            st.write(f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max values are provided for reference.</h5>', unsafe_allow_html=True)
            quantity_tons = st.text_input("Enter Quantity Tons (Min: 611728, Max: 1722207579)")
            thickness = st.text_input("Enter Thickness (Min: 0.18, Max: 400)")
            width = st.text_input("Enter Width (Min: 1, Max: 2990)")
            customer = st.text_input("Customer ID (Min: 12458, Max: 30408185)")
            submit_button = st.form_submit_button(label="Predict Selling Price")

        # Validate inputs
        pattern = "^(?:\d+|\d*\.\d+)$"
        flag = all(re.match(pattern, i) for i in [quantity_tons, thickness, width, customer])

    if submit_button and not flag:
        st.write("Please enter valid numeric values with no spaces.")
    elif submit_button and flag:
        try:
            quantity_tons = float(quantity_tons)
            thickness = float(thickness)
            width = float(width)
            customer = float(customer)
        except ValueError:
            st.write("Please enter valid numeric values.")
        else:
            # Load the models and scalers
            with open("model.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
            with open('scaler.pkl', 'rb') as f:
                scaler_loaded = pickle.load(f)
            with open("t.pkl", 'rb') as f:
                t_loaded = pickle.load(f)
            with open("s.pkl", 'rb') as f:
                s_loaded = pickle.load(f)

            new_sample = np.array([[np.log(quantity_tons), application, np.log(thickness), width, country, customer, int(product_ref), item_type, status]])
            new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
            new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0,1,2,3,4,5,6]], new_sample_ohe, new_sample_be), axis=1)
            new_sample_scaled = scaler_loaded.transform(new_sample)
            new_pred = loaded_model.predict(new_sample_scaled)[0]
            st.write('## :green[Predicted Selling Price:] ', np.exp(new_pred))

with tab2:
    with st.form("predict_status_form"):
        col1, col2, col3 = st.columns([5, 1, 5])
        with col1:
            cquantity_tons = st.text_input("Enter Quantity Tons (Min: 611728, Max: 1722207579)")
            cthickness = st.text_input("Enter Thickness (Min: 0.18, Max: 400)")
            cwidth = st.text_input("Enter Width (Min: 1, Max: 2990)")
            ccustomer = st.text_input("Customer ID (Min: 12458, Max: 30408185)")
            cselling_price = st.text_input("Selling Price (Min: 1, Max: 100001015)")
        with col3:
            st.write(' ')
            citem_type = st.selectbox("Item Type", item_type_options, key=21)
            ccountry = st.selectbox("Country", sorted(country_options), key=31)
            capplication = st.selectbox("Application", sorted(application_options), key=41)
            cproduct_ref = st.selectbox("Product Reference", products, key=51)
            csubmit_button = st.form_submit_button(label="Predict Status")

        cflag = all(re.match(pattern, k) for k in [cquantity_tons, cthickness, cwidth, ccustomer, cselling_price])

    if csubmit_button and not cflag:
        st.write("Please enter valid numeric values with no spaces.")
    elif csubmit_button and cflag:
        try:
            cquantity_tons = float(cquantity_tons)
            cthickness = float(cthickness)
            cwidth = float(cwidth)
            ccustomer = float(ccustomer)
            cselling_price = float(cselling_price)
        except ValueError:
            st.write("Please enter valid numeric values.")
        else:
            with open("cmodel.pkl", 'rb') as file:
                cloaded_model = pickle.load(file)
            with open('cscaler.pkl', 'rb') as f:
                cscaler_loaded = pickle.load(f)
            with open("ct.pkl", 'rb') as f:
                ct_loaded = pickle.load(f)

            new_sample = np.array([[np.log(cquantity_tons), np.log(cselling_price), capplication, np.log(cthickness), cwidth, ccountry, int(ccustomer), int(cproduct_ref), citem_type]])
            new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0,1,2,3,4,5,6,7]], new_sample_ohe), axis=1)
            new_sample_scaled = cscaler_loaded.transform(new_sample)
            new_pred = cloaded_model.predict(new_sample_scaled)
            if new_pred == 1:
                st.write('## :green[The Status is Won]')
            else:
                st.write('## :red[The Status is Lost]')

# Footer with creator information
st.write(f'<h6 style="color:rgb(0, 153, 153,0.35);">App Created by Sugin Elankavi</h6>', unsafe_allow_html=True)
