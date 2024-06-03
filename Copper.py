import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import streamlit as st
import joblib

# Load dataset
data = pd.read_excel('Copper_Set.xlsx')

# Data Preprocessing
data['material_ref'] = data['material_ref'].apply(lambda x: np.nan if str(x).startswith('00000') else x)

# Handling Missing Values
# For numerical columns, fill missing values with mean
numerical_cols = ['quantity tons', 'thickness', 'width']
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

# For categorical columns, fill missing values with mode
categorical_cols = ['customer', 'country', 'item type', 'application', 'material_ref', 'product_ref']
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Encoding Categorical Variables using LabelEncoder
label_encoder = LabelEncoder()
for col in categorical_cols:
    if data[col].dtype == 'object':  # Check if the column contains string/object data
        data[col] = label_encoder.fit_transform(data[col])

# Exclude 'id' column from features for modeling
X_reg = data.drop(['selling_price', 'status', 'id'], axis=1)  # Exclude 'id' column
y_reg = data['selling_price']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_model = ExtraTreesRegressor()
reg_model.fit(X_train_reg, y_train_reg)
reg_mse = mean_squared_error(y_test_reg, reg_model.predict(X_test_reg))
joblib.dump(reg_model, 'regression_model.pkl')

X_clf = data.drop(['selling_price', 'status', 'id'], axis=1)  # Exclude 'id' column
y_clf = data['status']
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
clf_model = ExtraTreesClassifier()
clf_model.fit(X_train_clf, y_train_clf)
clf_accuracy = accuracy_score(y_test_clf, clf_model.predict(X_test_clf))
joblib.dump(clf_model, 'classification_model.pkl')

# Streamlit Application
def main():
    st.title('Copper Industry Prediction Tool')

    task = st.sidebar.selectbox('Select Task', ('Regression', 'Classification'))

    if task == 'Regression':
        st.subheader('Regression Prediction')
        st.write('Please enter the values to predict Selling Price:')
        quantity_tons = st.number_input('Quantity (tons)', min_value=0.0)
        thickness = st.number_input('Thickness', min_value=0.0)
        width = st.number_input('Width', min_value=0.0)

        # Prepare the input for prediction with the trained regression model
        reg_input = np.array([[quantity_tons, thickness, width]])

        # Load the saved regression model
        reg_model = joblib.load('regression_model.pkl')

        # Display predicted selling_price
        predicted_price = reg_model.predict(reg_input)
        st.write(f'Predicted Selling Price: {predicted_price[0]}')

    elif task == 'Classification':
        st.subheader('Classification Prediction')

        # Input fields for classification prediction using Streamlit
        st.write('Please enter the values for classification prediction:')
        # Add input fields for features used in classification prediction
        # Example input fields:
        feature_1 = st.number_input('Feature 1', min_value=0.0)
        feature_2 = st.number_input('Feature 2', min_value=0.0)
        feature_3 = st.number_input('Feature 3', min_value=0.0)

        # Prepare the input for prediction with the trained classification model
        clf_input = np.array([[feature_1, feature_2, feature_3]])  # Update with your features

        # Load the saved classification model
        clf_model = joblib.load('classification_model.pkl')

        # Perform classification prediction
        predicted_status = clf_model.predict(clf_input)

        # Display predicted status
        st.write(f'Predicted Status: {predicted_status[0]}')

if __name__ == '__main__':
    main()