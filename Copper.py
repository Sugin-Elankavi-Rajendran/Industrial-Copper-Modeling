import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, LabelEncoder

#####

df = pd.read_excel("Copper_Set.xlsx")

# Drop irrelevant columns
df.drop(columns=['INDEX'], inplace=True, errors='ignore')

# Convert 'material_ref' column values starting with '00000' to NaN
df['material_ref'] = df['material_ref'].apply(lambda x: np.nan if str(x).startswith('00000') else x)

# Define continuous and categorical columns
continuous_cols = ['item_date', 'quantity tons', 'customer', 'country', 'application', 'thickness', 'width', 'product_ref', 'delivery date', 'selling_price']
categorical_cols = ['id', 'status', 'item type', 'material_ref']

# Display summary statistics for continuous columns
for col in continuous_cols:
    print(df[col].describe())

# Display value counts for categorical columns
for col in categorical_cols:
    print(df[col].value_counts())

#####

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].astype(str)

# Handle missing values for numeric columns using mean imputation
numeric_imputer = SimpleImputer(strategy='mean')
df_filled_numeric = pd.DataFrame(numeric_imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Handle missing values for categorical columns using mode imputation
categorical_imputer = SimpleImputer(strategy='most_frequent')
df_filled_categorical = pd.DataFrame(categorical_imputer.fit_transform(df[categorical_cols]), columns=categorical_cols)

# Concatenate the filled numeric and categorical columns
df_filled = pd.concat([df_filled_numeric, df_filled_categorical], axis=1)

# Apply log1p transformation to numeric columns
skewed_cols = df_filled.select_dtypes(include=['float64']).columns
for col in skewed_cols:
    df_filled[col] = np.log1p(df_filled[col])

# Preprocess categorical columns
categorical_cols = df_filled.select_dtypes(include=['object']).columns
# Limit the number of categories for one-hot encoding
top_categories = 10  # Set the number of top categories to include
for col in categorical_cols:
    top_categories_values = df_filled[col].value_counts().index[:top_categories]
    df_filled[col] = df_filled[col].apply(lambda x: x if x in top_categories_values else 'Other')
    df_filled = pd.get_dummies(df_filled, columns=[col], drop_first=True)


#####
