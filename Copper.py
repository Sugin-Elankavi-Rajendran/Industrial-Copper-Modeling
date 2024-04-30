import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, LabelEncoder

#####

df = pd.read_excel("Copper_Set.xlsx")

df.drop(columns=['INDEX'], inplace=True, errors='ignore')

df['material_ref'] = df['material_ref'].apply(lambda x: np.nan if str(x).startswith('00000') else x)

continuous_cols = ['item_date', 'quantity tons', 'customer', 'country', 'application', 'thickness', 'width', 'product_ref', 'delivery date', 'selling_price']

categorical_cols = ['id', 'status', 'item type', 'material_ref']

for col in continuous_cols:
    print(df[col].describe())

for col in categorical_cols:
    print(df[col].value_counts())

#####
