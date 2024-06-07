import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("Copper_Set.xlsx")
# print(df.head(2))

# print(len(df['item_date'].unique())) 
# print(len(df['customer'].unique())) 
# print(len(df['material_ref'].unique()))
# print(len(df['product_ref'].unique())) 
# print(len(df['delivery date'].unique())) 

missing_values = df.isnull().sum()
# print(missing_values)

df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce', downcast='float')
df['customer'] = pd.to_numeric(df['customer'], errors='coerce', downcast='integer')
df['country'] = pd.to_numeric(df['country'], errors='coerce', downcast='integer')
df['application'] = pd.to_numeric(df['application'], errors='coerce', downcast='integer')
df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce', downcast='float')
df['width'] = pd.to_numeric(df['width'], errors='coerce', downcast='float')
df['material_ref'] = df['material_ref'].str.lstrip('0')
df['product_ref'] = pd.to_numeric(df['product_ref'], errors='coerce', downcast='integer')
df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce', downcast='float')

missing_values_count = df.isnull().sum()
# print(missing_values_count)

df['material_ref'] = df['material_ref'].fillna('unknown')
df.dropna(inplace=True)

missing_values_count = df.isnull().sum()
# print(missing_values_count)

df_sample = df.sample(frac=0.2, random_state=42)

df_sample['selling_price'] = df_sample['selling_price'].apply(lambda x: np.nan if x <= 0 else x)
df_sample['quantity tons'] = df_sample['quantity tons'].apply(lambda x: np.nan if x <= 0 else x)
df_sample['thickness'] = df_sample['thickness'].apply(lambda x: np.nan if x <= 0 else x)

df_sample['selling_price_log'] = np.log(df_sample['selling_price'])
df_sample['quantity tons_log'] = np.log(df_sample['quantity tons'])
df_sample['thickness_log'] = np.log(df_sample['thickness'])

# sns.histplot(df_sample['selling_price_log'], kde=True)
# plt.show()
# sns.histplot(df_sample['quantity tons_log'], kde=True)
# plt.show()
# sns.histplot(df_sample['thickness_log'], kde=True)
# plt.show()

x = df_sample[['quantity tons_log', 'application', 'thickness_log', 'width', 'selling_price_log', 'country', 'customer', 'product_ref']].corr()

# sns.heatmap(x, annot=True, cmap='viridis', center=0)
# plt.show()
