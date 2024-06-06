import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("Copper_Set.xlsx")
#print(df.head(2))

# print(len(df['item_date'].unique())) 
# print(len(df['customer'].unique())) 
# print(len(df['material_ref'].unique()))
# print(len(df['product_ref'].unique())) 
# print(len(df['delivery date'].unique())) 

missing_values = df.isnull().sum()
# print(missing_values)

# print(df.info())

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
# print(df.shape)
# df.info()

df['material_ref'] = df['material_ref'].fillna('unknown')
df = df.dropna()

missing_values_count = df.isnull().sum()
# print(missing_values_count)
# print(df.shape)

# df_sample = df.sample(frac=0.2, random_state=42)

# sns.histplot(df_sample['quantity tons'], kde=True)
# plt.show()
# sns.histplot(df_sample['country'], kde=True)
# plt.show()
# sns.histplot(df_sample['application'], kde=True)
# plt.show()
# sns.histplot(df_sample['thickness'], kde=True)
# plt.show()
# sns.histplot(df_sample['width'], kde=True)
# plt.show()
# sns.histplot(df_sample['selling_price'], kde=True)
# plt.show()

mask1 = df['selling_price'] <= 0
# print(mask1.sum())
df.loc[mask1, 'selling_price'] = np.nan

mask1 = df['quantity tons'] <= 0
# print(mask1.sum())
df.loc[mask1, 'quantity tons'] = np.nan

mask1 = df['thickness'] <= 0
# print(mask1.sum())

# df.isnull().sum()

df.dropna(inplace=True)
# print(len(df))

df_sample = df.sample(frac=0.2, random_state=42)

df_sample['selling_price_log'] = np.log(df_sample['selling_price'])
sns.histplot(df_sample['selling_price_log'], kde=True)
plt.show()

df_sample['quantity tons_log'] = np.log(df_sample['quantity tons'])
sns.histplot(df_sample['quantity tons_log'], kde=True)
plt.show()

df_sample['thickness_log'] = np.log(df_sample['thickness'])
sns.histplot(df_sample['thickness_log'], kde=True)
plt.show()