import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
df['customer'] = pd.to_numeric(df['customer'], errors='coerce')
df['country'] = pd.to_numeric(df['country'], errors='coerce')
df['application'] = pd.to_numeric(df['application'], errors='coerce')
df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
df['width'] = pd.to_numeric(df['width'], errors='coerce')
df['material_ref'] = df['material_ref'].str.lstrip('0')
df['product_ref'] = pd.to_numeric(df['product_ref'], errors='coerce')
df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')

missing_values_count = df.isnull().sum()
# print(missing_values_count)
# print(df.shape)
# df.info()

df['material_ref'] = df['material_ref'].fillna('unknown')

df = df.dropna()

missing_values_count = df.isnull().sum()
# print(missing_values_count)
# print(df.shape)

df_copy=df.copy()

sns.histplot(df_copy['quantity tons'], kde=True)
plt.show()
sns.histplot(df_copy['country'], kde=True)
plt.show()
sns.histplot(df_copy['application'], kde=True)
plt.show()
sns.histplot(df_copy['thickness'], kde=True)
plt.show()
sns.histplot(df_copy['width'], kde=True)
plt.show()
sns.histplot(df_copy['selling_price'], kde=True)
plt.show()