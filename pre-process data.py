import pandas as pd

df = pd.read_excel("Copper_Set.xlsx")
#print(df.head(2))

# print(len(df['item_date'].unique())) 
# print(len(df['customer'].unique())) 
# print(len(df['material_ref'].unique()))
# print(len(df['product_ref'].unique())) 
# print(len(df['delivery date'].unique())) 

missing_values = df.isnull().sum()
print(missing_values)