import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel("Copper_Set.xlsx")
#print(df.head(2))

# Basic data exploration
print(len(df['item_date'].unique())) 
print(len(df['customer'].unique())) 
print(len(df['material_ref'].unique()))
print(len(df['product_ref'].unique())) 
print(len(df['delivery date'].unique())) 

missing_values = df.isnull().sum()
# print(missing_values)

# print(df.info())

>>>>>>> parent of ea45c53 (Heatmap done)
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

<<<<<<< HEAD
# Handle missing values
=======
missing_values_count = df.isnull().sum()
# print(missing_values_count)

>>>>>>> parent of ea45c53 (Heatmap done)
df['material_ref'] = df['material_ref'].fillna('unknown')

df = df.dropna()

<<<<<<< HEAD
# Remove rows where 'selling_price' or other essential columns are NaN
df.dropna(subset=['selling_price'], inplace=True)

# Sample the data
df_sample = df.sample(frac=0.2, random_state=42)

# Log transformations for selected columns
df_sample['selling_price'] = df_sample['selling_price'].apply(lambda x: np.nan if x <= 0 else x)
df_sample['quantity tons'] = df_sample['quantity tons'].apply(lambda x: np.nan if x <= 0 else x)
df_sample['thickness'] = df_sample['thickness'].apply(lambda x: np.nan if x <= 0 else x)
df_sample['selling_price_log'] = np.log(df_sample['selling_price'])
df_sample['quantity tons_log'] = np.log(df_sample['quantity tons'])
df_sample['thickness_log'] = np.log(df_sample['thickness'])

# Visualizations
sns.histplot(df_sample['selling_price_log'], kde=True)
plt.show()
sns.histplot(df_sample['quantity tons_log'], kde=True)
plt.show()
sns.histplot(df_sample['thickness_log'], kde=True)
plt.show()

# Correlation heatmap
correlation_matrix = df_sample[['quantity tons_log', 'application', 'thickness_log', 'width', 'selling_price_log', 'country', 'customer', 'product_ref']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', center=0)
plt.show()

# Preparing features for the model
X = df_sample[['quantity tons_log', 'application', 'thickness_log', 'width', 'country', 'customer', 'product_ref']]
y = df_sample['selling_price_log']

# One-hot encoding categorical variables
ohe_item_type = OneHotEncoder(handle_unknown='ignore')
ohe_status = OneHotEncoder(handle_unknown='ignore')

X_ohe_item_type = ohe_item_type.fit_transform(df_sample[['item type']]).toarray()
X_ohe_status = ohe_status.fit_transform(df_sample[['status']]).toarray()

# Concatenate encoded features with original features
X = np.concatenate((X.values, X_ohe_item_type, X_ohe_status), axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Decision tree model
dtr = DecisionTreeRegressor()
param_grid = {
    'max_depth': [2, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=dtr, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Predictions and evaluation
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean squared error:', mse)
print('R-squared:', r2)

# Making a prediction on new data
new_sample = np.array([[np.log(40), 10, np.log(250), 0, 28, 30202938, 1670798778]])
new_sample_ohe = ohe_item_type.transform(new_sample[:, [7]]).toarray()
new_sample_be = ohe_status.transform(new_sample[:, [8]]).toarray()
new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6]], new_sample_ohe, new_sample_be), axis=1)
new_sample1 = scaler.transform(new_sample)
new_pred = best_model.predict(new_sample1)
print('Predicted selling price:', np.exp(new_pred))

# Saving the model, scaler, and encoders
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('ohe_item_type.pkl', 'wb') as f:
    pickle.dump(ohe_item_type, f)
with open('ohe_status.pkl', 'wb') as f:
    pickle.dump(ohe_status, f)
=======
missing_values_count = df.isnull().sum()
# print(missing_values_count)

df_sample = df.sample(frac=0.2, random_state=42)

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