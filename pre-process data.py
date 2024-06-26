import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pickle


df = pd.read_excel("Copper_Set.xlsx")
# print(df.head(2))

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
# df = df.dropna()

missing_values_count = df.isnull().sum()
# print(missing_values_count)
# print(df.shape)

df_sample = df.sample(frac=0.2, random_state=42)

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
# sns.histplot(df_sample['selling_price_log'], kde=True)
# plt.show()

df_sample['quantity tons_log'] = np.log(df_sample['quantity tons'])
# sns.histplot(df_sample['quantity tons_log'], kde=True)
# plt.show()

df_sample['thickness_log'] = np.log(df_sample['thickness'])
# sns.histplot(df_sample['thickness_log'], kde=True)
# plt.show()

# print(df_sample.head())

# x = df_sample[['quantity tons_log', 'application', 'thickness_log', 'width', 'selling_price_log', 'country', 'customer', 'product_ref']].corr()

# sns.heatmap(x, annot=True, cmap='viridis', center=0) 
# plt.show()

##########################################################################

X=df_sample[['quantity tons_log','status','item type','application','thickness_log','width','country','customer','product_ref']]
y=df_sample['selling_price_log']

ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(X[['item type']])
X_ohe = ohe.fit_transform(X[['item type']]).toarray()
ohe2 = OneHotEncoder(handle_unknown='ignore')
ohe2.fit(X[['status']])
X_be = ohe2.fit_transform(X[['status']]).toarray()

X = np.concatenate((X[['quantity tons_log', 'application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe, X_be), axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

dtr = DecisionTreeRegressor()

param_grid = {
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None],  
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=dtr, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# print("Best hyperparameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# print('Mean squared error:', mse)
# print('R-squared:', r2)

new_sample = pd.DataFrame([[np.log(40), 'Won', 'PL', 10, np.log(250), 0, 28, 30202938, 1670798778]], 
                          columns=['quantity tons_log', 'status', 'item type', 'application', 'thickness_log', 'width', 'country', 'customer', 'product_ref'])

new_sample_ohe = ohe.transform(new_sample[['item type']]).toarray()
new_sample_be = ohe2.transform(new_sample[['status']]).toarray()
new_sample_combined = np.concatenate((new_sample[['quantity tons_log', 'application', 'thickness_log', 'width', 'country', 'customer', 'product_ref']].values, new_sample_ohe, new_sample_be), axis=1)

new_sample_scaled = scaler.transform(new_sample_combined)

new_pred = best_model.predict(new_sample_scaled)

# print('Predicted selling price:', np.exp(new_pred))

with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('t.pkl', 'wb') as f:
    pickle.dump(ohe, f)
with open('s.pkl', 'wb') as f:
    pickle.dump(ohe2, f)

# print(len(df_sample))
# print(df_sample.head(3))

df_c = df_sample[df_sample['status'].isin(['Won', 'Lost'])]
# print(len(df_c))

################################################################################

Y = df_c['status']
X = df_c[['quantity tons_log', 'selling_price_log', 'item type', 'application', 'thickness_log', 'width', 'country', 'customer', 'product_ref']]

ohe_item_type = OneHotEncoder(handle_unknown='ignore')
ohe_item_type.fit(X[['item type']])
X_ohe_item_type = ohe_item_type.transform(X[['item type']]).toarray()

lb = LabelBinarizer()
Y_bin = lb.fit_transform(Y)

X = np.concatenate((X[['quantity tons_log', 'selling_price_log', 'application', 'thickness_log', 'width', 'country', 'customer', 'product_ref']].values, X_ohe_item_type), axis=1)

scaler_clf = StandardScaler()
X = scaler_clf.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_bin, test_size=0.2, random_state=42)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
Y_pred = dtc.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
# print(f"Accuracy: {accuracy}")
cm = confusion_matrix(Y_test, Y_pred)
# print(f"Confusion Matrix:\n{cm}")

# print("Confusion Matrix:")
# print(confusion_matrix(Y_test, Y_pred))
# print("Classification Report:")
# print(classification_report(Y_test, Y_pred))

fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# plt.show()

new_sample_data = pd.DataFrame([[np.log(700), np.log(956), 10, np.log(2), 1500, 28.0, 30202938, 1670798778, 'W']],
                               columns=['quantity tons_log', 'selling_price_log', 'application', 'thickness_log', 'width', 'country', 'customer', 'product_ref', 'item type'])

new_sample_ohe_item_type = ohe_item_type.transform(new_sample_data[['item type']]).toarray()

new_sample_combined = np.concatenate((new_sample_data[['quantity tons_log', 'selling_price_log', 'application', 'thickness_log', 'width', 'country', 'customer', 'product_ref']].values, new_sample_ohe_item_type), axis=1)

new_sample_scaled = scaler_clf.transform(new_sample_combined)

new_pred_class = dtc.predict(new_sample_scaled)
status_prediction = 'Won' if new_pred_class[0] == 1 else 'Lost'
# print(f'The status is: {status_prediction}')

with open('cmodel.pkl', 'wb') as file:
    pickle.dump(dtc, file)
with open('cscaler.pkl', 'wb') as f:
    pickle.dump(scaler_clf, f)
with open('ct.pkl', 'wb') as f:
    pickle.dump(ohe_item_type, f)
