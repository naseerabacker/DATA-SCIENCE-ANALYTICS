import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py

data = pd.read_csv(r'C:\Users\USER\Downloads\SCMS_Delivery_History_Dataset.csv')

import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)
warnings.filterwarnings('ignore',category=UserWarning)
warnings.filterwarnings('ignore',category=FutureWarning)

# Convert 'Freight Cost (USD)','Weight (Kilograms)' to numeric (float) data type
data['Freight Cost (USD)'] = pd.to_numeric(data['Freight Cost (USD)'], errors='coerce')
data['Weight (Kilograms)'] = pd.to_numeric(data['Weight (Kilograms)'], errors='coerce')

#filling the missing values in the columns with median value

for col in ['Line Item Insurance (USD)']:
    data[col]=data[col].fillna(data[col].median())
    
data['Shipment Mode'].fillna(data['Shipment Mode'].mode()[0], inplace=True)
data['Weight (Kilograms)'].fillna(data['Weight (Kilograms)'].mode()[0], inplace=True)

data['Dosage'].fillna('Unknown', inplace=True)

date_columns = [ 'Scheduled Delivery Date', 'Delivered to Client Date', 'Delivery Recorded Date']
for col in date_columns:
    data[col] = pd.to_datetime(data[col])

# Calculate the variance between scheduled and actual delivery dates and add it as a new column
data['Delivery Variance'] = data['Scheduled Delivery Date'] - data['Delivered to Client Date']

# Identify early or delayed deliveries
data['Delivery Status'] = data['Delivery Variance'].apply(lambda x: 'Early' if x.days < 0 else 'On Time' if x.days == 0 else 'Delayed')

col=['Project Code','PQ #','PO / SO #','ASN/DN #','PQ First Sent to Client Date','PO Sent to Vendor Date','Scheduled Delivery Date','Delivered to Client Date','Delivery Recorded Date','Product Group','Item Description','Molecule/Test Type','Dosage','Unit of Measure (Per Pack)','Line Item Quantity','Pack Price','Unit Price','Manufacturing Site','Delivery Variance']
data1=data.drop(col,axis=1)

rare_threshold = 10

# Calculate the occurrence frequency of each variable
category_counts = data['Country'].value_counts()

# Identify rarely occurring variables
rare_categories = category_counts[category_counts <= rare_threshold].index

# Group the rarely occurring variables into a new category 'Rare'
data['Country'] = data['Country'].apply(lambda x: 'Rare' if x in rare_categories else x)

# Get the value counts after the grouping
value_counts = data['Country'].value_counts()

# Display the value counts, including the count of 'Rare' category
print(value_counts)

# Display the count of rare occurrences
rare_count = value_counts['Rare'] if 'Rare' in value_counts.index else 0
print("Rare Count:", rare_count)


rare_threshold = 10

# Calculate the occurrence frequency of each variable
category_counts = data['Vendor'].value_counts()

# Identify rarely occurring variables
rare_categories = category_counts[category_counts <= rare_threshold].index

# Group the rarely occurring variables into a new category 'Rare'
data['Vendor'] = data['Vendor'].apply(lambda x: 'Rare' if x in rare_categories else x)

# Get the value counts after the grouping
value_counts = data['Vendor'].value_counts()

# Display the value counts, including the count of 'Rare' category
print(value_counts)

# Display the count of rare occurrences
rare_count = value_counts['Rare'] if 'Rare' in value_counts.index else 0
print("Rare Count:", rare_count)
rare_threshold = 10

# Calculate the occurrence frequency of each variable
category_counts = data['Dosage'].value_counts()

# Identify rarely occurring variables
rare_categories = category_counts[category_counts <= rare_threshold].index

# Group the rarely occurring variables into a new category 'Rare'
data['Dosage'] = data['Dosage'].apply(lambda x: 'Rare' if x in rare_categories else x)

# Get the value counts after the grouping
value_counts = data['Dosage'].value_counts()

# Display the value counts, including the count of 'Rare' category
print(value_counts)

# Display the count of rare occurrences
rare_count = value_counts['Rare'] if 'Rare' in value_counts.index else 0
print("Rare Count:", rare_count)

rare_threshold = 10

# Calculate the occurrence frequency of each variable
category_counts = data['Brand'].value_counts()

# Identify rarely occurring variables
rare_categories = category_counts[category_counts <= rare_threshold].index

# Group the rarely occurring variables into a new category 'Rare'
data['Brand'] = data['Brand'].apply(lambda x: 'Rare' if x in rare_categories else x)

# Get the value counts after the grouping
value_counts = data['Brand'].value_counts()

# Display the value counts, including the count of 'Rare' category
print(value_counts)

# Display the count of rare occurrences
rare_count = value_counts['Rare'] if 'Rare' in value_counts.index else 0
print("Rare Count:", rare_count)

#Encoding

from sklearn.preprocessing import LabelEncoder

label_cols = ['Country', 'Managed By', 'Fulfill Via', 'Vendor INCO Term',
       'Shipment Mode', 'Sub Classification', 'Vendor', 'Brand', 'Dosage Form','First Line Designation','Delivery Status']
label_encoders = {}  # To store encoders for each column

for col in label_cols:
    le = LabelEncoder()
    encoded = le.fit_transform(data1[col])
    data1[col] = encoded
    label_encoders[col] = le
    
    
#Scaling

from sklearn.preprocessing import RobustScaler

# List of numerical columns to be scaled
cols_to_scale = ['Line Item Value', 'Weight (Kilograms)','Line Item Insurance (USD)']
robust_scaling = {}  # To store scaling for each column

for col in cols_to_scale:
    rs = RobustScaler()
    scaled_values = rs.fit_transform(data1[[col]])  # Notice the change here, using rs.fit_transform
    data1[col] = scaled_values
    robust_scaling[col] = rs
#Model creation

# Assuming "Freight Cost" column is in X
# Extract features (X) and target labels (y)
X = data1.drop(['Freight Cost (USD)'], axis=1)  # Drop the "Freight Cost" column from features
y = data1['Freight Cost (USD)']

# Check which rows have non-missing "Freight Cost" values
has_freight_cost = ~np.isnan(y)

# Separate the data into training and testing sets based on the availability of "Freight Cost" values
X_train = X[has_freight_cost]
X_test = X[~has_freight_cost]
y_train = y[has_freight_cost]

from sklearn.impute import SimpleImputer

# Impute missing values in the target variable for the training set
imputer = SimpleImputer(strategy='median')
y_train_imputed = imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()


# Assign feature names to the model
feature_names = ['Country', 'Managed By', 'Fulfill Via', 'Vendor INCO Term',
       'Shipment Mode', 'Sub Classification', 'Vendor', 'Brand', 'Dosage Form',
       'Line Item Value', 'First Line Designation', 'Weight (Kilograms)','Line Item Insurance (USD)', 'Delivery Status']

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming you have already split your data into features (X) and target (y)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values in y_train
# Replace NaN values with a suitable imputed value (e.g., mean or median)
y_train_imputed = np.nan_to_num(y_train, nan=np.nanmean(y_train))

# Handle missing values in y_test
# Replace NaN values with a suitable imputed value (e.g., mean or median)
y_test_imputed = np.nan_to_num(y_test, nan=np.nanmean(y_test))

# Define hyperparameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the Gradient Boosting Regressor model
gb_reg = GradientBoostingRegressor(random_state=42)

# Hyperparameter tuning using RandomizedSearchCV
random_search_gb = RandomizedSearchCV(gb_reg, param_distributions=param_dist, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search_gb.fit(X_train, y_train_imputed)

best_gb_model = random_search_gb.best_estimator_

# Make predictions on the test set
X_test = X_test[X_train.columns]
y_pred_gb = best_gb_model.predict(X_test)

# Calculate Mean Squared Error (MSE) and R-Squared (R2) error for the test set
MSE_gb = mean_squared_error(y_test_imputed, y_pred_gb)
R2_gb = r2_score(y_test_imputed, y_pred_gb)

# Print MSE and R-Squared error
print('Best GB Regressor MSE =', MSE_gb)
print('Best GB Regressor R-Squared =', R2_gb)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming you have already split your data into features (X) and target (y)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values in y_train
# Replace NaN values with a suitable imputed value (e.g., mean or median)
y_train_imputed = np.nan_to_num(y_train, nan=np.nanmean(y_train))

# Handle missing values in y_test
# Replace NaN values with a suitable imputed value (e.g., mean or median)
y_test_imputed = np.nan_to_num(y_test, nan=np.nanmean(y_test))

# Define hyperparameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the Gradient Boosting Regressor model
gb_reg = GradientBoostingRegressor(random_state=42)

# Hyperparameter tuning using RandomizedSearchCV
random_search_gb = RandomizedSearchCV(gb_reg, param_distributions=param_dist, n_iter=20, cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search_gb.fit(X_train, y_train_imputed)

best_gb_model = random_search_gb.best_estimator_

# Cross-validation predictions
y_cv_pred = cross_val_predict(best_gb_model, X_train, y_train_imputed, cv=5)

# Reorder the columns in X_test to match the order in X_train
X_test_reordered = X_test[X_train.columns]

# Make predictions using the model
y_pred_gb = best_gb_model.predict(X_test_reordered)

# Calculate Mean Squared Error (MSE) and R-Squared (R2) error for the test set
MSE_gb = mean_squared_error(y_test_imputed, y_pred_gb)
R2_gb = r2_score(y_test_imputed, y_pred_gb)

# Print cross-validated MSE and R-Squared
cv_MSE_gb = mean_squared_error(y_train_imputed, y_cv_pred)
cv_R2_gb = r2_score(y_train_imputed, y_cv_pred)
print('Cross-validated GB Regressor MSE =', cv_MSE_gb)
print('Cross-validated GB Regressor R-Squared =', cv_R2_gb)

# Print MSE and R-Squared error
print('Best GB Regressor MSE =', MSE_gb)
print('Best GB Regressor R-Squared =', R2_gb)


best_gb_model.feature_names = feature_names



import pickle

with open('label_encoders.pkl', 'wb') as file:
    pickle.dump((label_encoders), file)
    
with open('robust_scaling.pkl', 'wb') as file:
    pickle.dump(robust_scaling, file)

# Save the trained model and encoders using pickle
with open('Gradient_booster.pkl', 'wb') as file:
    pickle.dump(best_gb_model, file)