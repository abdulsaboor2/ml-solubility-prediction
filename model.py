import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/abdulsaboor2/ml-solubility-prediction/refs/heads/main/delaney_solubility_with_descriptors.csv')

# Display the dataset
print(df.head())

# Separate target variable (y) and features (X)
y = df['logS']
X = df.drop('logS', axis=1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict with Linear Regression
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# Evaluate Linear Regression
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print('Linear Regression Results:')
print(f'Training MSE: {lr_train_mse}, Training R2: {lr_train_r2}')
print(f'Test MSE: {lr_test_mse}, Test R2: {lr_test_r2}')

# Store results in a DataFrame
lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

# Train Random Forest model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

# Predict with Random Forest
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

# Evaluate Random Forest
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

print('Random Forest Results:')
print(f'Training MSE: {rf_train_mse}, Training R2: {rf_train_r2}')
print(f'Test MSE: {rf_test_mse}, Test R2: {rf_test_r2}')

# Store results in a DataFrame
rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

# Combine results from both models
df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)
print(df_models)

# Visualization: Linear Regression predictions
plt.figure(figsize=(5, 5))
plt.scatter(y_train, y_lr_train_pred, c="#7CAE00", alpha=0.3)
z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')
plt.title('Linear Regression: Experimental vs Predicted LogS')
plt.show()
