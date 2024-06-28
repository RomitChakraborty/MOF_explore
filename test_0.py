import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'traindata.csv'  # Update with the correct file path
data = pd.read_csv(file_path)

# Select the target variable
target_variable = 'pure_uptake_CO2_298.00K_0.15bar'

# Identify feature columns
feature_columns = [col for col in data.columns if col not in ['Unnamed: 0', target_variable]]

# Separate features and target
X = data[feature_columns]
y = data[target_variable]

# Check for and handle any non-numeric columns
non_numeric_columns = X.select_dtypes(include=['object']).columns
X = X.drop(columns=non_numeric_columns)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define range of n_estimators
n_estimators_range = range(10, 201, 10)  # From 10 to 200 with step size of 10
r2_scores = []

# Train and evaluate the model for different n_estimators
for n_estimators in n_estimators_range:
    rf_regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

# Save results to a DataFrame
results_df = pd.DataFrame({'n_estimators': n_estimators_range, 'r2_score': r2_scores})

# Save the results to a CSV file
results_df.to_csv('r2_scores_vs_n_estimators.csv', index=False)

