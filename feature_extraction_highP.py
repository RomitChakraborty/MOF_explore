import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'traindata.csv'  # Update with the correct file path
data = pd.read_csv(file_path)

# Select the target variable
target_variable = 'pure_uptake_CO2_298.00K_16bar'

# Bin the target variable into categories (e.g., low, medium, high)
y_binned = pd.qcut(data[target_variable], q=3, labels=['low', 'medium', 'high'])

# Identify feature columns
feature_columns = [col for col in data.columns if col not in ['Unnamed: 0', target_variable, 'pure_uptake_CO2_298.00K_0.15bar']]

# Separate features and target
X = data[feature_columns]

# Check for and handle any non-numeric columns
non_numeric_columns = X.select_dtypes(include=['object']).columns
X = X.drop(columns=non_numeric_columns)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binned, test_size=0.2, random_state=42)

# Train the RandomForestClassifier with an optimal number of estimators
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_classifier.fit(X_train, y_train)

# Extract feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame for feature importances
features_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})

# Sort the DataFrame by importance
features_df = features_df.sort_values(by='importance', ascending=False)

# Select the top ten features
top_features = features_df.head(10)

# Print the top ten features
print(top_features)

# Plot the top ten features
plt.figure(figsize=(10, 6))
plt.barh(top_features['feature'], top_features['importance'], color='green')
plt.xlabel('Feature Importance')
plt.title('Top 10 Features for Predicting CO2 Uptake at 16 bar (Excluding CO2 Uptake at 0.15 bar)')
plt.gca().invert_yaxis()
plt.show()
