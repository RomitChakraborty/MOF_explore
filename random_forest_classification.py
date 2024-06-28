import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import matplotlib.pyplot as plt

# Load the dataset
file_path = 'traindata.csv'  # Update with the correct file path
data = pd.read_csv(file_path)

# Select the target variable
target_variable = 'pure_uptake_CO2_298.00K_0.15bar'

# Bin the target variable into categories (e.g., low, medium, high)
# Assuming equal-frequency binning into three categories
y_binned = pd.qcut(data[target_variable], q=3, labels=['low', 'medium', 'high'])

# Identify feature columns
feature_columns = [col for col in data.columns if col not in ['Unnamed: 0', target_variable]]

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

# Define range of n_estimators
n_estimators_range = range(10, 201, 10)  # From 10 to 200 with step size of 10

# Initialize lists to store metrics
f1_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []

# Train and evaluate the model for different n_estimators
for n_estimators in n_estimators_range:
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    
    # Compute metrics
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_test, y_pred, average='weighted'))

# Save results to a DataFrame
results_df = pd.DataFrame({
    'n_estimators': n_estimators_range,
    'f1_score': f1_scores,
    'accuracy_score': accuracy_scores,
    'precision_score': precision_scores,
    'recall_score': recall_scores
})

# Save the results to a CSV file
results_df.to_csv('classification_metrics_vs_n_estimators.csv', index=False)
