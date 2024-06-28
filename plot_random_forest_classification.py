import pandas as pd
import matplotlib.pyplot as plt

# Load the results
results_df = pd.read_csv('classification_metrics_vs_n_estimators.csv')

# Plot F1 Score
plt.figure(figsize=(10, 6))
plt.plot(results_df['n_estimators'], results_df['f1_score'], marker='o')
plt.title('F1 Score vs. Number of Estimators in RandomForestClassifier')
plt.xlabel('Number of Estimators')
plt.ylabel('F1 Score')
plt.grid(True)
plt.savefig('f1_score_vs_n_estimators.png')


# Plot Accuracy Score
plt.figure(figsize=(10, 6))
plt.plot(results_df['n_estimators'], results_df['accuracy_score'], marker='o', color='r')
plt.title('Accuracy Score vs. Number of Estimators in RandomForestClassifier')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy Score')
plt.grid(True)
plt.savefig('accuracy_score_vs_n_estimators.png')


# Plot Precision Score
plt.figure(figsize=(10, 6))
plt.plot(results_df['n_estimators'], results_df['precision_score'], marker='o', color='g')
plt.title('Precision Score vs. Number of Estimators in RandomForestClassifier')
plt.xlabel('Number of Estimators')
plt.ylabel('Precision Score')
plt.grid(True)
plt.savefig('precision_score_vs_n_estimators.png')


# Plot Recall Score
plt.figure(figsize=(10, 6))
plt.plot(results_df['n_estimators'], results_df['recall_score'], marker='o', color='m')
plt.title('Recall Score vs. Number of Estimators in RandomForestClassifier')
plt.xlabel('Number of Estimators')
plt.ylabel('Recall Score')
plt.grid(True)
plt.savefig('recall_score_vs_n_estimators.png')

