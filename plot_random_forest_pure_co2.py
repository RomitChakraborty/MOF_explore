import pandas as pd
import matplotlib.pyplot as plt

# Load the results
results_df = pd.read_csv('r2_scores_vs_n_estimators.csv')

# Plot the R² scores as a function of n_estimators
plt.figure(figsize=(10, 6))
plt.plot(results_df['n_estimators'], results_df['r2_score'], marker='o')
plt.title('R² Score vs. Number of Estimators in RandomForestRegressor')
plt.xlabel('Number of Estimators')
plt.ylabel('R² Score')
plt.grid(True)
plt.savefig('r2_scores_vs_n_estimators.png')
plt.show()
