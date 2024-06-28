# MOF_explore
Explore CO2 uptake in MOFs using a random forest model

# CO2 Uptake Prediction

This repository contains the code and data for predicting CO2 uptake at different pressures using RandomForest models. The project includes regression and classification tasks to understand the most important features influencing CO2 uptake and to evaluate model performance across different numbers of estimators.

## Contents

- `traindata.csv`: The dataset used for model training and evaluation.
- `random_forest_pure_co2.py`: Script for the regression task predicting CO2 uptake at low pressures (0.15 bar).
- `random_forest_classification.py`: Script for the classification task predicting CO2 uptake categories (low, medium, high).
- `feature_extraction_lowP.py`: Script to identify and plot the top 10 features for CO2 uptake at 0.15 bar.
- `feature_extraction_highP.py`: Script to identify and plot the top 10 features for CO2 uptake at 16 bar.
- `README.md`: Project documentation.

## Setup

### Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/CO2_Uptake_Prediction.git
