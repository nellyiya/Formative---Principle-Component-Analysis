

# Fuel Economy Dataset

## Overview  

This dataset contains standardized and normalized values of various vehicle attributes related to fuel economy, emissions, and performance. The dataset consists of **3,929 records** with **14 features** and has been processed to ensure consistency and usability for analysis and machine learning applications.  

## Dataset Description  

The dataset includes the following columns:  

- **id**: Unique identifier for each vehicle entry (standardized).  
- **year**: Model year of the vehicle (normalized).  
- **cylinders**: Number of cylinders in the engine (normalized).  
- **displ**: Engine displacement in liters (normalized).  
- **pv2**: Power-to-weight ratio (normalized).  
- **pv4**: Another power-related variable (normalized).  
- **city**: City fuel economy in miles per gallon (MPG) (normalized).  
- **UCity**: Urban fuel economy (normalized).  
- **highway**: Highway fuel economy in MPG (normalized).  
- **UHighway**: Urban highway fuel economy (normalized).  
- **comb**: Combined fuel economy (normalized).  
- **co2**: CO2 emissions in grams per mile (normalized).  
- **feScore**: Fuel economy score based on efficiency (normalized).  
- **ghgScore**: Greenhouse gas emissions score (normalized).  

## Data Preprocessing  

The dataset has been preprocessed to ensure that numerical values are properly scaled and standardized:  

1. **Handling Missing Values**: Any missing data points were either imputed or removed to maintain dataset integrity.  
2. **Normalization**: Features were normalized to a range where the **mean = 0** and **standard deviation = 1** using **Min-Max Scaling** or **Z-score Normalization**.  
3. **Feature Engineering**: Derived or transformed variables were created where necessary.  
4. **Data Cleaning**: Non-numeric categorical variables were excluded or converted into numerical representations.  

### Code Implementation  

The dataset preprocessing and analysis were implemented using **Python** and the following libraries:  
- `pandas` for data manipulation  
- `numpy` for numerical operations  
- `matplotlib` and `seaborn` for visualization  
- `scikit-learn` for preprocessing and machine learning  

#### Data Loading and Exploration  
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("fuel_econ.csv")

# Display the first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())
```

#### Data Normalization  
```python
from sklearn.preprocessing import StandardScaler

# Selecting numeric columns for normalization
numeric_cols = ['year', 'cylinders', 'displ', 'pv2', 'pv4', 'city', 'UCity', 
                'highway', 'UHighway', 'comb', 'co2', 'feScore', 'ghgScore']

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(df.describe())  # Check statistics after normalization
```

#### Visualization of Fuel Economy Distribution  
```python
plt.figure(figsize=(10,6))
sns.histplot(df['comb'], bins=30, kde=True, color='blue')
plt.xlabel("Combined Fuel Economy (Normalized)")
plt.ylabel("Frequency")
plt.title("Distribution of Combined Fuel Economy")
plt.show()
```

## Usage  

This dataset can be used for:  
- **Analyzing fuel economy trends** across different vehicle models  
- **Predicting vehicle emissions and efficiency scores** based on input attributes  
- **Machine learning applications** for optimizing fuel efficiency and minimizing emissions  
- **Policy recommendations** for sustainable and eco-friendly vehicle choices  
