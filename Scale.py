# Dependencies needed:
# - numpy: for numerical operations
# - scikit-learn: for scaling and regression modeling
# Install via pip if needed:
# pip install numpy scikit-learn

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Example dataset:
# Features: [Weight (kg), Volume (liters)]
# Target: CO2 emissions (arbitrary units)
# Note: Volume is given in liters instead of cm3 to illustrate scaling necessity
X = np.array([
    [2000, 1.0],
    [2500, 1.5],
    [1800, 0.8],
    [2200, 1.2]
])

y = np.array([300, 400, 250, 320])  # Target values

# Why scale features?
# Different units and ranges (kg vs liters) make direct comparison difficult.
# Scaling transforms features to a common scale (mean=0, std=1),
# improving model performance and interpretability.

# Initialize the scaler
scale = StandardScaler()

# Fit the scaler on the training data and transform it
scaledX = scale.fit_transform(X)

# Initialize and train the regression model on scaled features
regr = LinearRegression()
regr.fit(scaledX, y)

# Now, suppose we want to predict CO2 emissions for a new data point:
# Weight = 2300 kg, Volume = 1.3 liters
new_data = np.array([[2300, 1.3]])

# Scale the new data using the same scaler fitted on training data
scaled = scale.transform(new_data)

# Predict CO2 emissions using the trained model
predictedCO2 = regr.predict(scaled)

print(f"Predicted CO2 emissions: {predictedCO2[0]:.2f}")
