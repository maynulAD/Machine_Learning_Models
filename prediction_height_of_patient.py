import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the data
data = pd.read_csv('predictfromparentsheight.csv')

# Create a new feature: average parental height
data['avg_parent_height'] = (data['Fathers height'] + data['mothers height']) / 2

# Define features and target
X = data[['age', 'Fathers height', 'mothers height', 'avg_parent_height']]  # Input features
y = data['height']  # Target variable (height to be predicted)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Train the model (using Random Forest Regression)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Predict height for a new data point
new_data = pd.DataFrame({
    'age': [22],
    'Fathers height': [160],
    'mothers height': [150],
    'avg_parent_height': [(160 + 150) / 2]
})
predicted_height = model.predict(new_data)
print(f'Predicted Height: {predicted_height[0]}')