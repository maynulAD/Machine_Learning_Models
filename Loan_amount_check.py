import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
data = pd.read_csv('Loan_Approval_Data.csv')

# Define features and target
X = data[['Applicant Age', 'Credit Score', 'Annual Income', 'Employment Status', 'Debt-to-Income Ratio']]
y = data['Loan Amount']

# Preprocessing: One-hot encode 'Employment Status' and scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Employment Status']),
        ('num', StandardScaler(), ['Applicant Age', 'Credit Score', 'Annual Income', 'Debt-to-Income Ratio'])
    ])

# Create a pipeline with preprocessing and Ridge Regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))  # Ridge Regression with regularization parameter alpha=1.0
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Predict Loan Amount for all applicants
data['Predicted Loan Amount'] = model.predict(X)
print(data[['Loan ID', 'Loan Amount', 'Predicted Loan Amount']])