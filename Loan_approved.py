import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("Loan_Approval_Data.csv")

# Drop unnecessary column
df.drop(columns=["Loan ID"], inplace=True)

# Encode categorical variables
le_loan = LabelEncoder()
df["Loan Approved"] = le_loan.fit_transform(df["Loan Approved"])  # Yes -> 1, No -> 0

le_emp = LabelEncoder()
df["Employment Status"] = le_emp.fit_transform(df["Employment Status"])  # Encode employment status

# Define features and target variable
X = df.drop(columns=["Loan Approved"])  # Features
y = df["Loan Approved"]  # Target

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Function to predict loan approval
def predict_loan_approval(applicant_data):
    input_df = pd.DataFrame([applicant_data], columns=X.columns)
    prediction = model.predict(input_df)
    return "Yes" if prediction[0] == 1 else "No"

# Example usage
sample_applicant = {
    "Applicant Age": 27,
    "Credit Score": 720,
    "Annual Income": 10000,
    "Loan Amount": 20000,
    "Employment Status": le_emp.transform(["Employed"])[0],
    "Debt-to-Income Ratio": 0.3
}

print("Predicted Loan Approval:", predict_loan_approval(sample_applicant))
