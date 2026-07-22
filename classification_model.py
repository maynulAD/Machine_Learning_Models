import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
file_path = "C:\\spyder\\ageVSheight.csv"
df = pd.read_csv(file_path)

# Define height categories
def categorize_height(height):
    if height < 100:
        return "Short"
    elif 100 <= height < 150:
        return "Medium"
    else:
        return "Tall"

# Apply categorization
df['height_category'] = df['height'].apply(categorize_height)

# Encode labels
le = LabelEncoder()
df['height_category_encoded'] = le.fit_transform(df['height_category'])

# Prepare data
X = df[['age']]
y = df['height_category_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

def predict_category():
    try:
        age = int(entry_age.get())
        prediction = model.predict([[age]])[0]
        category = le.inverse_transform([prediction])[0]
        messagebox.showinfo("Prediction", f"Predicted Height Category: {category}")
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid age.")

# Create UI
root = tk.Tk()
root.title("AI Height Classification")

tk.Label(root, text="Enter Age:").grid(row=0, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1)

tk.Button(root, text="Predict", command=predict_category).grid(row=1, columnspan=2)

root.mainloop()
