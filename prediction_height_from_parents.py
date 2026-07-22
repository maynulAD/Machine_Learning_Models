import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor

# Dummy dataset (Age, Gender, Father's Height, Mother's Height) -> Child's Height
data = np.array([
    [5, 0, 175, 160, 110],  # Male, 5 years old
    [10, 1, 170, 155, 130], # Female, 10 years old
    [15, 0, 180, 165, 150], # Male, 15 years old
    [8, 1, 175, 160, 120],  # Female, 8 years old
])

X = data[:, :4]  # Features: Age, Gender, Father's Height, Mother's Height
y = data[:, 4]   # Target: Predicted Height

# Train a simple Decision Tree Model
model = DecisionTreeRegressor()
model.fit(X, y)

# Save the model
with open("height_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load the trained model
with open("height_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Function to predict height
def predict_height():
    try:
        age = int(entry_age.get())
        gender = int(entry_gender.get())  # 0 = Male, 1 = Female
        height_father = float(entry_father.get())
        height_mother = float(entry_mother.get())

        # Prediction
        prediction = loaded_model.predict([[age, gender, height_father, height_mother]])[0]
        messagebox.showinfo("Prediction", f"Predicted Height: {prediction:.2f} cm")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid input.")

# Create GUI
root = tk.Tk()
root.title("Height Prediction App")

tk.Label(root, text="Age:").grid(row=0, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1)

tk.Label(root, text="Gender (0: Male, 1: Female):").grid(row=1, column=0)
entry_gender = tk.Entry(root)
entry_gender.grid(row=1, column=1)

tk.Label(root, text="Father's Height (cm):").grid(row=2, column=0)
entry_father = tk.Entry(root)
entry_father.grid(row=2, column=1)

tk.Label(root, text="Mother's Height (cm):").grid(row=3, column=0)
entry_mother = tk.Entry(root)
entry_mother.grid(row=3, column=1)

tk.Button(root, text="Predict", command=predict_height).grid(row=4, columnspan=2)

root.mainloop()
