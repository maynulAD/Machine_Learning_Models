# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('student_performance_data.csv')

# Select relevant features for clustering
X = data[['Hours Studied Per Week', 'Attendance Percentage', 'Past Exam Scores']]

# Scale the features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering with 2 clusters (Pass/Fail)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Interpret the clusters
# Cluster with higher average scores is labeled as "Pass", and the other as "Fail"
cluster_means = data.groupby('Cluster')[['Hours Studied Per Week', 'Attendance Percentage', 'Past Exam Scores']].mean()
print("Cluster Means:\n", cluster_means)

# Assign cluster labels to "Pass" or "Fail"
data['Pass/Fail'] = data['Cluster'].apply(lambda x: 'Pass' if x == cluster_means['Past Exam Scores'].idxmax() else 'Fail')

# Display the results
print("\nStudent Performance Predictions:")
print(data[['Student ID', 'Hours Studied Per Week', 'Attendance Percentage', 'Past Exam Scores', 'Pass/Fail']])

# Predict for a random student
random_student = np.array([[5, 50, 40]])  # Example: Hours Studied = 15, Attendance = 75%, Past Exam Score = 70
random_student_scaled = scaler.transform(random_student)
predicted_cluster = kmeans.predict(random_student_scaled)
predicted_pass_fail = 'Pass' if predicted_cluster == cluster_means['Past Exam Scores'].idxmax() else 'Fail'

print("\nPrediction for Random Student:")
print(f"Hours Studied: {random_student[0][0]}, Attendance: {random_student[0][1]}%, Past Exam Score: {random_student[0][2]}")
print(f"Predicted: {predicted_pass_fail}")