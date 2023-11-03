import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data from the Excel file
df = pd.read_excel(r"C:\Users\anish\Downloads\embeddingsdata.xlsx")

# Extract the features and target
X = df[['embed_0']]
y = df['embed_1']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regression
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train, y_train)
y_pred_dt = dt_regressor.predict(X_test)

# k-NN Regressor
knn_regressor = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors
knn_regressor.fit(X_train, y_train)
y_pred_knn = knn_regressor.predict(X_test)

# Calculate the mean squared error for Decision Tree
mse_dt = mean_squared_error(y_test, y_pred_dt)

# Calculate the mean squared error for k-NN
mse_knn = mean_squared_error(y_test, y_pred_knn)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='blue', marker='o', label='Actual')
plt.scatter(X_test, y_pred_dt, color='red', marker='x', label='Predicted (Decision Tree)')
plt.title('Decision Tree Regression')
plt.xlabel('embed_0')
plt.ylabel('embed_1')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', marker='o', label='Actual')
plt.scatter(X_test, y_pred_knn, color='green', marker='^', label='Predicted (k-NN)')
plt.title('k-NN Regression')
plt.xlabel('embed_0')
plt.ylabel('embed_1')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Mean Squared Error (Decision Tree):", mse_dt)
print("Mean Squared Error (k-NN):", mse_knn)
