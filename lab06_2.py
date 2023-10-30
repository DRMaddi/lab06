import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data from the Excel file
df = pd.read_excel(r"C:\Users\anish\Downloads\embeddingsdata.xlsx")

# Extract the features
a = df['embed_0'].values.reshape(-1, 1)
b = df['embed_1'].values

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(a, b, color='blue', marker='o', label='0 vs. 1')
plt.title('Scatter Plot of embed_0 vs. embed_1')
plt.xlabel('embed_0')
plt.ylabel('embed_1')
plt.legend()
plt.grid(True)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(a, b)

# Predict the values from the model
b_predicted = model.predict(a)

# Calculate the mean squared error
mse = mean_squared_error(b, b_predicted)

print("Mean Squared Error:", mse)

# Display the plot and regression line
plt.plot(a, b_predicted, color='red', linewidth=2, label='Regression Line')
plt.legend()
plt.show()
