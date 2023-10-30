import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_excel(r"C:\Users\anish\Downloads\embeddingsdata.xlsx")
# Hypothetical data
a=df['embed_0']
b=df['embed_1']
# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(a, b, color='blue', marker='o', label='0 vs. 1')
plt.title('Scatter Plot of 0 vs. 1')
plt.xlabel('embed_0')
plt.ylabel('embed_1')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
