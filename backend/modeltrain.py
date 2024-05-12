import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, accuracy_score, classification_report

# Importing statistical libraries
import statsmodels.api as sm

# Load the data
data = pd.read_csv("C:\\Users\\bhupa\\hackfest-ticket\\heckfest\\backend\\clothes.csv")

# Sort the data by "Profit %" column in ascending order
highest_profit = data.sort_values(by="Profit %", ascending=False)
data = data.sort_values(by="Unit Sold", ascending=True)

# Extract sorted data for plotting
x = data['Invoice Date']
y = pd.to_numeric(data['Unit Sold'], errors='coerce')

# Sort y values
y_sorted = y.sort_values()

# now to show the relationship bw sell price and invoice date
y_strings = y_sorted.astype(str)
dataframe= pd.DataFrame(data)
X = data[['Invoice Date']]  
y = data['Unit Sold']  
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
model= LinearRegression()
model.fit(X_train,y_train)
y_pred= model.predict(X_test)
# accuracy= accuracy_score(y_test,y_pred)
# report= classification_report(y_test,y_pred)

# #user input 
while True:
    user_date = int(input("Enter the date on which you want (1-31): "))
    if user_date > 31 or user_date < 1: 
        print("Error: Invalid input. Please enter a date between 1 and 31.")
    else: break

user_date_df = pd.DataFrame({'Invoice Date': [user_date]})
prediction = model.predict(user_date_df)
print("Predicted Sells are:", prediction[0])  
# codes for errors
mse= mean_squared_error(y_test, y_pred)
print("Mean Squared Error ",mse)
mape= mean_absolute_percentage_error(y_pred,y_test)
print("Mean absolute Percentage error:",mape)

# Bar graph
plt.xlabel('Invoice Date', fontsize=18)
plt.ylabel('Unit Sold', fontsize=16)
plt.bar(x, y_sorted)  
plt.show()