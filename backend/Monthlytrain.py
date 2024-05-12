import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

train_files = ['2016.csv', '2017.csv', '2018.csv', '2019.csv', '2020.csv', '2021.csv']
dfs_train = [pd.read_csv(file) for file in train_files]
train_data = pd.concat(dfs_train)

test_files = ['2023.csv', '2022.csv']
dfs_test = [pd.read_csv(file) for file in test_files]
test_data = pd.concat(dfs_test)

X_train = train_data[['Invoice Month']]  # Note the double square brackets [[]]
y_train = train_data['Units Sold']

X_test = test_data[['Invoice Month']]  # Note the double square brackets [[]]
y_test = test_data['Units Sold']

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

while True:
    user_month = int(input("Enter the month on which you want (1-12): "))
    if user_month > 12 or user_month < 1: 
        print("Error: Invalid input. Please enter a month between 1 and 12.")
    else: 
        break

user_month_df = pd.DataFrame({'Invoice Month': [user_month]})
prediction = model.predict(user_month_df)
print("Predicted Sells are:", prediction[0])  

# Calculate mean squared error and mean absolute percentage error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error:", 1-mape)


#now to locate these data on the graph

import plotly.graph_objs as go
import pandas as pd

# Load the data
files = ['2016.csv', '2017.csv', '2018.csv', '2019.csv', '2020.csv', '2021.csv']
dfs = [pd.read_csv(file) for file in files]
data = pd.concat(dfs)

# Plotting
fig = go.Figure()

for year, df in zip(range(2016, 2022), dfs):
    fig.add_trace(go.Scatter(x=df['Invoice Month'], y=df['Units Sold'], mode='lines', name=str(year)))

fig.update_layout(title='Units Sold Over 10 Years', xaxis_title='Invoice Month', yaxis_title='Units Sold')
fig.show()