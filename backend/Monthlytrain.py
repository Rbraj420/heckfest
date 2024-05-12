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

#got the error resolved here
X_train = train_data[['Invoice Month']]  
y_train = train_data['Units Sold']

X_test = test_data[['Invoice Month']]  
y_test = test_data['Units Sold']

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# Convert the multiple csv files into single dataframe
files = ['2016.csv', '2017.csv', '2018.csv', '2019.csv', '2020.csv', '2021.csv']
dfs = [pd.read_csv(file) for file in files]
data = pd.concat(dfs)

decide= int(input("Enter 0 if you want to enter month and predict the sold units or 1 if you want to know which month was having the max sold unit or if you want to know which month was having the best average sold units."))
if decide==0:
    while True:
        user_month = int(input("Enter the month on which you want (1-12): "))
        if user_month > 12 or user_month < 1: 
            print("Error: Invalid input. Please enter a month between 1 and 12.")
        else: 
            break

    user_month_df = pd.DataFrame({'Invoice Month': [user_month]})
    prediction = model.predict(user_month_df)
    print("Predicted Sells are:", prediction[0])  

    # Calculation of the error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("Mean Absolute Percentage Error:", mape)

elif decide==1:
    
    # Find the peak units sold for each month
    peak_units_per_month = data.groupby('Invoice Month')['Units Sold'].max()

    # Identify the month with the highest peak units sold
    best_month = peak_units_per_month.idxmax()
    peak_units = peak_units_per_month.max()

    print("Best month for investment (based on peak units sold):")
    print("According to max sold units Month:", best_month)
    print("Peak Units Sold:", peak_units)

else:
    # Calculate the average units sold for each month across all years
    average_units_per_month = data.groupby('Invoice Month')['Units Sold'].mean()

    # Identify the month with the highest average units sold
    best_month = average_units_per_month.idxmax()
    average_units = average_units_per_month.max()

    print("Best month for investment (based on average units sold across years):")
    print("According to average sold units Month:", best_month)
    print("Average Units Sold:", average_units)

# Plotting
fig = go.Figure()

for year, df in zip(range(2016, 2022), dfs):
    fig.add_trace(go.Scatter(x=df['Invoice Month'], y=df['Units Sold'], mode='lines', name=str(year)))

fig.update_layout(title='Units Sold Over 8 Years', xaxis_title='Invoice Month', yaxis_title='Units Sold')
fig.show()